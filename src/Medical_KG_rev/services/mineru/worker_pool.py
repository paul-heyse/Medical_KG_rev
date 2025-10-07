"""Parallel worker pool for orchestrating MinerU CLI execution."""

from __future__ import annotations

import heapq
import os
import queue
import signal
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from multiprocessing import Event as MpEvent, Process, Queue as MpQueue, get_context
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from Medical_KG_rev.config.settings import MineruSettings
from Medical_KG_rev.services.gpu.manager import GpuManager, GpuNotAvailableError

from .gpu_manager import MineruGpuAllocation, MineruGpuManager
from .metrics import MINERU_WORKER_QUEUE_DEPTH
from .types import Document, MineruRequest, MineruResponse, ProcessingMetadata

if TYPE_CHECKING:  # pragma: no cover - typing only
    from Medical_KG_rev.orchestration.kafka import KafkaClient, KafkaMessage

logger = structlog.get_logger(__name__)


class ProcessorProtocol(Protocol):
    """Protocol implemented by processors executed within worker processes."""

    def process(self, request: MineruRequest) -> MineruResponse:  # pragma: no cover - protocol
        ...


def _default_processor_factory(
    settings: MineruSettings, allocation: MineruGpuAllocation, worker_id: str
) -> ProcessorProtocol:
    """Instantiate a :class:`MineruProcessor` inside a worker process."""

    gpu_manager = GpuManager(
        min_memory_mb=settings.workers.vram_per_worker_mb,
        preferred_device=allocation.device.index,
    )
    from .service import MineruProcessor

    return MineruProcessor(
        gpu_manager,
        settings=settings,
        min_memory_mb=allocation.vram_limit_mb,
        worker_id=worker_id,
    )


@dataclass(slots=True)
class _SimulatedProcessor:
    """Fallback processor used when GPUs are unavailable but simulation enabled."""

    settings: MineruSettings
    worker_id: str
    reason: str

    def process(self, request: MineruRequest) -> MineruResponse:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        document = Document(
            document_id=request.document_id,
            tenant_id=request.tenant_id,
            blocks=[],
            tables=[],
            figures=[],
            equations=[],
            metadata={"simulated": True, "reason": self.reason},
            provenance={"worker_id": self.worker_id},
        )
        metadata = ProcessingMetadata(
            document_id=request.document_id,
            mineru_version=None,
            model_names={},
            gpu_id="simulated",
            worker_id=self.worker_id,
            started_at=now,
            completed_at=now,
            duration_seconds=0.0,
            cli_stdout="simulated",
            cli_stderr=self.reason,
            cli_descriptor="simulated",
        )
        return MineruResponse(
            document=document,
            processed_at=now,
            duration_seconds=0.0,
            metadata=metadata,
        )


def _export_cpu_environment(settings: MineruSettings) -> dict[str, str]:
    env = settings.cpu.export_environment()
    return {str(key): str(value) for key, value in env.items()}


def _worker_loop(
    worker_id: str,
    settings_payload: dict[str, Any],
    allocation: MineruGpuAllocation,
    task_queue: MpQueue,
    result_queue: MpQueue,
    stop_event: MpEvent,
    processor_factory: Callable[[MineruSettings, MineruGpuAllocation, str], ProcessorProtocol],
    simulation_mode: bool,
) -> None:
    """Entry point executed within the worker subprocess."""

    settings = MineruSettings.model_validate(settings_payload)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(allocation.device.index))
    for key, value in _export_cpu_environment(settings).items():
        os.environ.setdefault(key, value)

    processor: ProcessorProtocol
    try:
        processor = processor_factory(settings, allocation, worker_id)
    except GpuNotAvailableError as exc:
        if not (settings.simulate_if_unavailable or simulation_mode):
            result_queue.put(("fatal", worker_id, None, (type(exc).__name__, str(exc), traceback.format_exc())))
            return
        logger.warning(
            "mineru.worker.simulated",
            worker_id=worker_id,
            reason=str(exc),
        )
        processor = _SimulatedProcessor(settings=settings, worker_id=worker_id, reason=str(exc))
    except Exception as exc:  # pragma: no cover - defensive, surfaced via result queue
        result_queue.put(("fatal", worker_id, None, (type(exc).__name__, str(exc), traceback.format_exc())))
        return

    heartbeat_interval = 5.0
    next_heartbeat = time.monotonic() + heartbeat_interval

    while not stop_event.is_set():
        try:
            job = task_queue.get(timeout=1.0)
        except queue.Empty:
            now = time.monotonic()
            if now >= next_heartbeat:
                result_queue.put(("heartbeat", worker_id, None, now))
                next_heartbeat = now + heartbeat_interval
            continue

        if job is None:
            break

        job_id, request = job
        try:
            response = processor.process(request)
        except Exception as exc:  # pragma: no cover - surfaced to supervising thread
            result_queue.put(
                (
                    "error",
                    worker_id,
                    job_id,
                    (type(exc).__name__, str(exc), traceback.format_exc()),
                )
            )
            continue

        result_queue.put(("ok", worker_id, job_id, response))

    result_queue.put(("stopped", worker_id, None, time.monotonic()))


class MineruWorkerProcess(Process):
    """Wrapper around :func:`_worker_loop` simplifying process management."""

    def __init__(
        self,
        *,
        worker_id: str,
        settings: MineruSettings,
        allocation: MineruGpuAllocation,
        result_queue: MpQueue,
        processor_factory: Callable[[MineruSettings, MineruGpuAllocation, str], ProcessorProtocol],
        simulation_mode: bool,
    ) -> None:
        ctx = get_context("spawn")
        super().__init__(
            name=worker_id,
            daemon=True,
            target=_worker_loop,
            args=(
                worker_id,
                settings.model_dump(mode="json"),
                allocation,
                ctx.Queue(),
                result_queue,
                ctx.Event(),
                processor_factory,
                simulation_mode,
            ),
        )
        # The spawn context creates fresh objects, we stash references for external control.
        self._task_queue: MpQueue = self._args[3]
        self._stop_event: MpEvent = self._args[5]
        self.allocation = allocation

    @property
    def task_queue(self) -> MpQueue:
        return self._task_queue

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._task_queue.put_nowait(None)
        except Exception:  # pragma: no cover - queue may be closed during shutdown
            pass


@dataclass(slots=True)
class _Job:
    priority: int
    sequence: int
    job_id: str
    request: MineruRequest


@dataclass(slots=True)
class _WorkerHandle:
    process: MineruWorkerProcess
    allocation: MineruGpuAllocation
    busy: bool = False
    last_heartbeat: float = field(default_factory=lambda: time.monotonic())


class WorkerPool:
    """Multiprocessing-based worker pool with Kafka integration and health checks."""

    def __init__(
        self,
        settings: MineruSettings,
        gpu_manager: GpuManager,
        *,
        processor_factory: Callable[[MineruSettings, MineruGpuAllocation, str], ProcessorProtocol] = _default_processor_factory,
    ) -> None:
        self._settings = settings
        self._processor_factory = processor_factory
        self._simulation_mode = False
        self._result_queue: MpQueue = get_context("spawn").Queue()
        self._stop = threading.Event()
        self._dispatch_event = threading.Event()
        self._lock = threading.Lock()
        self._sequence = 0
        self._pending: list[tuple[int, int, str]] = []
        self._payloads: dict[str, _Job] = {}
        self._futures: dict[str, Future[MineruResponse]] = {}
        self._inflight: dict[str, str] = {}
        self._workers: dict[str, _WorkerHandle] = {}
        self._idle: set[str] = set()
        self._consumer_thread: threading.Thread | None = None
        self._consumer_stop = threading.Event()

        mineru_gpu = MineruGpuManager(gpu_manager, settings)
        self._mineru_gpu = mineru_gpu
        try:
            mineru_gpu.ensure_cuda_version()
        except GpuNotAvailableError as exc:
            if not settings.simulate_if_unavailable:
                raise
            self._simulation_mode = True
            logger.warning("mineru.worker_pool.cuda_unavailable", error=str(exc))

        for index in range(settings.workers.count):
            worker_id = f"worker-{index}"
            handle = self._spawn_worker(worker_id)
            self._workers[worker_id] = handle
            self._idle.add(worker_id)
            MINERU_WORKER_QUEUE_DEPTH.labels(worker_id=worker_id).set(0)

        MINERU_WORKER_QUEUE_DEPTH.labels(worker_id="pending").set(0)

        self._result_thread = threading.Thread(target=self._result_loop, name="mineru-result", daemon=True)
        self._result_thread.start()
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, name="mineru-dispatch", daemon=True)
        self._dispatch_thread.start()
        self._health_thread = threading.Thread(target=self._health_loop, name="mineru-health", daemon=True)
        self._health_thread.start()

        try:
            signal.signal(signal.SIGTERM, self._handle_sigterm)
        except ValueError:  # pragma: no cover - signal registration fails in non-main threads
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def submit(self, request: MineruRequest, *, priority: int = 0) -> Future[MineruResponse]:
        if self._stop.is_set():
            raise RuntimeError("MinerU worker pool has been shut down")
        job_id = uuid.uuid4().hex
        future: Future[MineruResponse] = Future()
        job = _Job(priority=priority, sequence=self._sequence, job_id=job_id, request=request)
        self._sequence += 1
        with self._lock:
            heapq.heappush(self._pending, (-job.priority, job.sequence, job.job_id))
            self._payloads[job.job_id] = job
            self._futures[job.job_id] = future
            MINERU_WORKER_QUEUE_DEPTH.labels(worker_id="pending").set(len(self._pending))
        self._dispatch_event.set()
        return future

    def consume_from_kafka(
        self,
        kafka: "KafkaClient",
        topic: str,
        *,
        poll_interval: float = 1.0,
        batch_size: int | None = None,
    ) -> None:
        if self._consumer_thread and self._consumer_thread.is_alive():
            raise RuntimeError("Kafka consumer already running")
        kafka.create_topics([topic])
        batch = batch_size or self._settings.workers.batch_size
        self._consumer_stop.clear()
        self._consumer_thread = threading.Thread(
            target=self._kafka_loop,
            name="mineru-kafka",
            daemon=True,
            args=(kafka, topic, poll_interval, batch),
        )
        self._consumer_thread.start()

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                worker_id: {
                    "alive": handle.process.is_alive(),
                    "busy": handle.busy,
                    "last_heartbeat": handle.last_heartbeat,
                }
                for worker_id, handle in self._workers.items()
            }

    def shutdown(self, timeout: float | None = None) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._dispatch_event.set()
        self._consumer_stop.set()
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=timeout)

        for worker_id, handle in self._workers.items():
            handle.process.stop()
        for worker_id, handle in self._workers.items():
            handle.process.join(timeout=timeout)
            if handle.process.is_alive():  # pragma: no cover - defensive, ensures clean shutdown
                handle.process.terminate()
            MINERU_WORKER_QUEUE_DEPTH.labels(worker_id=worker_id).set(0)
            self._mineru_gpu.release_worker(worker_id)

        self._result_thread.join(timeout=timeout)
        self._dispatch_thread.join(timeout=timeout)
        self._health_thread.join(timeout=timeout)
        MINERU_WORKER_QUEUE_DEPTH.labels(worker_id="pending").set(0)

        with self._lock:
            for future in self._futures.values():
                if not future.done():
                    future.cancel()
            self._futures.clear()
            self._pending.clear()
            self._payloads.clear()
            self._inflight.clear()
            self._idle.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _handle_sigterm(self, signum, frame):  # pragma: no cover - signal-based
        logger.info("mineru.worker_pool.sigterm", signal=signum)
        self.shutdown()

    def _spawn_worker(self, worker_id: str) -> _WorkerHandle:
        allocation = self._mineru_gpu.allocate_for_worker(worker_id)
        process = MineruWorkerProcess(
            worker_id=worker_id,
            settings=self._settings,
            allocation=allocation,
            result_queue=self._result_queue,
            processor_factory=self._processor_factory,
            simulation_mode=self._simulation_mode,
        )
        process.start()
        return _WorkerHandle(process=process, allocation=allocation)

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            self._dispatch_event.wait(timeout=0.5)
            self._dispatch_event.clear()
            self._assign_jobs()

    def _assign_jobs(self) -> None:
        with self._lock:
            while self._pending and self._idle:
                _, _, job_id = heapq.heappop(self._pending)
                job = self._payloads[job_id]
                worker_id = self._idle.pop()
                handle = self._workers[worker_id]
                try:
                    handle.process.task_queue.put_nowait((job.job_id, job.request))
                except Exception as exc:
                    logger.error("mineru.worker.enqueue_failed", worker_id=worker_id, error=str(exc))
                    heapq.heappush(self._pending, (-job.priority, job.sequence, job.job_id))
                    self._idle.add(worker_id)
                    break
                handle.busy = True
                MINERU_WORKER_QUEUE_DEPTH.labels(worker_id=worker_id).set(1)
                self._inflight[worker_id] = job.job_id
            MINERU_WORKER_QUEUE_DEPTH.labels(worker_id="pending").set(len(self._pending))

    def _result_loop(self) -> None:
        while not self._stop.is_set():
            try:
                status, worker_id, job_id, payload = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            handle = self._workers.get(worker_id)
            if handle is None:
                continue  # pragma: no cover - defensive guard for shutdown race

            if status == "heartbeat":
                handle.last_heartbeat = float(payload)
                continue

            if status == "stopped":
                handle.busy = False
                MINERU_WORKER_QUEUE_DEPTH.labels(worker_id=worker_id).set(0)
                continue

            if job_id is not None:
                self._inflight.pop(worker_id, None)
                job = self._payloads.pop(job_id, None)
                future = self._futures.pop(job_id, None)
            else:
                job = None
                future = None

            if status == "ok" and future is not None:
                future.set_result(payload)
            elif status == "error" and future is not None:
                exc_type, message, tb = payload
                future.set_exception(RuntimeError(f"{exc_type}: {message}\n{tb}"))
                if "OutOfMemory" in exc_type:
                    self._mineru_gpu.record_oom(worker_id, RuntimeError(message))
            elif status == "fatal":
                logger.error("mineru.worker.fatal", worker_id=worker_id, error=payload)

            handle.busy = False
            MINERU_WORKER_QUEUE_DEPTH.labels(worker_id=worker_id).set(0)
            if not self._stop.is_set():
                self._idle.add(worker_id)
                self._dispatch_event.set()

    def _health_loop(self) -> None:
        check_interval = 5.0
        while not self._stop.is_set():
            time.sleep(check_interval)
            self._check_workers()

    def _check_workers(self) -> None:
        with self._lock:
            for worker_id, handle in list(self._workers.items()):
                if not handle.process.is_alive() and not self._stop.is_set():
                    logger.warning("mineru.worker.restarting", worker_id=worker_id)
                    inflight = self._inflight.pop(worker_id, None)
                    if inflight:
                        job = self._payloads.get(inflight)
                        if job:
                            heapq.heappush(self._pending, (-job.priority, job.sequence, job.job_id))
                            MINERU_WORKER_QUEUE_DEPTH.labels(worker_id="pending").set(len(self._pending))
                            self._dispatch_event.set()
                    self._mineru_gpu.release_worker(worker_id)
                    new_handle = self._spawn_worker(worker_id)
                    self._workers[worker_id] = new_handle
                    self._idle.add(worker_id)
                    self._dispatch_event.set()
                else:
                    now = time.monotonic()
                    heartbeat_age = now - handle.last_heartbeat
                    if heartbeat_age > self._settings.workers.timeout_seconds:
                        logger.warning(
                            "mineru.worker.stale_heartbeat",
                            worker_id=worker_id,
                            age=heartbeat_age,
                        )

    def _kafka_loop(
        self,
        kafka: "KafkaClient",
        topic: str,
        poll_interval: float,
        batch: int,
    ) -> None:
        while not self._consumer_stop.is_set() and not self._stop.is_set():
            for message in kafka.consume(topic, max_messages=batch):
                self._submit_kafka_message(message)
            time.sleep(poll_interval)

    def _submit_kafka_message(self, message: "KafkaMessage") -> None:
        value = message.value
        tenant_id = str(value.get("tenant_id"))
        document_id = str(value.get("document_id"))
        content = value.get("content", b"")
        if isinstance(content, str):
            content = content.encode("utf-8")
        request = MineruRequest(tenant_id=tenant_id, document_id=document_id, content=content)
        priority = int(message.headers.get("x-priority", "0"))
        future = self.submit(request, priority=priority)

        def _ack_result(fut: Future[MineruResponse]) -> None:
            try:
                fut.result()
                logger.info(
                    "mineru.worker.kafka_ack",
                    document_id=document_id,
                    tenant_id=tenant_id,
                    priority=priority,
                )
            except Exception as exc:  # pragma: no cover - visible in logs for retries
                available_at = time.time() + 5.0
                kafka.publish(
                    message.topic,
                    value=message.value,
                    key=message.key,
                    headers=message.headers,
                    available_at=available_at,
                    attempts=message.attempts + 1,
                )
                logger.error(
                    "mineru.worker.kafka_retry",
                    document_id=document_id,
                    error=str(exc),
                    attempts=message.attempts + 1,
                )

        future.add_done_callback(_ack_result)


__all__ = ["WorkerPool", "ProcessorProtocol"]

