"""Unit tests for the MinerU worker pool orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import get_context

import pytest

from Medical_KG_rev.config.settings import MineruSettings
from Medical_KG_rev.services.gpu.manager import GpuDevice
from Medical_KG_rev.services.mineru.worker_pool import ProcessorProtocol, WorkerPool
from Medical_KG_rev.services.mineru.types import (
    Document,
    MineruRequest,
    MineruResponse,
    ProcessingMetadata,
)


CTX = get_context("spawn")
COUNTER = CTX.Value("i", 0)


@dataclass
class _StubGpuManager:
    """Minimal GPU manager compatible with :class:`MineruGpuManager`."""

    device: GpuDevice = GpuDevice(index=0, name="stub-gpu", total_memory_mb=32768)

    def get_device(self) -> GpuDevice:  # pragma: no cover - trivial accessor
        return self.device


class _RecordingProcessor:
    """Processor that records processing order for assertions."""

    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id

    def process(self, request: MineruRequest) -> MineruResponse:
        from datetime import datetime, timezone

        with COUNTER.get_lock():
            COUNTER.value += 1
            order = COUNTER.value
        now = datetime.now(timezone.utc)
        document = Document(
            document_id=request.document_id,
            tenant_id=request.tenant_id,
            blocks=[],
            tables=[],
            figures=[],
            equations=[],
            metadata={"order": order},
            provenance={"worker_id": self.worker_id},
        )
        metadata = ProcessingMetadata(
            document_id=request.document_id,
            mineru_version="test",
            model_names={},
            gpu_id=self.worker_id,
            worker_id=self.worker_id,
            started_at=now,
            completed_at=now,
            duration_seconds=0.0,
            cli_stdout=str(order),
            cli_stderr="",
            cli_descriptor="recording",
        )
        return MineruResponse(
            document=document,
            processed_at=now,
            duration_seconds=0.0,
            metadata=metadata,
        )


@dataclass
class _StubKafkaMessage:
    topic: str
    value: dict[str, object]
    key: str | None = None
    headers: dict[str, str] | None = None
    attempts: int = 0
    available_at: float = 0.0

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = {}
        if not self.available_at:
            self.available_at = time.time()


class _StubKafkaClient:
    """Simple in-memory Kafka façade used for unit testing."""

    def __init__(self) -> None:
        self._topics: dict[str, list[_StubKafkaMessage]] = {}

    def create_topics(self, topics) -> None:
        for topic in topics:
            self._topics.setdefault(str(topic), [])

    def publish(
        self,
        topic: str,
        value: dict[str, object],
        *,
        key: str | None = None,
        headers: dict[str, str] | None = None,
        available_at: float | None = None,
        attempts: int = 0,
    ) -> _StubKafkaMessage:
        message = _StubKafkaMessage(
            topic=topic,
            value=value,
            key=key,
            headers=headers,
            attempts=attempts,
            available_at=available_at or time.time(),
        )
        self._topics.setdefault(topic, []).append(message)
        return message

    def consume(self, topic: str, *, max_messages: int | None = None):
        queue = self._topics.get(topic, [])
        if not queue:
            return iter(())
        now = time.time()
        yielded = 0
        ready: list[_StubKafkaMessage] = []
        pending: list[_StubKafkaMessage] = []
        for message in queue:
            if message.available_at > now:
                pending.append(message)
                continue
            if max_messages is not None and yielded >= max_messages:
                pending.append(message)
                continue
            ready.append(message)
            yielded += 1
        self._topics[topic] = pending
        return iter(ready)


def _processor_factory(settings: MineruSettings, allocation, worker_id: str) -> ProcessorProtocol:
    return _RecordingProcessor(worker_id)


def _build_settings(worker_count: int = 1) -> MineruSettings:
    return MineruSettings(
        enabled=True,
        simulate_if_unavailable=True,
        workers={
            "count": worker_count,
            "vram_per_worker_gb": 1,
            "timeout_seconds": 60,
            "batch_size": 4,
        },
    )


@pytest.fixture(autouse=True)
def _reset_counter():
    with COUNTER.get_lock():
        COUNTER.value = 0
    yield
    with COUNTER.get_lock():
        COUNTER.value = 0


def test_worker_pool_respects_priority_order():
    settings = _build_settings(worker_count=1)
    gpu_manager = _StubGpuManager()
    pool = WorkerPool(settings, gpu_manager, processor_factory=_processor_factory)
    try:
        low_1 = pool.submit(MineruRequest("tenant", "doc-low-1", b""), priority=0)
        time.sleep(0.1)
        low_2 = pool.submit(MineruRequest("tenant", "doc-low-2", b""), priority=0)
        high = pool.submit(MineruRequest("tenant", "doc-high", b""), priority=5)

        responses = [low_1.result(timeout=5), low_2.result(timeout=5), high.result(timeout=5)]
        orders = {response.document.document_id: int(response.metadata.cli_stdout) for response in responses}
        assert orders["doc-low-1"] == 1
        assert orders["doc-high"] == 2
        assert orders["doc-low-2"] == 3
    finally:
        pool.shutdown()


def test_worker_pool_consumes_from_kafka():
    settings = _build_settings(worker_count=1)
    gpu_manager = _StubGpuManager()
    pool = WorkerPool(settings, gpu_manager, processor_factory=_processor_factory)
    kafka = _StubKafkaClient()
    topic = "pdf.parse.requests.v1"
    try:
        kafka.create_topics([topic])
        kafka.publish(
            topic,
            value={"tenant_id": "tenant", "document_id": "doc-kafka", "content": b"payload"},
            headers={"x-priority": "2"},
        )

        pool.consume_from_kafka(kafka, topic, poll_interval=0.1, batch_size=1)

        deadline = time.time() + 5
        while time.time() < deadline:
            with COUNTER.get_lock():
                if COUNTER.value >= 1:
                    break
            time.sleep(0.1)
        else:
            pytest.fail("Kafka message was not processed in time")
    finally:
        pool.shutdown()
