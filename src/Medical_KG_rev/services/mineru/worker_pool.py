from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import structlog

from Medical_KG_rev.config.settings import MineruSettings
from Medical_KG_rev.services.gpu.manager import GpuManager

from .cli_wrapper import MineruCliBase, create_cli
from .gpu_manager import MineruGpuAllocation, MineruGpuManager
from .output_parser import MineruOutputParser
from .service import MineruProcessor, MineruRequest, MineruResponse

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class Worker:
    worker_id: int
    gpu_allocation: MineruGpuAllocation
    processor: MineruProcessor

    def process(self, request: MineruRequest) -> MineruResponse:
        logger.debug(
            "mineru.worker.process",
            worker_id=self.worker_id,
            document_id=request.document_id,
        )
        return self.processor.process(request)


class WorkerPool:
    """Threaded worker pool coordinating MinerU processors."""

    def __init__(
        self,
        settings: MineruSettings,
        gpu_manager: GpuManager,
        *,
        cli_factory: Callable[[MineruSettings], MineruCliBase] = create_cli,
    ) -> None:
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=settings.workers.count)
        self._tasks: "queue.Queue[Future[MineruResponse]]" = queue.Queue()
        mineru_gpu = MineruGpuManager(gpu_manager, settings)
        mineru_gpu.ensure_cuda_version()
        self._workers: list[Worker] = []
        for worker_id in range(settings.workers.count):
            allocation = mineru_gpu.allocate()
            processor = MineruProcessor(
                gpu_manager,
                cli=cli_factory(settings),
                parser=MineruOutputParser(),
                min_memory_mb=allocation.vram_limit_mb,
                worker_id=f"worker-{worker_id}",
                fail_fast=False,
            )
            self._workers.append(
                Worker(worker_id=worker_id, gpu_allocation=allocation, processor=processor)
            )
        self._lock = threading.Lock()
        self._round_robin = 0

    def submit(self, request: MineruRequest) -> Future[MineruResponse]:
        with self._lock:
            worker = self._workers[self._round_robin]
            self._round_robin = (self._round_robin + 1) % len(self._workers)
        future = self._executor.submit(worker.process, request)
        self._tasks.put(future)
        return future

    def shutdown(self, wait: bool = True) -> None:
        logger.info("mineru.worker_pool.shutdown")
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        while not self._tasks.empty():
            future = self._tasks.get_nowait()
            if not future.done():
                future.cancel()


__all__ = ["Worker", "WorkerPool"]
