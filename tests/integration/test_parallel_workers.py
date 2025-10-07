from __future__ import annotations

import time
from datetime import datetime, timezone
from multiprocessing import get_context

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.services.mineru.types import (
    Document,
    MineruRequest,
    MineruResponse,
    ProcessingMetadata,
)
from Medical_KG_rev.services.mineru.worker_pool import ProcessorProtocol, WorkerPool

from ._mineru_test_utils import FakeGpuManager, build_mineru_settings


CTX = get_context("spawn")
COUNTER = CTX.Value("i", 0)


class _SleepyProcessor(ProcessorProtocol):
    def __init__(self, worker_id: str, sleep_seconds: float = 0.15) -> None:
        self.worker_id = worker_id
        self.sleep_seconds = sleep_seconds

    def process(self, request: MineruRequest) -> MineruResponse:
        time.sleep(self.sleep_seconds)
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
        )
        metadata = ProcessingMetadata(
            document_id=request.document_id,
            mineru_version="simulated",
            model_names={},
            gpu_id=f"simulated:{self.worker_id}",
            worker_id=self.worker_id,
            started_at=now,
            completed_at=now,
            duration_seconds=self.sleep_seconds,
            cli_stdout=str(order),
            cli_stderr="",
            cli_descriptor="sleepy",
        )
        return MineruResponse(document=document, processed_at=now, duration_seconds=self.sleep_seconds, metadata=metadata)


def _processor_factory(settings, allocation, worker_id: str) -> ProcessorProtocol:
    return _SleepyProcessor(worker_id)


@pytest.fixture(autouse=True)
def _reset_counter():
    with COUNTER.get_lock():
        COUNTER.value = 0
    yield
    with COUNTER.get_lock():
        COUNTER.value = 0


def test_parallel_workers_process_batches_concurrently():
    settings = build_mineru_settings(workers={"count": 2, "batch_size": 4})
    pool = WorkerPool(settings, FakeGpuManager(), processor_factory=_processor_factory)
    try:
        futures = [
            pool.submit(MineruRequest("tenant", f"doc-{index}", b"payload"))
            for index in range(4)
        ]
        responses = [future.result(timeout=5) for future in futures]
    finally:
        pool.shutdown()

    worker_ids = {response.metadata.worker_id for response in responses}
    assert len(worker_ids) >= 2, "expected work to be spread across workers"
