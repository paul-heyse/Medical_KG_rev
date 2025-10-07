from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.services.mineru.cli_wrapper import (
    MineruCliError,
    SimulatedMineruCli,
)
from Medical_KG_rev.services.mineru.metrics import MINERU_GPU_MEMORY_USAGE_BYTES
from Medical_KG_rev.services.mineru.service import MineruProcessor
from Medical_KG_rev.services.mineru.types import (
    Document,
    MineruRequest,
    MineruResponse,
    ProcessingMetadata,
)
from Medical_KG_rev.services.mineru.worker_pool import ProcessorProtocol, WorkerPool
from Medical_KG_rev.storage import InMemoryCache

from tests.integration._mineru_test_utils import FakeGpuManager, build_mineru_settings


class CountingSimulatedCli(SimulatedMineruCli):
    def __init__(self, settings, *, fail_first: bool = False) -> None:
        super().__init__(settings)
        self.invocations = 0
        self.fail_first = fail_first

    def run_batch(self, batch, *, gpu_id: int):
        self.invocations += 1
        if self.fail_first and self.invocations == 1 and gpu_id >= 0:
            raise MineruCliError("CUDA out of memory", stderr="CUDA out of memory")
        return super().run_batch(batch, gpu_id=gpu_id)


def _build_processor(*, cli=None, settings_override=None, cache=None) -> MineruProcessor:
    settings = build_mineru_settings(**(settings_override or {}))
    cli_instance = cli or CountingSimulatedCli(settings)
    return MineruProcessor(
        FakeGpuManager(),
        settings=settings,
        cli=cli_instance,
        cache=cache,
        fail_fast=False,
    )


def test_benchmark_processing_time_per_page():
    processor = _build_processor()
    request = MineruRequest(
        tenant_id="tenant",
        document_id="benchmark-doc",
        content=("Page 1 text\fPage 2 content\fPage 3 tables | values").encode("utf-8"),
    )
    start = time.monotonic()
    response = processor.process(request)
    elapsed = time.monotonic() - start
    assert elapsed < 0.25, f"Simulated CLI should complete quickly, took {elapsed:.3f}s"
    assert len(response.document.blocks) >= 3


def test_gpu_memory_usage_metric_populated():
    MINERU_GPU_MEMORY_USAGE_BYTES.labels(gpu_id="cuda:0", state="required").set(0)
    processor = _build_processor()
    request = MineruRequest("tenant", "metric-doc", b"Example table | values")
    processor.process(request)
    required_bytes = MINERU_GPU_MEMORY_USAGE_BYTES.labels(gpu_id="cuda:0", state="required")._value.get()
    expected = processor.settings.workers.vram_per_worker_mb * 1024 * 1024
    assert required_bytes == pytest.approx(expected)


class _FastProcessor(ProcessorProtocol):
    def __init__(self, worker_id: str, sleep_seconds: float = 0.05) -> None:
        self.worker_id = worker_id
        self.sleep_seconds = sleep_seconds

    def process(self, request: MineruRequest) -> MineruResponse:
        time.sleep(self.sleep_seconds)
        now = datetime.now(timezone.utc)
        document = Document(
            document_id=request.document_id,
            tenant_id=request.tenant_id,
            blocks=[],
            tables=[],
            figures=[],
            equations=[],
            metadata={"worker": self.worker_id},
        )
        metadata = ProcessingMetadata(
            document_id=request.document_id,
            mineru_version="simulated",
            model_names={},
            gpu_id=self.worker_id,
            worker_id=self.worker_id,
            started_at=now,
            completed_at=now,
            duration_seconds=self.sleep_seconds,
            cli_stdout="",
            cli_stderr="",
            cli_descriptor="fast",
        )
        return MineruResponse(document=document, processed_at=now, duration_seconds=self.sleep_seconds, metadata=metadata)


def _fast_processor_factory(settings, allocation, worker_id: str) -> ProcessorProtocol:
    return _FastProcessor(worker_id)


def test_worker_pool_throughput():
    settings = build_mineru_settings(workers={"count": 2, "batch_size": 4})
    pool = WorkerPool(settings, FakeGpuManager(), processor_factory=_fast_processor_factory)
    try:
        futures = [
            pool.submit(MineruRequest("tenant", f"doc-{index}", b"payload"))
            for index in range(6)
        ]
        responses = [future.result(timeout=5) for future in futures]
    finally:
        pool.shutdown()
    worker_ids = {response.metadata.worker_id for response in responses}
    assert len(worker_ids) >= 2


def test_cpu_fallback_on_oom():
    settings = build_mineru_settings()
    cli = CountingSimulatedCli(settings, fail_first=True)
    processor = MineruProcessor(FakeGpuManager(), settings=settings, cli=cli, fail_fast=False)
    request = MineruRequest("tenant", "oom-doc", b"Row | Value")
    response = processor.process(request)
    assert cli.invocations >= 2, "Processor should retry after OOM"
    assert response.metadata.gpu_id == "cpu"


def test_cache_hit_rate_and_latency():
    settings = build_mineru_settings(cache={"enabled": True, "backend": "memory", "namespace": "mineru-test"})
    cli = CountingSimulatedCli(settings)
    processor = MineruProcessor(
        FakeGpuManager(),
        settings=settings,
        cli=cli,
        cache=InMemoryCache(),
        fail_fast=False,
    )
    request1 = MineruRequest("tenant", "cache-doc", b"Cache me | yes")
    processor.process(request1)
    request2 = MineruRequest("tenant", "cache-doc", b"Cache me | yes")
    processor.process(request2)
    assert cli.invocations == 1, "Second invocation should be served from cache"
