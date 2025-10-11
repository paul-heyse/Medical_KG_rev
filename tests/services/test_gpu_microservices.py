import asyncio
import importlib
import sys
import types

import pytest

pytest.importorskip("pydantic")

# MinerU settings removed - using Docling VLM services instead


def _build_fake_torch():
    class _Props:
        name = "Fake GPU"
        total_memory = 4 * 1024 * 1024 * 1024

    class _DeviceCtx:
        def __init__(self, index: int):
            self.index = index

        def __enter__(self):
            _Cuda._current_device = self.index
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Cuda:
        _current_device = 0

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_properties(index: int):
            return _Props()

        @staticmethod
        def device(index: int):
            return _DeviceCtx(index)

        @staticmethod
        def mem_get_info():
            total = _Props.total_memory
            free = total - (256 * 1024 * 1024)
            return free, total

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return _Cuda._current_device

        @staticmethod
        def set_device(index: int) -> None:
            _Cuda._current_device = index

        @staticmethod
        def synchronize() -> None:
            pass

        @staticmethod
        def memory_allocated(index: int) -> int:
            return 128 * 1024 * 1024

        @staticmethod
        def utilization(index: int) -> float:
            return 25.0

        @staticmethod
        def memory_reserved(index: int) -> int:  # pragma: no cover - used via hasattr
            return 256 * 1024 * 1024

    fake = types.SimpleNamespace(cuda=_Cuda, version=types.SimpleNamespace(cuda="12.8"))
    fake.__spec__ = types.SimpleNamespace(loader=None)  # type: ignore[attr-defined]
    return fake


@pytest.fixture(scope="module")
def gpu_env():
    fake = _build_fake_torch()
    original = sys.modules.get("torch")
    sys.modules["torch"] = fake
    import Medical_KG_rev.services.gpu.manager as gpu_manager

    importlib.reload(gpu_manager)
    yield types.SimpleNamespace(gpu_manager=gpu_manager)
    if original is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = original
    importlib.reload(gpu_manager)


@pytest.fixture(scope="module")
def microservice_modules(gpu_env):
    import Medical_KG_rev.services.embedding.service as embedding_service
    import Medical_KG_rev.services.extraction.service as extraction_service
    import Medical_KG_rev.services.mineru.service as mineru_service

    importlib.reload(mineru_service)
    importlib.reload(embedding_service)
    importlib.reload(extraction_service)

    return types.SimpleNamespace(
        mineru=mineru_service,
        embedding=embedding_service,
        extraction=extraction_service,
        gpu_manager=gpu_env.gpu_manager,
    )


def test_gpu_manager_detects_device(gpu_env):
    manager = gpu_env.gpu_manager.GpuManager(min_memory_mb=256)
    device = manager.get_device()
    assert device.index == 0
    assert device.total_memory_mb >= 1024


def test_gpu_manager_failfast(gpu_env):
    module = gpu_env.gpu_manager
    original = module.torch
    module.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    manager = module.GpuManager()
    with pytest.raises(module.GpuNotAvailableError):
        manager.get_device()
    module.torch = original


def test_mineru_processor_generates_blocks(microservice_modules):
    processor = microservice_modules.mineru.MineruProcessor(
        microservice_modules.gpu_manager.GpuManager()
    )
    request = microservice_modules.mineru.MineruRequest(
        tenant_id="tenant-1",
        document_id="doc-1",
        content=b"Population of patients\nIntervention|Dose\nOutcome improved",
    )
    response = processor.process(request)
    assert response.document.document_id == "doc-1"
    assert any(block.kind == "table" for block in response.document.blocks)
    assert all(block.text for block in response.document.blocks)
    assert response.metadata.document_id == "doc-1"
    assert response.metadata.worker_id
    assert response.document.provenance["document_id"] == "doc-1"


def test_mineru_processor_respects_batch_size(microservice_modules):
    settings = MineruSettings(
        enabled=True,
        simulate_if_unavailable=True,
        cli_command="mineru-missing",
        workers=MineruWorkerSettings(count=1, batch_size=1, device_ids=[0], vram_per_worker_gb=1),
    )
    processor = microservice_modules.mineru.MineruProcessor(
        microservice_modules.gpu_manager.GpuManager(
            min_memory_mb=settings.workers.vram_per_worker_mb
        ),
        settings=settings,
        fail_fast=False,
    )
    requests = [
        microservice_modules.mineru.MineruRequest(
            tenant_id="tenant-1",
            document_id="doc-1",
            content=b"Population data\nDose|Value",
        ),
        microservice_modules.mineru.MineruRequest(
            tenant_id="tenant-1",
            document_id="doc-2",
            content=b"Secondary arm\nMetric|Result",
        ),
    ]

    batch = processor.process_batch(requests)

    assert len(batch.documents) == 2
    assert [doc.document_id for doc in batch.documents] == ["doc-1", "doc-2"]
    assert len(batch.metadata) == 2
    assert batch.duration_seconds >= 0.0


def test_extraction_service_validates_spans(microservice_modules):
    service = microservice_modules.extraction.ExtractionService(
        microservice_modules.gpu_manager.GpuManager()
    )
    text = "The population of patients received a treatment with positive outcome."
    result = service.extract(
        microservice_modules.extraction.ExtractionInput(
            tenant_id="tenant", document_id="doc", text=text, kind="pico"
        )
    )
    labels = {span.label for span in result.spans}
    assert "population" in labels
    assert "intervention" in labels or "outcome" in labels


def test_grpc_state_and_interceptors(monkeypatch):
    import Medical_KG_rev.services.grpc.server as grpc_server

    importlib.reload(grpc_server)
    state = grpc_server.GrpcServiceState("TestService")

    called = []

    class DummySpan:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            called.append(self.name)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_attribute(self, key, value):
            return None

    class DummyTracer:
        def start_as_current_span(self, name):
            return DummySpan(name)

    monkeypatch.setattr(
        grpc_server, "trace", types.SimpleNamespace(get_tracer=lambda *_: DummyTracer())
    )

    async def continuation(details):
        return "ok"

    interceptor = grpc_server.UnaryUnaryTracingInterceptor("TestService")
    details = types.SimpleNamespace(method="/test.Service/Method")

    async def exercise():
        result = await interceptor.intercept_service(continuation, details)
        assert result == "ok"
        assert called == ["/test.Service/Method"]

        logging_interceptor = grpc_server.UnaryUnaryLoggingInterceptor("TestService")
        await logging_interceptor.intercept_service(continuation, details)

        state.set_ready()
        await state.wait_until_ready(timeout=0.1)

    asyncio.run(exercise())
