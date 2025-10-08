from collections.abc import Sequence
from dataclasses import dataclass
import types

import pytest

from Medical_KG_rev.embeddings.namespace import NamespaceManager
from Medical_KG_rev.embeddings.ports import (
    EmbedderConfig,
    EmbeddingRecord,
    EmbeddingRequest as AdapterEmbeddingRequest,
)
from Medical_KG_rev.embeddings.utils.gpu import GPUMemoryInfo
from Medical_KG_rev.embeddings.utils.tokenization import TokenizerCache
from Medical_KG_rev.services import GpuNotAvailableError
from Medical_KG_rev.services.embedding.service import (
    EmbeddingModelRegistry,
    EmbeddingRequest,
    EmbeddingWorker,
)


@dataclass
class RecordingEmbedder:
    config: EmbedderConfig
    responses: list[object]
    calls: list[Sequence[str]]

    def embed_documents(self, request: AdapterEmbeddingRequest):  # type: ignore[override]
        self.calls.append(tuple(request.texts))
        if self.responses:
            effect = self.responses.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(request)
        return _records(self.config, request)

    def embed_queries(self, request: AdapterEmbeddingRequest):  # type: ignore[override]
        return self.embed_documents(request)


def _records(config: EmbedderConfig, request: AdapterEmbeddingRequest) -> list[EmbeddingRecord]:
    vectors = []
    for idx, text in enumerate(request.texts):
        vectors.append(
            EmbeddingRecord(
                id=request.ids[idx] if request.ids else f"chunk-{idx}",
                tenant_id=request.tenant_id,
                namespace=request.namespace,
                model_id=config.model_id,
                model_version=config.model_version,
                kind=config.kind,
                dim=config.dim,
                vectors=[[float(idx), float(len(text))]],
                metadata={"provider": config.provider},
            )
        )
    return vectors


@pytest.fixture()
def worker_setup(monkeypatch: pytest.MonkeyPatch):
    namespace_manager = NamespaceManager()
    config = EmbedderConfig(
        name="qwen3",
        provider="vllm",
        kind="single_vector",
        namespace="single_vector.qwen3.2.v1",
        model_id="Qwen/Qwen2.5-Embedding-8B-Instruct",
        model_version="v1",
        dim=2,
        normalize=True,
        batch_size=4,
        requires_gpu=True,
        parameters={
            "endpoint": "http://localhost:8001/v1",
            "candidate_batch_sizes": [2, 4],
            "gpu_memory_fraction": 0.9,
            "gpu_memory_reserve_mb": 512,
        },
    )
    namespace_manager.register(config)
    worker = EmbeddingWorker(namespace_manager=namespace_manager, vector_store=None)
    worker.namespace_manager.register(config)
    embedder = RecordingEmbedder(config=config, responses=[], calls=[])
    monkeypatch.setattr(EmbeddingWorker, "_resolve_configs", lambda self, request: [config])
    monkeypatch.setattr(EmbeddingModelRegistry, "get", lambda self, cfg: embedder)
    monkeypatch.setattr(TokenizerCache, "ensure_within_limit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "Medical_KG_rev.services.embedding.service.ensure_memory_budget",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "Medical_KG_rev.services.embedding.service.ensure_available", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "Medical_KG_rev.embeddings.utils.gpu.memory_info",
        lambda *args, **kwargs: GPUMemoryInfo(available=True, total_mb=24576, free_mb=20000, used_mb=4576),
    )
    monkeypatch.setattr(
        "Medical_KG_rev.embeddings.utils.gpu.logger",
        types.SimpleNamespace(error=lambda *a, **k: None, warning=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        "Medical_KG_rev.services.embedding.service.logger",
        types.SimpleNamespace(
            debug=lambda *a, **k: None,
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
    )
    return worker, config, embedder


def make_request(texts: list[str]) -> EmbeddingRequest:
    return EmbeddingRequest(
        tenant_id="tenant",
        chunk_ids=[f"chunk-{i}" for i in range(len(texts))],
        texts=texts,
    )


def test_worker_batches_requests(worker_setup) -> None:
    worker, config, embedder = worker_setup
    embedder.responses.append(lambda request: _records(config, request))
    embedder.responses.append(lambda request: _records(config, request))
    embedder.responses.append(lambda request: _records(config, request))
    worker.run(make_request(["a", "b", "c", "d", "e"]))
    assert [len(call) for call in embedder.calls] == [4, 1]


def test_worker_reduces_batch_on_oom(worker_setup) -> None:
    worker, config, embedder = worker_setup
    embedder.responses.append(GpuNotAvailableError("CUDA out of memory"))
    embedder.responses.append(lambda request: _records(config, request))
    embedder.responses.append(lambda request: _records(config, request))
    response = worker.run(make_request(["a", "b", "c", "d"]))
    assert len(response.vectors) == 4
    assert worker._batch_controller.overrides[config.namespace] == 2


def test_worker_prefers_candidate_with_history(worker_setup) -> None:
    worker, config, embedder = worker_setup
    worker._batch_controller.history[config.namespace] = [(2, 0.4), (4, 0.2)]
    embedder.responses.append(lambda request: _records(config, request))
    worker.run(make_request(["a", "b", "c", "d"]))
    assert embedder.calls[0] == ("a", "b", "c", "d")


def test_worker_invokes_gpu_budget(worker_setup, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, config, embedder = worker_setup
    calls: list[tuple[float | None, int | None]] = []

    def fake_budget(require_gpu: bool, *, operation: str, fraction=None, reserve_mb=None):
        calls.append((fraction, reserve_mb))

    monkeypatch.setattr(
        "Medical_KG_rev.services.embedding.service.ensure_memory_budget",
        fake_budget,
    )
    embedder.responses.append(lambda request: _records(config, request))
    worker.run(make_request(["a", "b"]))
    assert calls[0] == (0.9, 512)


def test_worker_raises_when_gpu_budget_exhausted(worker_setup, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, config, embedder = worker_setup

    def explode(require_gpu: bool, *, operation: str, fraction=None, reserve_mb=None):
        raise GpuNotAvailableError("GPU memory limit reached")

    monkeypatch.setattr(
        "Medical_KG_rev.services.embedding.service.ensure_memory_budget",
        explode,
    )
    with pytest.raises(GpuNotAvailableError):
        worker.run(make_request(["a"]))
