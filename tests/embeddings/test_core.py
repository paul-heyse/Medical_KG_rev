from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from Medical_KG_rev.embeddings.dense.openai_compat import OpenAICompatEmbedder
from Medical_KG_rev.embeddings.namespace import DimensionMismatchError, NamespaceManager
from Medical_KG_rev.embeddings.ports import (
    EmbedderConfig,
    EmbeddingRecord,
    EmbeddingRequest,
    EmbeddingRequest as AdapterEmbeddingRequest,
)
from Medical_KG_rev.services import GpuNotAvailableError
from Medical_KG_rev.services.embedding.registry import EmbeddingModelRegistry
from Medical_KG_rev.services.embedding.service import EmbeddingRequest as ServiceRequest, EmbeddingWorker
from Medical_KG_rev.embeddings.utils import tokenization
from Medical_KG_rev.embeddings.utils.tokenization import TokenLimitExceededError, TokenizerCache


def test_embedding_record_validation() -> None:
    record = EmbeddingRecord(
        id="chunk-1",
        tenant_id="tenant-a",
        namespace="single_vector.test.4.v1",
        model_id="test",
        model_version="v1",
        kind="single_vector",
        dim=4,
        vectors=[[0.1, 0.2, 0.3, 0.4]],
    )
    assert record.dim == 4
    with pytest.raises(ValueError):
        EmbeddingRecord(
            id="chunk-2",
            tenant_id="tenant-a",
            namespace="sparse.test.0.v1",
            model_id="test",
            model_version="v1",
            kind="sparse",
            dim=0,
            vectors=None,
            terms=None,
        )


def test_namespace_manager_dimension_validation() -> None:
    config = EmbedderConfig(
        name="test",
        provider="sentence-transformers",
        kind="single_vector",
        namespace="single_vector.test.4.v1",
        model_id="test",
        dim=4,
    )
    manager = NamespaceManager()
    manager.register(config)
    manager.introspect_dimension(config.namespace, 4)
    with pytest.raises(DimensionMismatchError):
        manager.introspect_dimension(config.namespace, 8)


def test_openai_compat_embedder_normalizes_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EmbedderConfig(
        name="qwen3",
        provider="vllm",
        kind="single_vector",
        namespace="single_vector.qwen3.4096.v1",
        model_id="Qwen/Qwen2.5-Embedding-8B-Instruct",
        model_version="v1",
        dim=4096,
        normalize=True,
        parameters={"endpoint": "http://localhost:8001/v1"},
    )
    embedder = OpenAICompatEmbedder(config)

    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {
                "data": [
                    {"embedding": [3.0, 4.0]},
                    {"embedding": [8.0, 15.0]},
                ]
            }

        def raise_for_status(self) -> None:  # noqa: D401 - match httpx API
            return None

    monkeypatch.setattr(
        "Medical_KG_rev.embeddings.dense.openai_compat.httpx.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    request = EmbeddingRequest(
        tenant_id="tenant-x",
        namespace=config.namespace,
        texts=["alpha", "beta"],
        ids=["chunk-1", "chunk-2"],
    )
    records = embedder.embed_documents(request)
    norms = [sum(value * value for value in record.vectors[0]) ** 0.5 for record in records]
    assert pytest.approx(norms[0], rel=1e-6) == 1.0
    assert pytest.approx(norms[1], rel=1e-6) == 1.0


def test_openai_compat_embedder_raises_on_gpu_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EmbedderConfig(
        name="qwen3",
        provider="vllm",
        kind="single_vector",
        namespace="single_vector.qwen3.4096.v1",
        model_id="Qwen/Qwen2.5-Embedding-8B-Instruct",
        model_version="v1",
        dim=4096,
        normalize=True,
        parameters={"endpoint": "http://localhost:8001/v1"},
    )
    embedder = OpenAICompatEmbedder(config)

    class FakeResponse:
        status_code = 503

        def json(self) -> dict[str, Any]:
            return {"error": {"message": "GPU unavailable"}}

        def raise_for_status(self) -> None:  # pragma: no cover - not called
            raise AssertionError("raise_for_status should not be reached")

    monkeypatch.setattr(
        "Medical_KG_rev.embeddings.dense.openai_compat.httpx.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    request = EmbeddingRequest(
        tenant_id="tenant-x",
        namespace=config.namespace,
        texts=["alpha"],
        ids=["chunk-1"],
    )
    with pytest.raises(GpuNotAvailableError):
        embedder.embed_documents(request)


def test_embedding_worker_with_stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = NamespaceManager()
    worker = EmbeddingWorker(namespace_manager=manager, vector_store=None)
    fake_config = EmbedderConfig(
        name="qwen3",
        provider="vllm",
        kind="single_vector",
        namespace="single_vector.qwen3.4096.v1",
        model_id="Qwen/Qwen2.5-Embedding-8B-Instruct",
        model_version="v1",
        dim=4096,
        parameters={"max_tokens": 8192},
        requires_gpu=False,
    )
    manager.register(fake_config)

    class DummyEmbedder:
        def embed_documents(self, request: AdapterEmbeddingRequest) -> list[EmbeddingRecord]:  # type: ignore[override]
            return [
                EmbeddingRecord(
                    id=request.ids[0],
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model_id=fake_config.model_id,
                    model_version=fake_config.model_version,
                    kind=fake_config.kind,
                    dim=fake_config.dim,
                    vectors=[[0.1 for _ in range(fake_config.dim or 0)]],
                    metadata={"provider": fake_config.provider},
                )
            ]

    monkeypatch.setattr(EmbeddingWorker, "_resolve_configs", lambda self, request: [fake_config])
    monkeypatch.setattr(EmbeddingModelRegistry, "get", lambda self, config: DummyEmbedder())
    monkeypatch.setattr(TokenizerCache, "ensure_within_limit", lambda *args, **kwargs: None)

    request = ServiceRequest(
        tenant_id="tenant-x",
        chunk_ids=["chunk-1"],
        texts=["hello world"],
    )
    response = worker.run(request)
    assert response.vectors
    assert response.vectors[0].metadata["storage_target"] == "faiss"


def test_tokenizer_cache_enforces_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_load(self: tokenization._TokenizerWrapper):  # type: ignore[attr-defined]
        class Dummy:
            def encode(self, text: str, add_special_tokens: bool = False):  # noqa: D401 - mimic HF
                return text.split()

        if getattr(self, "_tokenizer", None) is None:
            calls.append("load")
            self._tokenizer = Dummy()
        return self._tokenizer

    class DummyLogger:
        def error(self, *args, **kwargs):  # noqa: D401 - capture structured args
            return None

        def debug(self, *args, **kwargs):  # noqa: D401 - capture structured args
            return None

    monkeypatch.setattr(tokenization, "logger", DummyLogger())
    monkeypatch.setattr(tokenization._TokenizerWrapper, "_load", fake_load, raising=False)
    cache = TokenizerCache()
    cache.ensure_within_limit(
        model_id="qwen3",
        texts=["short text", "two words"],
        max_tokens=3,
    )
    with pytest.raises(TokenLimitExceededError):
        cache.ensure_within_limit(
            model_id="qwen3",
            texts=["this sentence has four tokens"],
            max_tokens=3,
        )
    assert calls.count("load") == 1


def test_tokenizer_cache_reuses_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = 0

    def fake_load(self: tokenization._TokenizerWrapper):  # type: ignore[attr-defined]
        nonlocal loads
        class Dummy:
            def encode(self, text: str, add_special_tokens: bool = False):  # noqa: D401 - mimic HF
                return list(text)

        if getattr(self, "_tokenizer", None) is None:
            loads += 1
            self._tokenizer = Dummy()
        return self._tokenizer

    class DummyLogger:
        def error(self, *args, **kwargs):  # noqa: D401 - capture structured args
            return None

        def debug(self, *args, **kwargs):  # noqa: D401 - capture structured args
            return None

    monkeypatch.setattr(tokenization, "logger", DummyLogger())
    monkeypatch.setattr(tokenization._TokenizerWrapper, "_load", fake_load, raising=False)
    cache = TokenizerCache()
    cache.ensure_within_limit(model_id="qwen3", texts=["alpha"], max_tokens=10)
    cache.ensure_within_limit(model_id="qwen3", texts=["beta"], max_tokens=10)
    assert loads == 1
