from __future__ import annotations

from Medical_KG_rev.embeddings.frameworks.haystack import HaystackEmbedderAdapter
from Medical_KG_rev.embeddings.frameworks.langchain import LangChainEmbedderAdapter
from Medical_KG_rev.embeddings.frameworks.llama_index import LlamaIndexEmbedderAdapter
from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingRequest


class _BatchOnly:
    def embed(self, texts):  # pragma: no cover - invoked via delegate helper
        return [[float(len(text))] * 3 for text in texts]


class _QueryOnly:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, text):  # pragma: no cover - invoked via delegate helper
        self.calls.append(text)
        length = float(len(text))
        return [length, length + 1.0]


class _LlamaStyle:
    def get_text_embedding(self, text):  # pragma: no cover - invoked via delegate helper
        base = float(len(text))
        return [base, base / 2.0, base / 4.0]


def _config(name: str, namespace: str, class_path: str, dim: int = 3) -> EmbedderConfig:
    return EmbedderConfig(
        name=name,
        provider=name,
        kind="single_vector",
        namespace=namespace,
        model_id="dummy",
        dim=dim,
        normalize=False,  # Disable normalization for tests to get raw mock values
        parameters={"class_path": class_path, "init": {}},
    )


def test_langchain_adapter_batches_when_only_embed_available() -> None:
    config = _config(
        "langchain-test",
        "single_vector.langchain_test.3.v1",
        "Medical_KG_rev.embeddings.frameworks.test_mocks.BatchOnly",
    )
    adapter = LangChainEmbedderAdapter(config=config)
    request = EmbeddingRequest(
        tenant_id="tenant",
        namespace=config.namespace,
        texts=["alpha", "beta"],
        ids=["a", "b"],
        metadata=[{"source": "unit"}, {"source": "unit"}],
    )
    records = adapter.embed_documents(request)
    assert [record.id for record in records] == ["a", "b"]
    assert records[0].vectors == [[5.0, 5.0, 5.0]]
    assert records[1].vectors == [[4.0, 4.0, 4.0]]
    assert records[0].metadata["provider"] == config.provider


def test_haystack_adapter_uses_query_delegate_per_text() -> None:
    config = _config(
        "haystack-test",
        "single_vector.haystack_test.2.v1",
        "Medical_KG_rev.embeddings.frameworks.test_mocks.QueryOnly",
        dim=2,
    )
    adapter = HaystackEmbedderAdapter(config=config)
    request = EmbeddingRequest(
        tenant_id="tenant",
        namespace=config.namespace,
        texts=["gamma", "delta"],
        ids=["g", "d"],
    )
    records = adapter.embed_queries(request)
    assert [record.id for record in records] == ["g", "d"]
    assert records[0].vectors == [[5.0, 6.0]]
    assert records[1].vectors == [[5.0, 6.0]]


def test_llama_adapter_prefers_get_text_embedding() -> None:
    config = _config(
        "llama-index-test",
        "single_vector.llama_index_test.3.v1",
        "Medical_KG_rev.embeddings.frameworks.test_mocks.LlamaStyle",
    )
    adapter = LlamaIndexEmbedderAdapter(config=config)
    request = EmbeddingRequest(
        tenant_id="tenant",
        namespace=config.namespace,
        texts=["epsilon"],
        ids=["e"],
    )
    records = adapter.embed_documents(request)
    assert records[0].vectors == [[7.0, 3.5, 1.75]]
