from dataclasses import replace

import pytest

from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingRequest
from Medical_KG_rev.embeddings.sparse import splade


@pytest.fixture()
def stub_loader(monkeypatch: pytest.MonkeyPatch):
    created: dict[str, object] = {}

    class StubEncoder:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.calls: list[tuple[str, int]] = []
            self.responses: list[object] = []

        def encode(self, text: str, top_k: int = 400):
            self.calls.append((text, top_k))
            if self.responses:
                value = self.responses.pop(0)
                if callable(value):
                    return value(text, top_k)
                return value
            return {f"{text}_term": 1.0}

    def loader(mode: str):  # noqa: D401 - compatibility shim
        created["mode"] = mode
        created["encoder_class"] = StubEncoder
        return StubEncoder

    monkeypatch.setattr(splade, "_load_pyserini_encoder", loader)
    return created


def base_config(**parameters) -> EmbedderConfig:
    params = {"mode": "document", "top_k": 400, "max_terms": 5} | parameters
    return EmbedderConfig(
        name="splade",
        provider="pyserini",
        kind="sparse",
        namespace="sparse.splade_v3.400.v1",
        model_id="naver/splade-v3",
        model_version="v3",
        dim=400,
        normalize=False,
        parameters=params,
    )


def embedding_request(texts: list[str]) -> EmbeddingRequest:
    return EmbeddingRequest(
        tenant_id="tenant",
        namespace="sparse.splade_v3.400.v1",
        texts=texts,
        ids=[f"chunk-{i}" for i in range(len(texts))],
    )


def test_pyserini_loader_receives_mode(stub_loader) -> None:
    config = base_config(mode="query")
    embedder = splade.PyseriniSparseEmbedder(config=config)
    assert stub_loader["mode"] == "query"
    assert embedder.name == "splade"


def test_pyserini_encoder_expands_terms(stub_loader) -> None:
    embedder = splade.PyseriniSparseEmbedder(config=base_config())
    encoder = embedder._encoder  # type: ignore[attr-defined]
    encoder.responses.append({"drug": 1.5, "therapy": 0.5})
    record = embedder.embed_documents(embedding_request(["chemotherapy"]))[0]
    assert record.terms == {"drug": 1.5, "therapy": 0.5}


def test_pyserini_encoder_limits_max_terms(stub_loader) -> None:
    embedder = splade.PyseriniSparseEmbedder(config=base_config(max_terms=1))
    encoder = embedder._encoder  # type: ignore[attr-defined]
    encoder.responses.append({"alpha": 2.0, "beta": 1.0})
    record = embedder.embed_documents(embedding_request(["abc"]))[0]
    assert list(record.terms) == ["alpha"]


def test_pyserini_handles_empty_text(stub_loader) -> None:
    embedder = splade.PyseriniSparseEmbedder(config=base_config())
    encoder = embedder._encoder  # type: ignore[attr-defined]
    encoder.responses.append({})
    record = embedder.embed_documents(embedding_request([""]))[0]
    assert record.terms == {}
    assert record.dim == 0


def test_pyserini_metadata_includes_mode(stub_loader) -> None:
    embedder = splade.PyseriniSparseEmbedder(config=base_config(top_k=25))
    encoder = embedder._encoder  # type: ignore[attr-defined]
    encoder.responses.append({"term": 1.0})
    record = embedder.embed_documents(embedding_request(["doc"]))[0]
    assert record.metadata["mode"] == "document"
    assert record.metadata["top_k"] == 25


def test_pyserini_query_mode_sets_metadata(stub_loader) -> None:
    embedder = splade.PyseriniSparseEmbedder(config=base_config(mode="query"))
    encoder = embedder._encoder  # type: ignore[attr-defined]
    encoder.responses.append({"term": 1.0})
    record = embedder.embed_queries(embedding_request(["query"]))[0]
    assert record.metadata["mode"] == "query"


def test_pyserini_raises_when_encoder_returns_non_dict(stub_loader) -> None:
    embedder = splade.PyseriniSparseEmbedder(config=base_config())
    encoder = embedder._encoder  # type: ignore[attr-defined]
    encoder.responses.append([("term", 1.0)])
    with pytest.raises(TypeError):
        embedder.embed_documents(embedding_request(["boom"]))


def test_pyserini_passes_top_k_to_encoder(stub_loader) -> None:
    embedder = splade.PyseriniSparseEmbedder(config=base_config(top_k=123))
    encoder = embedder._encoder  # type: ignore[attr-defined]
    encoder.responses.append({"term": 1.0})
    embedder.embed_documents(embedding_request(["a"]))
    assert encoder.calls[0][1] == 123


def test_pyserini_supports_multiple_texts(stub_loader) -> None:
    embedder = splade.PyseriniSparseEmbedder(config=base_config())
    encoder = embedder._encoder  # type: ignore[attr-defined]
    encoder.responses.extend([{"t1": 1.0}, {"t2": 2.0}])
    records = embedder.embed_documents(embedding_request(["a", "b"]))
    assert len(records) == 2
    assert records[1].terms == {"t2": 2.0}


def test_pyserini_respects_normalize_flag(stub_loader) -> None:
    config = replace(base_config(), normalize=True)
    embedder = splade.PyseriniSparseEmbedder(config=config)
    encoder = embedder._encoder  # type: ignore[attr-defined]
    encoder.responses.append({"term": 1.0})
    record = embedder.embed_documents(embedding_request(["a"]))[0]
    assert record.normalized is True
