from __future__ import annotations

from pathlib import Path

import pytest

from Medical_KG_rev.embeddings.namespace import DimensionMismatchError, NamespaceManager
from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from Medical_KG_rev.embeddings.dense.sentence_transformers import SentenceTransformersEmbedder
from Medical_KG_rev.services.embedding.service import EmbeddingRequest as ServiceRequest, EmbeddingWorker


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


def test_sentence_transformers_embedder_generates_vectors() -> None:
    config = EmbedderConfig(
        name="bge-small",
        provider="sentence-transformers",
        kind="single_vector",
        namespace="single_vector.bge_small.384.v1",
        model_id="BAAI/bge-small-en",
        model_version="v1",
        dim=384,
        normalize=True,
        batch_size=2,
    )
    embedder = SentenceTransformersEmbedder(config)
    request = EmbeddingRequest(
        tenant_id="tenant-x",
        namespace=config.namespace,
        texts=["a quick brown fox"],
        ids=["chunk-1"],
    )
    records = embedder.embed_documents(request)
    assert len(records) == 1
    assert len(records[0].vectors or []) == 1
    assert pytest.approx(sum(v * v for v in records[0].vectors[0]), rel=1e-3) == pytest.approx(1.0, rel=1e-3)
    assert records[0].metadata["onnx_optimized"] is False


def test_embedding_worker_runs_with_default_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / "config" / "embeddings.yaml"
    worker = EmbeddingWorker(config_path=str(config_path))
    request = ServiceRequest(
        tenant_id="tenant-123",
        chunk_ids=["chunk-1", "chunk-2"],
        texts=["doc one text", "doc two text"],
    )
    response = worker.run(request)
    assert response.vectors
    namespaces = {vector.namespace for vector in response.vectors}
    assert "single_vector.bge_small_en.384.v1" in namespaces
    storage_targets = {vector.metadata.get("storage_target") for vector in response.vectors}
    assert "qdrant" in storage_targets
