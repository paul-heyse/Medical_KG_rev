"""Tests for the Milvus vector store adapter."""

from __future__ import annotations

import pytest

from Medical_KG_rev.services.vector_store.models import (
    CompressionPolicy,
    IndexParams,
    VectorQuery,
    VectorRecord,
)
from Medical_KG_rev.services.vector_store.stores.milvus import (
    InMemoryMilvusClient,
    MilvusVectorStore,
)


@pytest.fixture()
def store() -> MilvusVectorStore:
    return MilvusVectorStore(client=InMemoryMilvusClient())


def _index_params(kind: str = "ivf_flat", **kwargs: object) -> IndexParams:
    return IndexParams(dimension=4, metric="cosine", kind=kind, **kwargs)


def _records() -> list[VectorRecord]:
    return [
        VectorRecord(vector_id="a", values=[0.1, 0.2, 0.3, 0.4], metadata={"label": "x"}),
        VectorRecord(vector_id="b", values=[0.2, 0.1, 0.0, 0.1], metadata={"label": "y"}),
        VectorRecord(vector_id="c", values=[0.9, 0.8, 0.7, 0.6], metadata={"label": "x"}),
    ]


def test_create_and_list_collections(store: MilvusVectorStore) -> None:
    params = _index_params(nlist=16)
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="research",
        params=params,
        compression=CompressionPolicy(kind="int8"),
        metadata={"description": "clinical"},
    )
    assert store.list_collections(tenant_id="tenant") == ["research"]


def test_upsert_and_query(store: MilvusVectorStore) -> None:
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="research",
        params=_index_params(kind="hnsw", ef_construct=64),
        compression=CompressionPolicy(),
    )
    store.upsert(tenant_id="tenant", namespace="research", records=_records())
    results = store.query(
        tenant_id="tenant",
        namespace="research",
        query=VectorQuery(values=[0.2, 0.1, 0.0, 0.2], top_k=2),
    )
    assert {match.vector_id for match in results} <= {"a", "b", "c"}


def test_named_vector_support(store: MilvusVectorStore) -> None:
    params = _index_params(kind="ivf_pq", nlist=8)
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="multiv",
        params=params,
        compression=CompressionPolicy(kind="pq", pq_m=2, pq_nbits=4),
        named_vectors={"title": _index_params(kind="gpu_cagra")},
    )
    store.upsert(
        tenant_id="tenant",
        namespace="multiv",
        records=[
            VectorRecord(
                vector_id="a",
                values=[0.1, 0.1, 0.1, 0.1],
                metadata={},
                named_vectors={"title": [0.9, 0.8, 0.7, 0.6]},
            )
        ],
    )
    results = store.query(
        tenant_id="tenant",
        namespace="multiv",
        query=VectorQuery(values=[0.9, 0.8, 0.7, 0.6], vector_name="title", top_k=1),
    )
    assert results[0].vector_id == "a"


def test_filter_queries(store: MilvusVectorStore) -> None:
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="filters",
        params=_index_params(kind="diskann", m=12),
        compression=CompressionPolicy(),
    )
    store.upsert(tenant_id="tenant", namespace="filters", records=_records())
    filtered = store.query(
        tenant_id="tenant",
        namespace="filters",
        query=VectorQuery(values=[0.1, 0.2, 0.3, 0.4], filters={"label": "x"}, top_k=5),
    )
    assert all(match.metadata.get("label") == "x" for match in filtered)


def test_delete_vectors(store: MilvusVectorStore) -> None:
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="delete",
        params=_index_params(),
        compression=CompressionPolicy(),
    )
    store.upsert(tenant_id="tenant", namespace="delete", records=_records())
    removed = store.delete(tenant_id="tenant", namespace="delete", vector_ids=["a", "c"])
    assert removed == 2


def test_gpu_flag_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MilvusVectorStore(client=InMemoryMilvusClient(), gpu_required=True)
    monkeypatch.setattr("Medical_KG_rev.services.vector_store.gpu.gpu_available", lambda: False)
    with pytest.raises(Exception):
        store.create_or_update_collection(
            tenant_id="tenant",
            namespace="gpu",
            params=_index_params(use_gpu=True, gpu_id=0),
            compression=CompressionPolicy(),
        )
