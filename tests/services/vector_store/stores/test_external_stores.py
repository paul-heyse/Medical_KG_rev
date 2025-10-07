from __future__ import annotations

from pathlib import Path

import numpy as np

from Medical_KG_rev.services.vector_store.models import (
    CompressionPolicy,
    IndexParams,
    VectorQuery,
    VectorRecord,
)
from Medical_KG_rev.services.vector_store.stores.external import (
    AnnoyIndex,
    ChromaStore,
    DiskANNStore,
    DuckDBVSSStore,
    HNSWLibIndex,
    LanceDBStore,
    NMSLibIndex,
    PgvectorStore,
    ScaNNIndex,
    VespaStore,
    WeaviateStore,
)


def _records() -> list[VectorRecord]:
    return [
        VectorRecord(vector_id="a", values=[0.1, 0.2, 0.3], metadata={"text": "alpha"}),
        VectorRecord(vector_id="b", values=[0.2, 0.1, 0.4], metadata={"text": "beta"}),
        VectorRecord(vector_id="c", values=[0.4, 0.3, 0.2], metadata={"text": "gamma"}),
    ]


def _query() -> VectorQuery:
    return VectorQuery(values=[0.15, 0.2, 0.35], top_k=2)


def _setup(store, *, engine: str | None = None) -> None:
    metadata = {"engine": engine} if engine else None
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="default",
        params=IndexParams(dimension=3),
        compression=CompressionPolicy(),
        metadata=metadata,
    )
    store.upsert(tenant_id="tenant", namespace="default", records=_records())


def test_weaviate_hybrid_query() -> None:
    store = WeaviateStore()
    _setup(store)
    store.configure_hybrid(tenant_id="tenant", namespace="default", vector_weight=0.3)
    results = store.query(
        tenant_id="tenant",
        namespace="default",
        query=VectorQuery(
            values=_query().values,
            top_k=2,
            filters={"lexical_query": "alpha", "mode": "hybrid"},
        ),
    )
    assert results


def test_vespa_rank_profile_training() -> None:
    store = VespaStore()
    _setup(store, engine="faiss")
    metrics = store.train_rank_profile(
        tenant_id="tenant",
        namespace="default",
        profile="rrf",
        samples=[[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]],
    )
    assert metrics["samples"] == 2


def test_pgvector_tuning_updates_options() -> None:
    store = PgvectorStore()
    _setup(store)
    store.tune_ivf(tenant_id="tenant", namespace="default", lists=64, probes=8)


def test_diskann_precompute_builds_cache() -> None:
    store = DiskANNStore()
    _setup(store)
    metrics = store.precompute(
        tenant_id="tenant", namespace="default", vectors=[[0.1, 0.2], [0.3, 0.4]]
    )
    assert metrics["nodes"] == 2


def test_hnswlib_and_nmslib_configuration() -> None:
    hnsw = HNSWLibIndex()
    _setup(hnsw)
    hnsw.build_graph(tenant_id="tenant", namespace="default", m=16, ef_construction=200)

    nmslib = NMSLibIndex()
    _setup(nmslib)
    nmslib.configure(tenant_id="tenant", namespace="default", space="cosinesimil")


def test_annoy_and_scann_tuning() -> None:
    annoy = AnnoyIndex()
    _setup(annoy)
    annoy.build(tenant_id="tenant", namespace="default", trees=20)

    scann = ScaNNIndex()
    _setup(scann)
    scann.configure(tenant_id="tenant", namespace="default", partitions=32, leaves_to_search=10)


def test_lancedb_creates_fragments(tmp_path: Path) -> None:
    store = LanceDBStore(root=tmp_path)
    _setup(store)
    fragment = store.create_fragment(tenant_id="tenant", namespace="default")
    assert fragment.exists()


def test_duckdb_materialise_matrix() -> None:
    store = DuckDBVSSStore()
    _setup(store)
    metrics = store.materialise(
        tenant_id="tenant", namespace="default", vectors=[[0.1, 0.2, 0.3]]
    )
    assert metrics["rows"] == 1


def test_chroma_hybrid_routes() -> None:
    store = ChromaStore()
    _setup(store)
    results = store.query(
        tenant_id="tenant",
        namespace="default",
        query=VectorQuery(values=_query().values, top_k=2, filters={"lexical_query": "beta"}),
    )
    assert results
