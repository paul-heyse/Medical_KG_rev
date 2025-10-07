from __future__ import annotations

import numpy as np
import pytest

from Medical_KG_rev.services.vector_store import BackendUnavailableError
from Medical_KG_rev.services.vector_store.models import (
    CompressionPolicy,
    IndexParams,
    VectorQuery,
    VectorRecord,
)
from Medical_KG_rev.services.vector_store.stores.opensearch import OpenSearchKNNStore


def _records() -> list[VectorRecord]:
    base_vectors = [
        np.array([0.1, 0.9, 0.3]),
        np.array([0.2, 0.1, 0.8]),
        np.array([0.7, 0.2, 0.1]),
    ]
    texts = [
        "hypertension treatment reduces blood pressure",
        "placebo arm shows minimal change",
        "adverse events captured in trial",
    ]
    records: list[VectorRecord] = []
    for idx, vector in enumerate(base_vectors):
        records.append(
            VectorRecord(
                vector_id=f"doc-{idx}",
                values=vector.tolist(),
                metadata={"text": texts[idx], "arm": "treatment" if idx == 0 else "control"},
            )
        )
    return records


def test_faiss_engine_requires_training() -> None:
    store = OpenSearchKNNStore()
    params = IndexParams(dimension=3, kind="hnsw")
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="clinical",
        params=params,
        compression=CompressionPolicy(kind="fp16"),
        metadata={"engine": "faiss"},
    )
    store.upsert(tenant_id="tenant", namespace="clinical", records=_records())
    query = VectorQuery(values=[0.2, 0.8, 0.1], top_k=2)
    with pytest.raises(BackendUnavailableError):
        store.query(tenant_id="tenant", namespace="clinical", query=query)

    training = store.train_index(
        tenant_id="tenant",
        namespace="clinical",
        samples=[[0.1, 0.9, 0.3], [0.2, 0.1, 0.8]],
        encoder="pq",
    )
    assert training["samples"] == 2

    results = store.query(tenant_id="tenant", namespace="clinical", query=query)
    assert results
    assert results[0].metadata["arm"] in {"treatment", "control"}


def test_hybrid_search_combines_scores() -> None:
    store = OpenSearchKNNStore()
    params = IndexParams(dimension=3)
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="clinical",
        params=params,
        compression=CompressionPolicy(),
        metadata={"engine": "lucene"},
    )
    store.upsert(tenant_id="tenant", namespace="clinical", records=_records())
    query = VectorQuery(
        values=[0.2, 0.7, 0.2],
        top_k=3,
        filters={"lexical_query": "blood pressure", "mode": "hybrid", "vector_weight": 0.7},
    )
    results = store.query(tenant_id="tenant", namespace="clinical", query=query)
    assert results
    assert any("blood" in result.metadata.get("text", "") for result in results)

    lexical_only = store.query(
        tenant_id="tenant",
        namespace="clinical",
        query=VectorQuery(
            values=[0.2, 0.7, 0.2],
            top_k=1,
            filters={"lexical_query": "adverse events", "mode": "lexical"},
        ),
    )
    assert lexical_only
    assert lexical_only[0].vector_id == "doc-2"


def test_capabilities_reports_engines() -> None:
    store = OpenSearchKNNStore()
    params = IndexParams(dimension=3)
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="default",
        params=params,
        compression=CompressionPolicy(),
    )
    capabilities = store.capabilities()
    assert "lucene" in capabilities["engines"]
    assert "faiss" in capabilities["engines"]
