from __future__ import annotations

from Medical_KG_rev.services.vector_store.evaluation import (
    EvaluationRun,
    ann_parameter_sweep,
    build_leaderboard,
    compression_ab_test,
    compute_ndcg,
    compute_recall_at_k,
    estimate_memory_usage,
    hybrid_retrieval_evaluation,
    profile_latency,
)
from Medical_KG_rev.services.vector_store.models import (
    CompressionPolicy,
    IndexParams,
    VectorMatch,
    VectorQuery,
    VectorRecord,
)
from Medical_KG_rev.services.vector_store.stores.memory import InMemoryVectorStore


def test_compute_metrics() -> None:
    matches = [VectorMatch(vector_id="a", score=1.0), VectorMatch(vector_id="b", score=0.5)]
    recall = compute_recall_at_k(matches, ["a"], 2)
    ndcg = compute_ndcg(matches, {"a": 1.0, "b": 0.5}, 2)
    assert recall == 1.0
    assert ndcg > 0


def test_ann_parameter_sweep_aggregates() -> None:
    store = InMemoryVectorStore()
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="default",
        params=IndexParams(dimension=3),
        compression=CompressionPolicy(),
        metadata={},
    )
    store.upsert(
        tenant_id="tenant",
        namespace="default",
        records=[VectorRecord(vector_id="doc-1", values=[0.1, 0.2, 0.3], metadata={})],
    )
    runs = ann_parameter_sweep(
        store,
        "default",
        tenant_id="tenant",
        configs=[{"ef_search": 32}],
        queries=[[0.1, 0.2, 0.3]],
        ground_truth={"default": ["doc-1"]},
    )
    assert isinstance(runs[0], EvaluationRun)


def test_compression_and_latency_helpers() -> None:
    store = InMemoryVectorStore()
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="default",
        params=IndexParams(dimension=3),
        compression=CompressionPolicy(),
        metadata={},
    )
    store.upsert(
        tenant_id="tenant",
        namespace="default",
        records=[VectorRecord(vector_id="doc-1", values=[0.1, 0.2, 0.3], metadata={})],
    )
    recall_scores = compression_ab_test(
        store,
        "default",
        tenant_id="tenant",
        policies=["none", "int8"],
        query=[0.1, 0.2, 0.3],
        ground_truth=["doc-1"],
    )
    assert recall_scores["none"] == 1.0
    latency = profile_latency(
        store,
        "default",
        [VectorQuery(values=[0.1, 0.2, 0.3], top_k=1)],
        tenant_id="tenant",
    )
    assert latency["p50"] >= 0.0


def test_memory_and_hybrid_helpers() -> None:
    size = estimate_memory_usage([[0.1, 0.2, 0.3]])
    assert size > 0
    store = InMemoryVectorStore()
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="default",
        params=IndexParams(dimension=3),
        compression=CompressionPolicy(),
        metadata={},
    )
    store.upsert(
        tenant_id="tenant",
        namespace="default",
        records=[
            VectorRecord(
                vector_id="doc-1",
                values=[0.1, 0.2, 0.3],
                metadata={"text": "blood pressure"},
            )
        ],
    )
    hybrid = hybrid_retrieval_evaluation(
        store,
        "default",
        tenant_id="tenant",
        query=[0.1, 0.2, 0.3],
        lexical_terms=["blood"],
    )
    assert hybrid["unique_documents"] >= 1


def test_leaderboard_orders_runs() -> None:
    runs = [
        EvaluationRun(
            params={"a": 1}, recall_at_k={10: 0.5}, ndcg_at_k={}, latency_ms=10, memory_bytes=1
        ),
        EvaluationRun(
            params={"b": 1}, recall_at_k={10: 0.9}, ndcg_at_k={}, latency_ms=20, memory_bytes=1
        ),
    ]
    ordered = build_leaderboard(runs)
    assert ordered[0].params["b"] == 1
