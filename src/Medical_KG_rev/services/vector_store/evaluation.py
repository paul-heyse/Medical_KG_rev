"""Utilities for evaluating vector store configurations and hybrid retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log2
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np

from .models import VectorMatch, VectorQuery
from .types import VectorStorePort


@dataclass(slots=True)
class EvaluationRun:
    params: Mapping[str, Any]
    recall_at_k: Mapping[int, float]
    ndcg_at_k: Mapping[int, float]
    latency_ms: float
    memory_bytes: int


def _build_query(vector: Sequence[float], top_k: int) -> VectorQuery:
    return VectorQuery(values=list(vector), top_k=top_k)


def compute_recall_at_k(results: Sequence[VectorMatch], ground_truth: Sequence[str], k: int) -> float:
    top = {match.vector_id for match in results[:k]}
    if not ground_truth:
        return 0.0
    relevant = top.intersection(set(ground_truth))
    return len(relevant) / len(ground_truth)


def compute_ndcg(results: Sequence[VectorMatch], ground_truth: Mapping[str, float], k: int) -> float:
    if not results:
        return 0.0
    dcg = 0.0
    for index, match in enumerate(results[:k], start=1):
        gain = ground_truth.get(match.vector_id, 0.0)
        if gain <= 0:
            continue
        dcg += gain / log2(index + 1)
    ideal = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = sum(score / log2(idx + 2) for idx, score in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def profile_latency(store: VectorStorePort, namespace: str, queries: Sequence[VectorQuery], *, tenant_id: str) -> dict[str, float]:
    timings: list[float] = []
    for query in queries:
        start = perf_counter()
        list(store.query(tenant_id=tenant_id, namespace=namespace, query=query))
        timings.append(perf_counter() - start)
    if not timings:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    timings.sort()
    get = lambda percentile: timings[min(int(len(timings) * percentile) - 1, len(timings) - 1)]
    return {"p50": get(0.5) * 1000, "p95": get(0.95) * 1000, "p99": get(0.99) * 1000}


def estimate_memory_usage(vectors: Sequence[Sequence[float]]) -> int:
    matrix = np.asarray(vectors, dtype=np.float32)
    return int(matrix.nbytes)


def ann_parameter_sweep(
    store: VectorStorePort,
    namespace: str,
    *,
    tenant_id: str,
    configs: Sequence[Mapping[str, Any]],
    queries: Sequence[Sequence[float]],
    ground_truth: Mapping[str, Sequence[str]],
) -> list[EvaluationRun]:
    runs: list[EvaluationRun] = []
    for config in configs:
        recall_scores: dict[int, float] = {}
        ndcg_scores: dict[int, float] = {}
        latencies: list[float] = []
        for vector in queries:
            query = _build_query(vector, top_k=10)
            start = perf_counter()
            results = list(store.query(tenant_id=tenant_id, namespace=namespace, query=query))
            latencies.append(perf_counter() - start)
            truth = ground_truth.get(query.vector_name or "default", [])
            recall_scores[10] = recall_scores.get(10, 0.0) + compute_recall_at_k(results, truth, 10)
            ndcg_scores[10] = ndcg_scores.get(10, 0.0) + compute_ndcg(
                results, dict.fromkeys(truth, 1.0), 10
            )
        count = len(queries) or 1
        runs.append(
            EvaluationRun(
                params=config,
                recall_at_k={10: recall_scores.get(10, 0.0) / count},
                ndcg_at_k={10: ndcg_scores.get(10, 0.0) / count},
                latency_ms=mean(latencies) * 1000 if latencies else 0.0,
                memory_bytes=estimate_memory_usage(queries),
            )
        )
    return runs


def compression_ab_test(
    store: VectorStorePort,
    namespace: str,
    *,
    tenant_id: str,
    policies: Sequence[str],
    query: Sequence[float],
    ground_truth: Sequence[str],
) -> dict[str, float]:
    results: dict[str, float] = {}
    for policy in policies:
        vector_query = _build_query(query, top_k=5)
        matches = list(store.query(tenant_id=tenant_id, namespace=namespace, query=vector_query))
        results[policy] = compute_recall_at_k(matches, ground_truth, 5)
    return results


def hybrid_retrieval_evaluation(
    store: VectorStorePort,
    namespace: str,
    *,
    tenant_id: str,
    query: Sequence[float],
    lexical_terms: Sequence[str],
) -> Mapping[str, Any]:
    vector_results = list(
        store.query(
            tenant_id=tenant_id,
            namespace=namespace,
            query=_build_query(query, top_k=5),
        )
    )
    lexical_results = list(
        store.query(
            tenant_id=tenant_id,
            namespace=namespace,
            query=VectorQuery(values=list(query), top_k=5, filters={"lexical_query": " ".join(lexical_terms), "mode": "lexical"}),
        )
    )
    combined_ids = {match.vector_id for match in vector_results + lexical_results}
    return {
        "vector_only": len(vector_results),
        "lexical_only": len(lexical_results),
        "unique_documents": len(combined_ids),
    }


def build_leaderboard(runs: Sequence[EvaluationRun]) -> list[EvaluationRun]:
    return sorted(runs, key=lambda run: (run.recall_at_k.get(10, 0.0), -run.latency_ms), reverse=True)


__all__ = [
    "EvaluationRun",
    "ann_parameter_sweep",
    "build_leaderboard",
    "compression_ab_test",
    "compute_ndcg",
    "compute_recall_at_k",
    "estimate_memory_usage",
    "hybrid_retrieval_evaluation",
    "profile_latency",
]

