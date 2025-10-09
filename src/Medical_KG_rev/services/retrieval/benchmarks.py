"""Benchmark helpers for evaluating reranking latency."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import mean
from time import perf_counter

from Medical_KG_rev.services.retrieval.reranker import CrossEncoderReranker


def benchmark_reranking_latency(
    reranker: CrossEncoderReranker,
    query: str,
    documents: Sequence[Mapping[str, object]],
    *,
    runs: int = 5,
    top_k: int = 20,
) -> dict[str, float]:
    """Measure reranking latency across repeated runs and compute summary stats."""
    if runs <= 0:
        raise ValueError("runs must be a positive integer")
    timings: list[float] = []
    for _ in range(runs):
        started = perf_counter()
        reranker.rerank(query, documents, top_k=top_k)
        duration_ms = (perf_counter() - started) * 1000.0
        timings.append(duration_ms)
    timings.sort()
    p95_index = min(len(timings) - 1, int(round(0.95 * (len(timings) - 1))))
    return {
        "runs": float(runs),
        "mean_ms": mean(timings),
        "median_ms": timings[len(timings) // 2],
        "p95_ms": timings[p95_index],
        "min_ms": timings[0],
        "max_ms": timings[-1],
    }


__all__ = ["benchmark_reranking_latency"]
