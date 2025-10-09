"""Tests for the retrieval router fusion logic."""

from __future__ import annotations

from collections.abc import Sequence

from Medical_KG_rev.services.retrieval.router import (
    RetrievalRouter,
    RetrievalStrategy,
    RouterMatch,
    RoutingRequest,
)


def _constant_strategy(name: str, scores: Sequence[float]) -> RetrievalStrategy:
    def handler(request: RoutingRequest) -> list[RouterMatch]:
        return [
            RouterMatch(
                id=f"doc-{idx}",
                score=score,
                metadata={"strategy": name},
                source=name,
            )
            for idx, score in enumerate(scores, start=1)
        ]

    return RetrievalStrategy(name=name, handler=handler)


def test_rrf_fusion_prefers_consensus() -> None:
    router = RetrievalRouter(rrf_k=10)
    request = RoutingRequest(query="test", top_k=3, filters=None, namespace=None, context=None)
    strategies = [
        _constant_strategy("s1", [0.9, 0.2, 0.1]),
        _constant_strategy("s2", [0.8, 0.7, 0.1]),
    ]
    results = router.execute(request, strategies)
    assert results[0].id == "doc-1"
    assert results[1].id == "doc-2"


def test_linear_fusion_uses_weights() -> None:
    router = RetrievalRouter()
    request = RoutingRequest(query="test", top_k=2, filters=None, namespace="ns", context=None)

    def high_weight(request: RoutingRequest) -> list[RouterMatch]:
        return [RouterMatch(id="doc-a", score=0.2, metadata={}, source="high")]

    def low_weight(request: RoutingRequest) -> list[RouterMatch]:
        return [RouterMatch(id="doc-b", score=0.9, metadata={}, source="low")]

    strategies = [
        RetrievalStrategy(name="high", handler=high_weight, weight=5.0, fusion="linear"),
        RetrievalStrategy(name="low", handler=low_weight, weight=1.0, fusion="linear"),
    ]
    results = router.execute(request, strategies)
    assert results[0].id == "doc-a"
    assert results[0].metadata["namespace"] == "ns"
