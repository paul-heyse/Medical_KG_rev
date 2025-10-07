"""Retrieval router supporting multi-strategy fusion and fan-out."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class RouterMatch:
    """Normalized representation of a retrieval hit."""

    id: str
    score: float
    metadata: Mapping[str, object] = field(default_factory=dict)
    source: str = ""


@dataclass(slots=True, frozen=True)
class RoutingRequest:
    """Contextual request passed to strategy handlers."""

    query: str
    top_k: int
    filters: Mapping[str, object] | None
    namespace: str | None
    context: object | None = None


@dataclass(slots=True)
class RetrievalStrategy:
    """Defines a single retrieval strategy invocation."""

    name: str
    handler: Callable[[RoutingRequest], Sequence[RouterMatch]]
    weight: float = 1.0
    fusion: str = "rrf"  # "rrf" or "linear"
    namespace: str | None = None


@dataclass(slots=True)
class RetrievalRouter:
    """Coordinates fan-out and fusion across multiple retrieval strategies."""

    max_workers: int = 4
    rrf_k: int = 60

    def execute(
        self,
        request: RoutingRequest,
        strategies: Iterable[RetrievalStrategy],
    ) -> list[RouterMatch]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: dict[Future[Sequence[RouterMatch]], RetrievalStrategy] = {}
            for strategy in strategies:
                strategy_request = RoutingRequest(
                    query=request.query,
                    top_k=request.top_k,
                    filters=request.filters,
                    namespace=strategy.namespace or request.namespace,
                    context=request.context,
                )
                futures[executor.submit(strategy.handler, strategy_request)] = strategy
            aggregates: dict[str, dict[str, object]] = defaultdict(
                lambda: {"score": 0.0, "sources": {}, "metadata": {}, "namespace": request.namespace}
            )
            for future in as_completed(futures):
                strategy = futures[future]
                matches = future.result()
                for rank, match in enumerate(matches, start=1):
                    payload = aggregates[match.id]
                    weight = strategy.weight
                    if strategy.fusion == "linear":
                        payload["score"] += weight * match.score
                    else:
                        payload["score"] += weight / (self.rrf_k + rank)
                    payload["metadata"] = {**payload["metadata"], **match.metadata}
                    payload["sources"][strategy.name] = match.score
                    if strategy.namespace:
                        payload["namespace"] = strategy.namespace
            ordered = sorted(aggregates.items(), key=lambda item: item[1]["score"], reverse=True)
        results: list[RouterMatch] = []
        for identifier, payload in ordered[: request.top_k]:
            metadata = dict(payload["metadata"])
            metadata.setdefault("sources", payload["sources"])
            if payload.get("namespace"):
                metadata.setdefault("namespace", payload["namespace"])
            results.append(
                RouterMatch(
                    id=identifier,
                    score=float(payload["score"]),
                    metadata=metadata,
                    source="fusion",
                )
            )
        return results


__all__ = [
    "RetrievalRouter",
    "RetrievalStrategy",
    "RouterMatch",
    "RoutingRequest",
]

