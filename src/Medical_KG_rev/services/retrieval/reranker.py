"""Compatibility wrapper around the new reranking engine."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.reranking import (
    BatchProcessor,
    CircuitBreaker,
    RerankCacheManager,
    RerankerFactory,
    RerankingEngine,
    ScoredDocument,
)


@dataclass(slots=True)
class CrossEncoderReranker:
    """Thin adapter preserving the historical API used by the gateway."""

    reranker_id: str = "cross_encoder:bge"
    batch_size: int = 32
    cache_ttl: int = 3600
    _engine: RerankingEngine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._engine = RerankingEngine(
            factory=RerankerFactory(),
            cache=RerankCacheManager(ttl_seconds=self.cache_ttl),
            batch_processor=BatchProcessor(max_batch_size=self.batch_size),
            circuit_breaker=CircuitBreaker(),
        )

    def rerank(
        self,
        query: str,
        candidates: Iterable[Mapping[str, object]],
        *,
        text_field: str = "text",
        top_k: int = 10,
        context: SecurityContext | None = None,
    ) -> tuple[list[Mapping[str, object]], MutableMapping[str, object]]:
        items = [dict(candidate) for candidate in candidates]
        if not items:
            return [], {"model": self.reranker_id, "evaluated": 0, "applied": False}

        security_context = context or SecurityContext(
            subject="system",
            tenant_id="system",
            scopes={"*", "retrieve:read"},
        )

        documents: list[ScoredDocument] = []
        for item in items:
            doc_id = str(item.get("id") or item.get("doc_id") or len(documents))
            text = str(item.get(text_field, ""))
            metadata = dict(item)
            metadata.pop(text_field, None)
            score = float(item.get("score", 0.0))
            tenant = str(metadata.get("tenant_id", security_context.tenant_id))
            documents.append(
                ScoredDocument(
                    doc_id=doc_id,
                    content=text,
                    tenant_id=tenant,
                    source=str(metadata.get("source", "candidate")),
                    strategy_scores={"initial": score},
                    metadata=metadata,
                    highlights=list(metadata.get("highlights", []))
                    if isinstance(metadata.get("highlights"), list)
                    else [],
                    score=score,
                )
            )

        response = self._engine.rerank(
            context=security_context,
            query=query,
            documents=documents,
            reranker_id=self.reranker_id,
            top_k=top_k,
        )
        score_map = {result.doc_id: result.score for result in response.results}

        ranked: list[Mapping[str, object]] = []
        for document in documents:
            payload = dict(document.metadata)
            payload.setdefault("id", document.doc_id)
            payload.setdefault("text", document.content)
            payload.setdefault("score", document.score)
            if document.doc_id in score_map:
                payload["rerank_score"] = score_map[document.doc_id]
            ranked.append(payload)
        ranked.sort(
            key=lambda entry: entry.get("rerank_score", entry.get("score", 0.0)), reverse=True
        )

        metrics: MutableMapping[str, object] = dict(response.metrics)
        metrics.setdefault("model", self.reranker_id)
        metrics.setdefault("applied", True)
        return ranked, metrics
