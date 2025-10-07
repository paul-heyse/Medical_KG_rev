from __future__ import annotations

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.reranking import (
    BatchProcessor,
    CircuitBreaker,
    RerankCacheManager,
    RerankerFactory,
    RerankingEngine,
    ScoredDocument,
)


def _build_engine() -> RerankingEngine:
    return RerankingEngine(
        factory=RerankerFactory(),
        cache=RerankCacheManager(ttl_seconds=10),
        batch_processor=BatchProcessor(max_batch_size=8),
        circuit_breaker=CircuitBreaker(failure_threshold=3, reset_timeout=1.0),
    )


def test_reranking_engine_scores_and_caches():
    engine = _build_engine()
    context = SecurityContext(subject="user", tenant_id="tenant", scopes={"retrieve:read"})
    documents = [
        ScoredDocument(
            doc_id="1",
            content="hypertension treatment reduces blood pressure",
            tenant_id="tenant",
            source="bm25",
            strategy_scores={"bm25": 0.8},
            metadata={"dense_score": 0.5},
            score=0.8,
        ),
        ScoredDocument(
            doc_id="2",
            content="diabetes management guidance",
            tenant_id="tenant",
            source="bm25",
            strategy_scores={"bm25": 0.6},
            metadata={"dense_score": 0.2},
            score=0.6,
        ),
    ]

    response = engine.rerank(
        context=context,
        query="hypertension",
        documents=documents,
        reranker_id="cross_encoder:minilm",
        top_k=2,
    )
    assert len(response.results) == 2
    assert response.results[0].score >= response.results[1].score

    cached = engine.rerank(
        context=context,
        query="hypertension",
        documents=documents,
        reranker_id="cross_encoder:minilm",
        top_k=2,
    )
    cache_metrics = cached.metrics.get("cache", {})
    assert cache_metrics.get("hits", 0) >= 1


def test_reranking_engine_enforces_tenant_isolation():
    engine = _build_engine()
    context = SecurityContext(subject="user", tenant_id="tenant-a", scopes={"retrieve:read"})
    documents = [
        ScoredDocument(
            doc_id="doc",
            content="example",
            tenant_id="tenant-b",
            source="bm25",
            strategy_scores={"bm25": 0.2},
            metadata={},
            score=0.2,
        )
    ]

    try:
        engine.rerank(
            context=context,
            query="example",
            documents=documents,
            reranker_id="cross_encoder:bge",
        )
    except Exception as exc:  # noqa: BLE001 - verifying error type
        assert exc.__class__.__name__ == "RerankingError"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected reranking error for tenant mismatch")
