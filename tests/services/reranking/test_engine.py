from __future__ import annotations

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.reranking import (
    BatchProcessor,
    CircuitBreaker,
    RerankCacheManager,
    RerankerFactory,
    RerankingEngine,
    RerankResult,
    ScoredDocument,
)
from Medical_KG_rev.services.reranking.base import BaseReranker
from Medical_KG_rev.services.reranking.errors import GPUUnavailableError
from Medical_KG_rev.services.reranking.models import QueryDocumentPair


def _build_engine() -> RerankingEngine:
    return RerankingEngine(
        factory=RerankerFactory(),
        cache=RerankCacheManager(ttl_seconds=10),
        batch_processor=BatchProcessor(max_batch_size=8),
        circuit_breaker=CircuitBreaker(failure_threshold=3, reset_timeout=1.0),
    )


def test_batch_processor_timeout_split():
    processor = BatchProcessor(max_batch_size=8, batch_timeout=0.01)
    pairs = [object()] * 4
    extra = processor.split_on_timeout(pairs, duration_seconds=0.5)
    assert len(extra) == 2


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
    except Exception as exc:
        assert exc.__class__.__name__ == "RerankingError"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected reranking error for tenant mismatch")


def test_reranking_engine_explain_mode_populates_metadata():
    engine = _build_engine()
    context = SecurityContext(subject="user", tenant_id="tenant", scopes={"retrieve:read"})
    documents = [
        ScoredDocument(
            doc_id="doc-1",
            content="blood pressure treatment guidance",
            tenant_id="tenant",
            source="bm25",
            strategy_scores={"bm25": 0.7},
            metadata={"bm25_score": 12.0},
            score=0.7,
        )
    ]

    response = engine.rerank(
        context=context,
        query="blood pressure",
        documents=documents,
        reranker_id="lexical:bm25",
        explain=True,
    )
    assert "bm25_explain" in response.results[0].metadata


def test_cache_warm_allows_prepopulation():
    engine = _build_engine()
    engine.warm_cache(
        reranker_id="cross_encoder:minilm",
        tenant_id="tenant",
        version="v1.0",
        results=[RerankResult(doc_id="doc", score=0.9, rank=1)],
    )
    context = SecurityContext(subject="user", tenant_id="tenant", scopes={"retrieve:read"})
    documents = [
        ScoredDocument(
            doc_id="doc",
            content="example",
            tenant_id="tenant",
            source="bm25",
            strategy_scores={"bm25": 0.5},
            metadata={},
            score=0.5,
        )
    ]
    response = engine.rerank(
        context=context,
        query="example",
        documents=documents,
        reranker_id="cross_encoder:minilm",
    )
    assert response.results[0].score == 0.9


def test_health_reports_registered_rerankers():
    engine = _build_engine()
    status = engine.health()
    assert "cross_encoder:bge" in status
    entry = status["cross_encoder:bge"]
    assert entry["available"] is True
    assert entry["identifier"] == "bge-reranker-v2-m3"


def test_gpu_reranker_requires_gpu():
    class DummyGpuReranker(BaseReranker):
        def __init__(self) -> None:
            super().__init__("dummy-gpu", "v1", batch_size=1, requires_gpu=True)

        def _score_pair(self, pair: QueryDocumentPair) -> float:  # pragma: no cover - not used
            return 0.5

    factory = RerankerFactory()
    factory.register("test:gpu", DummyGpuReranker)
    engine = RerankingEngine(
        factory=factory,
        cache=RerankCacheManager(ttl_seconds=10),
        batch_processor=BatchProcessor(max_batch_size=4),
        circuit_breaker=CircuitBreaker(failure_threshold=2, reset_timeout=1.0),
    )
    context = SecurityContext(subject="user", tenant_id="tenant", scopes={"retrieve:read"})
    document = ScoredDocument(
        doc_id="doc",
        content="example",
        tenant_id="tenant",
        source="bm25",
        strategy_scores={"bm25": 0.5},
        metadata={},
        score=0.5,
    )
    try:
        engine.rerank(
            context=context,
            query="example",
            documents=[document],
            reranker_id="test:gpu",
        )
    except GPUUnavailableError:
        pass
    else:  # pragma: no cover - ensures failure if GPU unexpectedly available
        raise AssertionError("Expected GPUUnavailableError when GPU is not available")
