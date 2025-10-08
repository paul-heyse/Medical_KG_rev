from __future__ import annotations

from Medical_KG_rev.orchestration.pipeline import (
    ParallelExecutor,
    PipelineConfig,
    PipelineContext,
    PipelineDefinition,
    ProfileDefinition,
)
from Medical_KG_rev.orchestration.profiles import ProfileDetector, ProfileManager
from Medical_KG_rev.orchestration.retrieval_pipeline import (
    StrategySpec,
    FinalSelectorOrchestrator,
    FusionOrchestrator,
    QueryPipelineExecutor,
    RerankCache,
    RerankOrchestrator,
    RetrievalOrchestrator,
)
from Medical_KG_rev.services.retrieval.router import RetrievalRouter, RetrievalStrategy, RouterMatch, RoutingRequest


def test_retrieval_pipeline_returns_results() -> None:
    executor = ParallelExecutor(max_workers=2)
    strategies = {
        "bm25": StrategySpec(
            name="bm25",
            runner=lambda context, options: [
                {"id": "d1", "score": 1.0, "document": {"title": "Doc"}},
                {"id": "d2", "score": 0.5, "document": {"title": "Doc2"}},
            ],
        ),
        "dense": StrategySpec(
            name="dense",
            runner=lambda context, options: [
                {"id": "d2", "score": 0.9, "document": {"title": "Doc2"}},
            ],
        ),
    }
    retrieval = RetrievalOrchestrator(name="retrieval", strategies=strategies, fanout=executor)
    fusion = FusionOrchestrator(name="fusion")
    rerank = RerankOrchestrator(
        name="rerank",
        rerank=lambda context, candidates, options: candidates,
        cache=RerankCache(ttl_seconds=1.0),
    )
    final = FinalSelectorOrchestrator(name="final")
    pipeline = QueryPipelineExecutor([retrieval, fusion, rerank, final], pipeline_name="hybrid")
    context = PipelineContext(tenant_id="tenant", operation="retrieve", data={"query": "test"})

    result = pipeline.run(context)

    assert result.data["results"][0]["id"] == "d1"


def test_fusion_boosts_table_candidates() -> None:
    context = PipelineContext(
        tenant_id="tenant",
        operation="retrieve",
        data={
            "query": "pembrolizumab adverse events",
            "intent": {"detected": "tabular", "confidence": 0.9},
        },
    )
    context.data["retrieval_candidates"] = [
        {
            "strategy": "bm25",
            "results": [
                {
                    "id": "doc-narrative",
                    "score": 1.0,
                    "document": {"title": "Narrative", "metadata": {"title": "Narrative"}},
                },
                {
                    "id": "doc-table",
                    "score": 0.6,
                    "document": {
                        "title": "Table",
                        "metadata": {"is_table": True, "table_html": "<table></table>"},
                    },
                },
            ],
        }
    ]
    fusion = FusionOrchestrator(name="fusion")
    fusion.execute(context)

    fused = context.data["fusion_results"]
    assert fused[0]["id"] == "doc-table"
    metadata = fused[0]["document"]["metadata"]
    assert metadata["boosts"]["tabular_intent"]["multiplier"] > 1.0
    assert context.data["intent"]["boost_applied"] is True


def test_retrieval_pipeline_handles_timeouts() -> None:
    executor = ParallelExecutor(max_workers=1)
    strategies = {
        "slow": StrategySpec(
            name="slow",
            runner=lambda context, options: [
                {"id": "d1", "score": 1.0, "document": {"title": "Doc"}},
            ],
            timeout_ms=1,
        )
    }
    retrieval = RetrievalOrchestrator(
        name="retrieval",
        strategies=strategies,
        fanout=executor,
        timeout_ms=1,
    )
    pipeline = QueryPipelineExecutor([retrieval], total_timeout_ms=10)
    context = PipelineContext(tenant_id="tenant", operation="retrieve", data={"query": "test"})

    result = pipeline.run(context)

    assert result.partial is True
    assert result.errors


def test_query_pipeline_applies_profile_overrides() -> None:
    config = PipelineConfig(
        version="1.0",
        ingestion={"default": PipelineDefinition(name="default", stages=[])},
        query={"hybrid": PipelineDefinition(name="hybrid", stages=[])},
        profiles={
            "pmc": ProfileDefinition(
                name="pmc",
                ingestion="default",
                query="hybrid",
                overrides={
                    "retrieval": {"strategies": ["dense"]},
                    "final": {"top_k": 1},
                },
            )
        },
    )
    manager = ProfileManager(config, config.profiles)
    detector = ProfileDetector(manager, default_profile="pmc")

    executor = ParallelExecutor(max_workers=2)
    strategies = {
        "dense": StrategySpec(
            name="dense",
            runner=lambda context, options: [
                {"id": "dense-doc", "score": 0.9, "document": {"title": "Dense"}}
            ],
        ),
        "bm25": StrategySpec(
            name="bm25",
            runner=lambda context, options: [
                {"id": "bm25-doc", "score": 1.0, "document": {"title": "BM25"}}
            ],
        ),
    }
    retrieval = RetrievalOrchestrator(name="retrieval", strategies=strategies, fanout=executor)
    fusion = FusionOrchestrator(name="fusion")
    rerank = RerankOrchestrator(
        name="rerank",
        rerank=lambda context, candidates, options: candidates,
        cache=RerankCache(ttl_seconds=1.0),
    )
    final = FinalSelectorOrchestrator(name="final")
    pipeline = QueryPipelineExecutor(
        [retrieval, fusion, rerank, final],
        pipeline_name="hybrid",
        pipeline_version="hybrid:1.0",
        profile_detector=detector,
    )
    context = PipelineContext(
        tenant_id="tenant",
        operation="retrieve",
        data={"query": "heart", "metadata": {"source": "pmc"}},
    )

    result = pipeline.run(context)

    assert result.data["results"][0]["id"] == "dense-doc"
    assert result.pipeline_version == "hybrid:1.0"


def test_strategy_spec_from_router_adapts_results() -> None:
    router = RetrievalRouter(max_workers=1)

    def handler(request: RoutingRequest) -> list[RouterMatch]:
        return [
            RouterMatch(
                id="doc-1",
                score=0.75,
                metadata={"title": "Router Doc", "summary": "Summary", "source": "router"},
            )
        ]

    strategy = RetrievalStrategy(name="router", handler=handler)
    spec = StrategySpec.from_router(router, strategy)

    context = PipelineContext(
        tenant_id="tenant",
        operation="retrieve",
        data={"query": "oncology", "top_k": 1},
    )

    results = spec.execute(context, {})

    assert results[0]["id"] == "doc-1"
    assert results[0]["document"]["title"] == "Router Doc"
