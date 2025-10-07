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
    FinalSelectorOrchestrator,
    FusionOrchestrator,
    QueryPipelineExecutor,
    RerankCache,
    RerankOrchestrator,
    RetrievalOrchestrator,
    RetrievalStrategy,
)


def test_retrieval_pipeline_returns_results() -> None:
    executor = ParallelExecutor(max_workers=2)
    retrieval = RetrievalOrchestrator(
        strategies=[
            RetrievalStrategy(
                name="bm25",
                executor=lambda context: [
                    {"id": "d1", "score": 1.0, "document": {"title": "Doc"}},
                    {"id": "d2", "score": 0.5, "document": {"title": "Doc2"}},
                ],
            ),
            RetrievalStrategy(
                name="dense",
                executor=lambda context: [
                    {"id": "d2", "score": 0.9, "document": {"title": "Doc2"}},
                ],
            ),
        ],
        fanout=executor,
    )
    fusion = FusionOrchestrator()
    rerank = RerankOrchestrator(
        rerank=lambda context, candidates, top_n: candidates,
        cache=RerankCache(ttl_seconds=1.0),
    )
    final = FinalSelectorOrchestrator()
    pipeline = QueryPipelineExecutor([retrieval, fusion, rerank, final])
    context = PipelineContext(tenant_id="tenant", operation="retrieve", data={"query": "test"})

    result = pipeline.run(context)

    assert result.data["results"][0]["id"] == "d1"


def test_retrieval_pipeline_handles_timeouts() -> None:
    executor = ParallelExecutor(max_workers=1)
    retrieval = RetrievalOrchestrator(
        strategies=[
            RetrievalStrategy(
                name="slow",
                executor=lambda context: [
                    {"id": "d1", "score": 1.0, "document": {"title": "Doc"}},
                ],
                timeout_ms=1,
            )
        ],
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
    retrieval = RetrievalOrchestrator(
        strategies=[
            RetrievalStrategy(
                name="dense",
                executor=lambda context: [
                    {"id": "dense-doc", "score": 0.9, "document": {"title": "Dense"}}
                ],
            ),
            RetrievalStrategy(
                name="bm25",
                executor=lambda context: [
                    {"id": "bm25-doc", "score": 1.0, "document": {"title": "BM25"}}
                ],
            ),
        ],
        fanout=executor,
    )
    fusion = FusionOrchestrator()
    rerank = RerankOrchestrator(
        rerank=lambda context, candidates, top_n: candidates,
        cache=RerankCache(ttl_seconds=1.0),
    )
    final = FinalSelectorOrchestrator()
    pipeline = QueryPipelineExecutor(
        [retrieval, fusion, rerank, final],
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
