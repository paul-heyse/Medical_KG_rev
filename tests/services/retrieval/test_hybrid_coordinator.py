import asyncio

import pytest

from Medical_KG_rev.services.retrieval.hybrid import (
    HybridComponentSettings,
    HybridSearchCoordinator,
    InMemoryHybridCache,
)


@pytest.mark.asyncio
async def test_hybrid_coordinator_runs_components_and_caches_results():
    calls: list[str] = []

    async def bm25_component(*, index: str, query: str, k: int, filters, context=None):
        await asyncio.sleep(0.01)
        calls.append("bm25")
        return [{"_id": "bm25-1", "_score": 1.0, "_source": {"text": query}}]

    async def splade_component(*, index: str, query: str, k: int, filters, context=None):
        await asyncio.sleep(0.01)
        calls.append("splade")
        return [{"_id": "splade-1", "_score": 0.9, "_source": {"text": query}}]

    settings = HybridComponentSettings(
        enable_splade=True,
        enable_dense=False,
        enable_query_expansion=False,
        timeout_ms=500,
        cache_ttl_seconds=60,
        default_components=("bm25", "splade"),
        component_timeouts={"bm25": 500, "splade": 500},
    )
    coordinator = HybridSearchCoordinator(
        {"bm25": bm25_component, "splade": splade_component},
        settings=settings,
        cache=InMemoryHybridCache(),
    )

    result = await coordinator.search(
        index="literature",
        query="Hypertension treatment",
        k=5,
        filters={},
        correlation_id="corr-1",
        context={"tenant": "alpha"},
        cache_scope="tenant-alpha",
    )
    assert not result.cache_hit
    assert set(result.component_results) == {"bm25", "splade"}
    assert result.component_errors == []
    assert result.timings_ms["bm25"] > 0
    assert result.correlation_id == "corr-1"

    cached = await coordinator.search(
        index="literature",
        query="Hypertension treatment",
        k=5,
        filters={},
        correlation_id="corr-1",
        context={"tenant": "alpha"},
        cache_scope="tenant-alpha",
    )
    assert cached.cache_hit
    assert calls.count("bm25") == 1
    assert calls.count("splade") == 1


@pytest.mark.asyncio
async def test_hybrid_coordinator_records_timeouts_and_missing_components():
    async def slow_component(*, index: str, query: str, k: int, filters, context=None):
        await asyncio.sleep(0.05)
        return []

    settings = HybridComponentSettings(
        enable_splade=True,
        enable_dense=True,
        enable_query_expansion=False,
        timeout_ms=10,
        default_components=("bm25", "splade", "dense"),
        component_timeouts={"bm25": 10, "splade": 10, "dense": 10},
    )
    coordinator = HybridSearchCoordinator(
        {"bm25": slow_component},
        settings=settings,
        cache=InMemoryHybridCache(),
    )

    result = await coordinator.search(
        index="literature",
        query="Cardiology",
        k=3,
        filters={},
        correlation_id="corr-2",
        context={"tenant": "beta"},
        cache_scope="tenant-beta",
    )

    assert sorted(result.component_results.keys()) == ["bm25", "dense", "splade"]
    assert "bm25:timeout" in result.component_errors
    # splade and dense are missing handlers
    assert "splade:unavailable" in result.component_errors
    assert "dense:unavailable" in result.component_errors
    assert result.timings_ms["splade"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_query_expansion_applies_synonyms():
    async def noop_component(*, index: str, query: str, k: int, filters, context=None):
        return []

    settings = HybridComponentSettings(
        enable_splade=False,
        enable_dense=False,
        enable_query_expansion=True,
        timeout_ms=100,
        default_components=("bm25",),
        component_timeouts={"bm25": 100},
        synonyms={"diabetes": ("dm", "hyperglycemia")},
    )
    coordinator = HybridSearchCoordinator(
        {"bm25": noop_component},
        settings=settings,
        cache=InMemoryHybridCache(),
    )

    result = await coordinator.search(
        index="literature",
        query="Diabetes management",
        k=5,
        filters={},
        correlation_id="corr-3",
        context={"tenant": "gamma"},
        cache_scope="tenant-gamma",
    )

    assert result.expanded_query is not None
    assert "dm" in result.expanded_query
    assert "hyperglycemia" in result.expanded_query
