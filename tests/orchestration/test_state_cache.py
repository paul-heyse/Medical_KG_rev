import pytest

from Medical_KG_rev.adapters.plugins.models import AdapterDomain, AdapterRequest
from Medical_KG_rev.orchestration.stages.contracts import PipelineState, StageContext
from Medical_KG_rev.orchestration.state import cache as cache_module
from Medical_KG_rev.orchestration.state.cache import PipelineStateCache


class _FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def time(self) -> float:
        return self._now

    def advance(self, delta: float) -> None:
        self._now += delta


def _snapshot() -> tuple[str, object]:
    context = StageContext(
        tenant_id="tenant",
        correlation_id="corr",
        pipeline_name="unit",
    )
    request = AdapterRequest(
        tenant_id="tenant",
        correlation_id="corr",
        domain=AdapterDomain.BIOMEDICAL,
        parameters={},
    )
    state = PipelineState.initialise(context=context, adapter_request=request)
    snapshot = state.snapshot()
    return "job-1", snapshot


def test_pipeline_state_cache_store_and_hit() -> None:
    key, snapshot = _snapshot()
    cache = PipelineStateCache(max_entries=4)
    cache.store(key, snapshot)

    cached = cache.get(key)
    assert cached is snapshot


def test_pipeline_state_cache_eviction_on_capacity() -> None:
    key, snapshot = _snapshot()
    cache = PipelineStateCache(max_entries=1)
    cache.store(key, snapshot)

    cache.store("job-2", _snapshot()[1])
    assert cache.get(key) is None
    assert cache.get("job-2") is not None


def test_pipeline_state_cache_ttl_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_clock = _FakeClock()
    monkeypatch.setattr(cache_module, "time", fake_clock)

    key, snapshot = _snapshot()
    cache = PipelineStateCache(max_entries=2, ttl_seconds=10.0)
    cache.store(key, snapshot)

    assert cache.get(key) is snapshot
    fake_clock.advance(15.0)
    assert cache.get(key) is None
