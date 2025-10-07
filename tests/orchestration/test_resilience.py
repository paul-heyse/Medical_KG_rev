import time

import pytest

from Medical_KG_rev.orchestration.pipeline import StageFailure
from Medical_KG_rev.orchestration.resilience import CircuitBreaker, CircuitState, TimeoutManager


def test_timeout_manager_raises_on_breach() -> None:
    manager = TimeoutManager()
    with pytest.raises(StageFailure):
        manager.ensure(operation="retrieve", stage="fusion", duration_seconds=0.2, timeout_ms=50)


def test_circuit_breaker_transitions() -> None:
    breaker = CircuitBreaker(service="reranker", failure_threshold=1, recovery_timeout_seconds=0.01)
    assert breaker._state is CircuitState.CLOSED
    with pytest.raises(StageFailure):
        with breaker.guard("rerank"):
            raise RuntimeError("boom")
    assert breaker._state is CircuitState.OPEN
    time.sleep(0.02)
    with pytest.raises(StageFailure):
        breaker.before_call("rerank")

