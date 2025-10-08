"""Tests for the MinerU vLLM circuit breaker implementation."""

import asyncio

from Medical_KG_rev.services.mineru.circuit_breaker import CircuitBreaker, CircuitState


def test_circuit_breaker_opens_after_failures() -> None:
    async def _run() -> None:
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0, success_threshold=1)
        assert breaker.state == CircuitState.CLOSED

        await breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert await breaker.can_execute() is False

    asyncio.run(_run())


def test_circuit_breaker_transitions_to_half_open() -> None:
    async def _run() -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05, success_threshold=1)
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        await asyncio.sleep(0.06)
        assert await breaker.can_execute() is True
        assert breaker.state == CircuitState.HALF_OPEN

    asyncio.run(_run())


def test_circuit_breaker_closes_after_successes() -> None:
    async def _run() -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05, success_threshold=2)
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        await asyncio.sleep(0.06)
        assert await breaker.can_execute() is True
        assert breaker.state == CircuitState.HALF_OPEN

        await breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN
        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    asyncio.run(_run())


def test_circuit_breaker_reopens_on_half_open_failure() -> None:
    async def _run() -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05, success_threshold=2)
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        await asyncio.sleep(0.06)
        assert await breaker.can_execute() is True
        assert breaker.state == CircuitState.HALF_OPEN

        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    asyncio.run(_run())
