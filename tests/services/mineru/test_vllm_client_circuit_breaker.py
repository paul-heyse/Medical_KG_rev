"""Tests covering circuit breaker behaviour within the VLLM client."""

import asyncio
from collections import deque

import pytest

pytest.importorskip("respx")
httpx = pytest.importorskip("httpx")

from Medical_KG_rev.services.mineru.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)
from Medical_KG_rev.services.mineru.vllm_client import VLLMClient, VLLMServerError


def test_vllm_client_opens_circuit_after_failures(respx_mock) -> None:
    responses = deque(
        [
            httpx.Response(500, text="Internal Server Error"),
            httpx.Response(500, text="Internal Server Error"),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}},
                    ],
                },
            ),
        ]
    )

    respx_mock.post("http://localhost:8000/v1/chat/completions").mock(
        side_effect=lambda request: responses.popleft()
    )

    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05, success_threshold=1)

    async def _run() -> None:
        async with VLLMClient(
            base_url="http://localhost:8000",
            circuit_breaker=breaker,
        ) as client:
            with pytest.raises(VLLMServerError):
                await client.chat_completion(messages=[{"role": "user", "content": "ping"}])

            assert breaker.state == CircuitState.CLOSED

            with pytest.raises(VLLMServerError):
                await client.chat_completion(messages=[{"role": "user", "content": "ping"}])

            assert breaker.state == CircuitState.OPEN

            with pytest.raises(CircuitBreakerOpenError):
                await client.chat_completion(messages=[{"role": "user", "content": "ping"}])

            assert breaker.state == CircuitState.OPEN

            await asyncio.sleep(0.06)
            assert await breaker.can_execute() is True
            assert breaker.state == CircuitState.HALF_OPEN

            response = await client.chat_completion(messages=[{"role": "user", "content": "ping"}])
            assert response["choices"][0]["message"]["content"] == "ok"
            assert breaker.state == CircuitState.CLOSED

    asyncio.run(_run())
