import asyncio

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from Medical_KG_rev.services.mineru.vllm_client import VLLMClient

from .utils import run_async

pytestmark = pytest.mark.integration


def test_real_vllm_chat_completion(live_vllm_client: VLLMClient):
    response = run_async(
        live_vllm_client.chat_completion(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=64,
            temperature=0.0,
        )
BASE_URL = os.getenv("TEST_VLLM_BASE_URL", "http://localhost:8000")


@pytest.fixture
async def healthy_vllm_client():
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=5.0) as probe:
            response = await probe.get("/health")
            if response.status_code != 200:
                pytest.skip("vLLM server not healthy for integration test")
    except httpx.HTTPError:
        pytest.skip("vLLM server not reachable for integration test")

    client = VLLMClient(base_url=BASE_URL)
    async with client:
        yield client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_vllm_chat_completion(healthy_vllm_client: VLLMClient):
    response = await healthy_vllm_client.chat_completion(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=64,
        temperature=0.0,
    )
    assert "choices" in response
    assert response["choices"], "Expected at least one completion choice"
    content = response["choices"][0]["message"].get("content", "")
    assert isinstance(content, str) and content.strip(), "Completion content should be non-empty"
    assert "Paris" in content or "France" in content


def test_concurrent_requests(live_vllm_client: VLLMClient):
    async def _invoke() -> list[dict]:
        tasks = [
            live_vllm_client.chat_completion(
                messages=[{"role": "user", "content": f"Compute {value}+{value}"}],
                max_tokens=32,
                temperature=0.0,
            )
            for value in range(1, 5)
        ]
        return await asyncio.gather(*tasks)

    responses = run_async(_invoke())
    assert len(responses) == 4
    for payload in responses:
        assert "choices" in payload
        assert payload["choices"], "Response must contain at least one choice"
        message = payload["choices"][0]["message"].get("content", "")
        assert isinstance(message, str) and message.strip()
