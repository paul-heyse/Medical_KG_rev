"""Shared fixtures for MinerU integration tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from Medical_KG_rev.config.settings import MineruSettings
from Medical_KG_rev.services.mineru.cli_wrapper import SimulatedMineruCli
from Medical_KG_rev.services.mineru.service import MineruProcessor
from Medical_KG_rev.services.mineru.vllm_client import VLLMClient

from .utils import run_async

_DEFAULT_VLLM_BASE_URL = os.getenv("TEST_VLLM_BASE_URL", "http://localhost:8000")


@pytest.fixture
def live_vllm_client() -> Iterator[VLLMClient]:
    """Provide a healthy vLLM client or skip integration tests if unavailable."""

    client = VLLMClient(base_url=_DEFAULT_VLLM_BASE_URL)
    try:
        healthy = run_async(client.health_check())
    except Exception as exc:  # pragma: no cover - exercised in CI environments
        run_async(client.close())
        pytest.skip(f"vLLM server not reachable for integration test: {exc}")
    if not healthy:
        run_async(client.close())
        pytest.skip("vLLM server not healthy for integration test")

    try:
        yield client
    finally:
        run_async(client.close())


@pytest.fixture
def simulated_processor(live_vllm_client: VLLMClient) -> Iterator[MineruProcessor]:
    """Provide a MinerU processor wired to the simulated CLI and healthy vLLM client."""

    settings = MineruSettings(vllm_server={"base_url": _DEFAULT_VLLM_BASE_URL})
    cli = SimulatedMineruCli(settings)
    processor = MineruProcessor(
        settings=settings,
        cli=cli,
        worker_id="integration-worker",
        vllm_client=live_vllm_client,
    )

    yield processor
