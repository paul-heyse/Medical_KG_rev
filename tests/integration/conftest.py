from __future__ import annotations

import os

import pytest

from Medical_KG_rev.config.settings import MineruSettings
from Medical_KG_rev.services.mineru.cli_wrapper import SimulatedMineruCli
from Medical_KG_rev.services.mineru.service import MineruProcessor
from Medical_KG_rev.services.mineru.vllm_client import VLLMClient

from .utils import run_async

_DEFAULT_VLLM_BASE_URL = os.getenv("TEST_VLLM_BASE_URL", "http://localhost:8000")


@pytest.fixture
def live_vllm_client() -> VLLMClient:
    """Provide a healthy vLLM client or skip the test if unavailable."""

    client = VLLMClient(base_url=_DEFAULT_VLLM_BASE_URL)
    try:
        healthy = run_async(client.health_check())
    except Exception as exc:  # pragma: no cover - exercised in integration environments
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
def simulated_processor() -> MineruProcessor:
    """Provide a MinerU processor wired to the simulated CLI and live vLLM client."""

    settings = MineruSettings(vllm_server={"base_url": _DEFAULT_VLLM_BASE_URL})
    cli = SimulatedMineruCli(settings)
    client = VLLMClient(base_url=_DEFAULT_VLLM_BASE_URL)

    try:
        healthy = run_async(client.health_check())
    except Exception as exc:  # pragma: no cover - exercised in integration environments
        run_async(client.close())
        pytest.skip(f"vLLM server not reachable for MinerU pipeline test: {exc}")
    if not healthy:
        run_async(client.close())
        pytest.skip("vLLM server not healthy for MinerU pipeline test")

    processor = MineruProcessor(
        settings=settings,
        cli=cli,
        worker_id="integration-worker",
        vllm_client=client,
    )

    try:
        yield processor
    finally:
        run_async(client.close())
