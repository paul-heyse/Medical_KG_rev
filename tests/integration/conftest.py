from __future__ import annotations

import pytest

from Medical_KG_rev.config.settings import MineruSettings
from Medical_KG_rev.services.mineru.cli_wrapper import SimulatedMineruCli
from Medical_KG_rev.services.mineru.service import MineruProcessor


class _HealthyVLLMClient:
    async def health_check(self) -> bool:  # pragma: no cover - async helper
        return True


@pytest.fixture
def simulated_processor() -> MineruProcessor:
    """Provide a MineruProcessor wired to the simulated CLI and healthy vLLM client."""

    settings = MineruSettings()
    cli = SimulatedMineruCli(settings)
    processor = MineruProcessor(
        settings=settings,
        cli=cli,
        worker_id="integration-worker",
        vllm_client=_HealthyVLLMClient(),
    )
    return processor
