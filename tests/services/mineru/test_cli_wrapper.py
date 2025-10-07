from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.services.mineru.cli_wrapper import MineruCliInput, SimulatedMineruCli, create_cli


def test_simulated_cli_generates_structured_output():
    workers = SimpleNamespace(vram_per_worker_gb=7, timeout_seconds=30)
    cpu = SimpleNamespace(export_environment=lambda: {})
    settings = SimpleNamespace(
        cli_command="mineru",
        workers=workers,
        cpu=cpu,
        simulate_if_unavailable=True,
    )
    settings.environment = lambda: {}
    cli = create_cli(settings)
    assert isinstance(cli, SimulatedMineruCli)
    content = b"Header1|Header2\nValue1|Value2"
    result = cli.run_batch([MineruCliInput(document_id="doc-1", content=content)], gpu_id=0)
    assert result.outputs
    data = json.loads(result.outputs[0].path.read_text(encoding="utf-8"))
    assert data["document_id"] == "doc-1"
    assert len(data["blocks"]) == 2
    assert data["tables"][0]["headers"] == ["Header1", "Header2"]
