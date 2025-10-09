from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.services.mineru.cli_wrapper import (
    MineruCliInput,
    SimulatedMineruCli,
    create_cli,
)


def test_simulated_cli_generates_structured_output():
    workers = SimpleNamespace(backend="vlm-http-client", timeout_seconds=30)
    vllm_server = SimpleNamespace(base_url="http://localhost:8000")
    settings = SimpleNamespace(
        cli_command="mineru",
        workers=workers,
        vllm_server=vllm_server,
    )
    settings.cli_timeout_seconds = lambda: workers.timeout_seconds
    cli = create_cli(settings)
    assert isinstance(cli, SimulatedMineruCli)
    content = b"Header1|Header2\nValue1|Value2"
    result = cli.run_batch([MineruCliInput(document_id="doc-1", content=content)])
    assert result.outputs
    data = json.loads(result.outputs[0].path.read_text(encoding="utf-8"))
    assert data["document_id"] == "doc-1"
    assert len(data["blocks"]) == 2
    assert data["tables"][0]["headers"] == ["Header1", "Header2"]
