from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.services.mineru.cli_wrapper import SimulatedMineruCli
from Medical_KG_rev.services.mineru.service import MineruProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest

from ._mineru_test_utils import FakeGpuManager, build_mineru_settings


def test_mineru_cli_processes_pdf():
    settings = build_mineru_settings()
    processor = MineruProcessor(
        FakeGpuManager(),
        settings=settings,
        cli=SimulatedMineruCli(settings),
        fail_fast=False,
    )

    request = MineruRequest(
        tenant_id="tenant-a",
        document_id="integration-doc",
        content=(
            "Introduction to MinerU\n"
            "Dose | Response | Notes\n"
            "5mg | Positive | Stable"
        ).encode("utf-8"),
    )

    response = processor.process(request)
    document = response.document

    assert document.document_id == "integration-doc"
    assert document.tables, "MinerU should extract tabular content from simulated input"
    table = document.tables[0]
    assert table.headers == ("Dose", "Response", "Notes")
    assert any(block.table == table for block in document.blocks)
    assert response.metadata.cli_stdout.startswith("simulated")
