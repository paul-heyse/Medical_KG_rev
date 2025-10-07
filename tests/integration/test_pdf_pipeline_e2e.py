from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.services.mineru.cli_wrapper import SimulatedMineruCli
from Medical_KG_rev.services.mineru.service import MineruProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest

from ._mineru_test_utils import FakeGpuManager, build_mineru_settings


def _build_processor() -> MineruProcessor:
    settings = build_mineru_settings()
    return MineruProcessor(
        FakeGpuManager(),
        settings=settings,
        cli=SimulatedMineruCli(settings),
        fail_fast=False,
    )


def test_pdf_pipeline_e2e_produces_tables_and_metadata():
    processor = _build_processor()
    pdf_payload = (
        "Clinical Trial Summary\n"
        "Visit | Observation | Notes\n"
        "Day 1 | Improvement | Mild"
    ).encode("utf-8")
    response = processor.process(
        MineruRequest(
            tenant_id="tenant-a",
            document_id="e2e-doc",
            content=pdf_payload,
        )
    )

    mineru_document = response.document
    assert mineru_document.document_id == "e2e-doc"
    assert mineru_document.tables, "expected tables from MinerU output"
    table_payloads = [table.model_dump() for table in mineru_document.tables]
    figures_payloads = [figure.model_dump() for figure in mineru_document.figures]
    equations_payloads = [equation.model_dump() for equation in mineru_document.equations]
    ledger_states = ["pdf_parsing", "pdf_parsed", "postpdf_processing"]
    metrics = {
        "pdf_parsing": {
            "duration_ms": pytest.approx(response.duration_seconds * 1000, rel=0.5),
            "tables": len(mineru_document.tables),
            "figures": len(mineru_document.figures),
            "equations": len(mineru_document.equations),
        }
    }

    assert list(table_payloads[0]["headers"]) == ["Visit", "Observation", "Notes"]
    assert ledger_states[-1] == "postpdf_processing"
    assert metrics["pdf_parsing"]["tables"] == len(table_payloads)
    assert figures_payloads == []
    assert equations_payloads == []
