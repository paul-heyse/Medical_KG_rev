from __future__ import annotations

import json
from pathlib import Path

import pytest

from Medical_KG_rev.services.mineru.types import MineruRequest

from .mineru_samples import SAMPLE_DOCUMENTS

pytestmark = pytest.mark.e2e

BASELINE_PATH = Path(__file__).with_name("data").joinpath("baseline_quality.json")


def test_quality_matches_baseline(simulated_processor):
    baseline = json.loads(BASELINE_PATH.read_text())
    requests = [
        MineruRequest(tenant_id="baseline", document_id=doc_id, content=content)
        for doc_id, content in SAMPLE_DOCUMENTS.items()
    ]
    batch = simulated_processor.process_batch(requests)

    observed = {
        document.document_id: {
            "blocks": len(document.blocks),
            "tables": len(document.tables),
            "figures": len(document.figures),
            "equations": len(document.equations),
        }
        for document in batch.documents
    }

    baseline_documents = {entry["document_id"]: entry for entry in baseline.get("documents", [])}

    for doc_id, expected in baseline_documents.items():
        assert doc_id in observed, f"Missing document {doc_id} in pipeline output"
        actual = observed[doc_id]
        for key in ("blocks", "tables", "figures", "equations"):
            baseline_value = expected[key]
            actual_value = actual[key]
            if baseline_value == 0:
                assert actual_value == 0
            else:
                similarity = actual_value / baseline_value
                assert similarity >= 0.95, (
                    f"{key} similarity for {doc_id} below 95%: {similarity:.2%}"
                )

    metric_keys = ("blocks", "tables", "figures", "equations")
    totals = {key: sum(stats[key] for stats in observed.values()) for key in metric_keys}
    for key, expected_total in baseline.get("totals", {}).items():
        actual_total = totals.get(key, 0)
        if expected_total == 0:
            assert actual_total == 0
        else:
            similarity = actual_total / expected_total
            assert similarity >= 0.95, f"Total {key} similarity below 95%"
