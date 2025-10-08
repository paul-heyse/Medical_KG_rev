from __future__ import annotations

import pytest

from Medical_KG_rev.services.mineru.types import MineruRequest

from .mineru_samples import E2E_PRIMARY_DOCUMENT_ID, SAMPLE_DOCUMENTS

pytestmark = pytest.mark.e2e


def test_simulated_pdf_pipeline(simulated_processor):
    document_id = E2E_PRIMARY_DOCUMENT_ID
    content = SAMPLE_DOCUMENTS[document_id]
    request = MineruRequest(tenant_id="tenant-a", document_id=document_id, content=content)

    response = simulated_processor.process(request)

    assert response.document.document_id == document_id
    assert response.document.tenant_id == "tenant-a"
    assert response.document.blocks, "Expected parsed blocks in MinerU response"
    assert response.document.tables, "Expected at least one table parsed from simulated input"
    assert response.metadata.worker_id == "integration-worker"
    assert response.metadata.duration_seconds >= 0.0
    assert response.metadata.model_names
    assert response.metadata.as_dict()["cli"] == "simulated-cli"
