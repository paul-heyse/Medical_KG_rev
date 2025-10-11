"""Integration-style tests for Docling VLM pipeline processing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from Medical_KG_rev.gateway.coordinators import CoordinatorError
from Medical_KG_rev.gateway.models import DoclingProcessingRequest
from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMResult
from Medical_KG_rev.services.parsing.exceptions import DoclingModelUnavailableError


class StubDoclingService:
    def __init__(self, *, model_name: str = "stub-model") -> None:
        self.model_name = model_name

    def process_pdf(self, path: str, *, document_id: str, batch_size: int = 1) -> DoclingVLMResult:
        return DoclingVLMResult(
            document_id=document_id,
            text=f"processed:{path}",
            tables=[],
            figures=[],
            metadata={"provenance": {"model_name": self.model_name}},
        )

    def health(self) -> dict[str, object]:  # pragma: no cover - compatibility
        return {"status": "ok"}


@pytest.fixture()
def docling_coordinator() -> MagicMock:
    """Create a mock docling coordinator for testing."""
    coordinator = MagicMock()
    coordinator.docling_service = StubDoclingService()
    return coordinator


def test_process_docling_pdf_with_file_path(docling_coordinator: MagicMock) -> None:
    request = DoclingProcessingRequest(document_id="doc-1", pdf_path="/tmp/doc.pdf")
    response = docling_coordinator.process_docling_pdf(request)
    assert response.result.document_id == "doc-1"
    assert response.model_name == "stub-model"
    assert response.result.text.startswith("processed")


def test_process_docling_pdf_downloads_when_url_provided(docling_coordinator: MagicMock) -> None:
    cleanup_called: list[bool] = []

    def _cleanup() -> None:
        cleanup_called.append(True)

    with patch.object(
        gateway_service,
        "_download_pdf_from_url",
        return_value=("/tmp/downloaded.pdf", _cleanup),
    ) as downloader:
        request = DoclingProcessingRequest(
            document_id="doc-2", pdf_url="https://example.com/doc.pdf"
        )
        response = docling_coordinator.process_docling_pdf(request)
        downloader.assert_called_once()
        assert response.result.document_id == "doc-2"
        assert cleanup_called == [True]


def test_process_docling_pdf_surface_coordinator_error(docling_coordinator: MagicMock) -> None:
    docling_coordinator.docling_service = MagicMock()
    docling_coordinator.docling_service.process_pdf.side_effect = DoclingModelUnavailableError("gpu")
    request = DoclingProcessingRequest(document_id="doc-3", pdf_path="/tmp/doc.pdf")
    with pytest.raises(CoordinatorError) as exc:
        docling_coordinator.process_docling_pdf(request)
    assert exc.value.detail.status == 503
