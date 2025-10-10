"""Unit tests for PDF download and gate stages."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from Medical_KG_rev.orchestration.stages.pdf_download import (
    AsyncStorageAwarePdfDownloadStage,
    StorageAwarePdfDownloadStage,
)
from Medical_KG_rev.orchestration.stages.pdf_gate import PdfGateStage, SimplePdfGateStage
from Medical_KG_rev.orchestration.stages.types import PipelineState
from Medical_KG_rev.storage.clients import PdfAsset, PdfStorageClient


class TestStorageAwarePdfDownloadStage:
    """Test storage-aware PDF download stage."""

    @pytest.fixture
    def mock_pdf_storage(self) -> PdfStorageClient:
        """Mock PDF storage client."""
        storage = MagicMock(spec=PdfStorageClient)
        storage.store_pdf = AsyncMock()
        return storage

    @pytest.fixture
    def stage(self, mock_pdf_storage: PdfStorageClient) -> StorageAwarePdfDownloadStage:
        """PDF download stage with mocked dependencies."""
        return StorageAwarePdfDownloadStage(pdf_storage=mock_pdf_storage)

    @pytest.fixture
    def pipeline_state(self) -> PipelineState:
        """Sample pipeline state with PDF URLs."""
        return PipelineState(
            job_id="test-job",
            tenant_id="test-tenant",
            document_id="test-doc",
            metadata={
                "pdf_urls": [
                    "https://example.com/paper1.pdf",
                    "https://example.com/paper2.pdf",
                ]
            },
        )

    def test_stage_name(self, stage: StorageAwarePdfDownloadStage) -> None:
        """Test stage name."""
        assert stage._stage_name == "pdf-download"

    @pytest.mark.asyncio
    async def test_execute_with_pdf_urls(
        self,
        stage: StorageAwarePdfDownloadStage,
        pipeline_state: PipelineState,
        mock_pdf_storage: PdfStorageClient,
    ) -> None:
        """Test stage execution with PDF URLs."""
        # Mock HTTP client response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake pdf content"

        with patch(
            "Medical_KG_rev.orchestration.stages.pdf_download.HttpClient"
        ) as mock_http_client:
            mock_client = MagicMock()
            mock_client.request.return_value = mock_response
            mock_http_client.return_value = mock_client

            # Mock PDF storage
            mock_asset = PdfAsset(
                uri="s3://bucket/pdf/test-tenant/test-doc/abc123.pdf",
                checksum="abc123",
                content_type="application/pdf",
                size=1024,
            )
            mock_pdf_storage.store_pdf.return_value = mock_asset

            result = await stage.execute(pipeline_state)

            # Verify PDF storage was called
            assert mock_pdf_storage.store_pdf.call_count == 2

            # Verify result contains storage URIs
            assert "pdf_assets" in result.metadata
            assert len(result.metadata["pdf_assets"]) == 2
            assert all(
                asset["storage_uri"].startswith("s3://") for asset in result.metadata["pdf_assets"]
            )

    @pytest.mark.asyncio
    async def test_execute_no_pdf_urls(self, stage: StorageAwarePdfDownloadStage) -> None:
        """Test stage execution with no PDF URLs."""
        pipeline_state = PipelineState(
            job_id="test-job",
            tenant_id="test-tenant",
            document_id="test-doc",
            metadata={},
        )

        result = await stage.execute(pipeline_state)

        # Should return state unchanged
        assert result == pipeline_state

    @pytest.mark.asyncio
    async def test_execute_http_error(
        self, stage: StorageAwarePdfDownloadStage, pipeline_state: PipelineState
    ) -> None:
        """Test stage execution with HTTP error."""
        # Mock HTTP client to return error
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch(
            "Medical_KG_rev.orchestration.stages.pdf_download.HttpClient"
        ) as mock_http_client:
            mock_client = MagicMock()
            mock_client.request.return_value = mock_response
            mock_http_client.return_value = mock_client

            result = await stage.execute(pipeline_state)

            # Should handle error gracefully and return state with error info
            assert "pdf_assets" in result.metadata
            assert len(result.metadata["pdf_assets"]) == 2
            assert all(asset["error"] is not None for asset in result.metadata["pdf_assets"])

    @pytest.mark.asyncio
    async def test_execute_storage_error(
        self,
        stage: StorageAwarePdfDownloadStage,
        pipeline_state: PipelineState,
        mock_pdf_storage: PdfStorageClient,
    ) -> None:
        """Test stage execution with storage error."""
        # Mock HTTP client response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake pdf content"

        with patch(
            "Medical_KG_rev.orchestration.stages.pdf_download.HttpClient"
        ) as mock_http_client:
            mock_client = MagicMock()
            mock_client.request.return_value = mock_response
            mock_http_client.return_value = mock_client

            # Mock PDF storage to raise exception
            mock_pdf_storage.store_pdf.side_effect = Exception("Storage error")

            result = await stage.execute(pipeline_state)

            # Should handle error gracefully
            assert "pdf_assets" in result.metadata
            assert len(result.metadata["pdf_assets"]) == 2
            assert all(asset["error"] is not None for asset in result.metadata["pdf_assets"])

    def test_download_with_retries_success(self, stage: StorageAwarePdfDownloadStage) -> None:
        """Test PDF download with retries - success case."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake pdf content"

        with patch(
            "Medical_KG_rev.orchestration.stages.pdf_download.HttpClient"
        ) as mock_http_client:
            mock_client = MagicMock()
            mock_client.request.return_value = mock_response
            mock_http_client.return_value = mock_client

            result = stage._download_with_retries("https://example.com/test.pdf")

            assert result == b"fake pdf content"
            mock_client.request.assert_called_once_with("GET", "https://example.com/test.pdf")

    def test_download_with_retries_failure(self, stage: StorageAwarePdfDownloadStage) -> None:
        """Test PDF download with retries - failure case."""
        with patch(
            "Medical_KG_rev.orchestration.stages.pdf_download.HttpClient"
        ) as mock_http_client:
            mock_client = MagicMock()
            mock_client.request.side_effect = Exception("Network error")
            mock_http_client.return_value = mock_client

            result = stage._download_with_retries("https://example.com/test.pdf")

            assert result is None

    def test_get_content_type_from_url(self, stage: StorageAwarePdfDownloadStage) -> None:
        """Test content type detection from URL."""
        assert stage._get_content_type("https://example.com/test.pdf") == "application/pdf"
        assert stage._get_content_type("https://example.com/test.PDF") == "application/pdf"
        assert stage._get_content_type("https://example.com/test") == "application/pdf"  # default

    def test_create_in_memory_artifact(self, stage: StorageAwarePdfDownloadStage) -> None:
        """Test creating in-memory artifact."""
        pdf_data = b"fake pdf content"
        url = "https://example.com/test.pdf"
        tenant_id = "test-tenant"
        document_id = "test-doc"

        artifact = stage._create_in_memory_artifact(pdf_data, url, tenant_id, document_id)

        assert artifact["url"] == url
        assert artifact["content"] == pdf_data
        assert artifact["content_type"] == "application/pdf"
        assert artifact["size"] == len(pdf_data)
        assert artifact["storage_uri"] is None
        assert artifact["error"] is None


class TestAsyncStorageAwarePdfDownloadStage:
    """Test async storage-aware PDF download stage."""

    @pytest.fixture
    def mock_pdf_storage(self) -> PdfStorageClient:
        """Mock PDF storage client."""
        storage = MagicMock(spec=PdfStorageClient)
        storage.store_pdf = AsyncMock()
        return storage

    @pytest.fixture
    def stage(self, mock_pdf_storage: PdfStorageClient) -> AsyncStorageAwarePdfDownloadStage:
        """Async PDF download stage with mocked dependencies."""
        return AsyncStorageAwarePdfDownloadStage(pdf_storage=mock_pdf_storage)

    @pytest.fixture
    def pipeline_state(self) -> PipelineState:
        """Sample pipeline state with PDF URLs."""
        return PipelineState(
            job_id="test-job",
            tenant_id="test-tenant",
            document_id="test-doc",
            metadata={"pdf_urls": ["https://example.com/paper1.pdf"]},
        )

    def test_stage_name(self, stage: AsyncStorageAwarePdfDownloadStage) -> None:
        """Test stage name."""
        assert stage._stage_name == "pdf-download"

    @pytest.mark.asyncio
    async def test_execute_async_success(
        self,
        stage: AsyncStorageAwarePdfDownloadStage,
        pipeline_state: PipelineState,
        mock_pdf_storage: PdfStorageClient,
    ) -> None:
        """Test async stage execution success."""
        # Mock async HTTP client response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake pdf content"

        with patch(
            "Medical_KG_rev.orchestration.stages.pdf_download.AsyncHttpClient"
        ) as mock_http_client:
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_http_client.return_value = mock_client

            # Mock PDF storage
            mock_asset = PdfAsset(
                uri="s3://bucket/pdf/test-tenant/test-doc/abc123.pdf",
                checksum="abc123",
                content_type="application/pdf",
                size=1024,
            )
            mock_pdf_storage.store_pdf.return_value = mock_asset

            result = await stage.execute(pipeline_state)

            # Verify PDF storage was called
            mock_pdf_storage.store_pdf.assert_called_once()

            # Verify result contains storage URI
            assert "pdf_assets" in result.metadata
            assert len(result.metadata["pdf_assets"]) == 1
            assert result.metadata["pdf_assets"][0]["storage_uri"].startswith("s3://")

    @pytest.mark.asyncio
    async def test_download_with_retries_async_success(
        self, stage: AsyncStorageAwarePdfDownloadStage
    ) -> None:
        """Test async PDF download with retries - success case."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake pdf content"

        with patch(
            "Medical_KG_rev.orchestration.stages.pdf_download.AsyncHttpClient"
        ) as mock_http_client:
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_http_client.return_value = mock_client

            result = await stage._download_with_retries_async("https://example.com/test.pdf")

            assert result == b"fake pdf content"
            mock_client.request.assert_called_once_with("GET", "https://example.com/test.pdf")

    @pytest.mark.asyncio
    async def test_download_with_retries_async_failure(
        self, stage: AsyncStorageAwarePdfDownloadStage
    ) -> None:
        """Test async PDF download with retries - failure case."""
        with patch(
            "Medical_KG_rev.orchestration.stages.pdf_download.AsyncHttpClient"
        ) as mock_http_client:
            mock_client = MagicMock()
            mock_client.request = AsyncMock(side_effect=Exception("Network error"))
            mock_http_client.return_value = mock_client

            result = await stage._download_with_retries_async("https://example.com/test.pdf")

            assert result is None


class TestPdfGateStage:
    """Test PDF gate stage."""

    @pytest.fixture
    def stage(self) -> PdfGateStage:
        """PDF gate stage."""
        return PdfGateStage()

    def test_stage_name(self, stage: PdfGateStage) -> None:
        """Test stage name."""
        assert stage._gate_name == "pdf-ir-gate"

    @pytest.mark.asyncio
    async def test_execute_gate_open(self, stage: PdfGateStage) -> None:
        """Test gate execution when gate is open."""
        from Medical_KG_rev.orchestration.stages.types import PdfGateStatus

        pipeline_state = PipelineState(
            job_id="test-job",
            tenant_id="test-tenant",
            document_id="test-doc",
            pdf_gate=PdfGateStatus(ir_ready=True),
        )

        result = await stage.execute(pipeline_state)

        # Gate should pass through unchanged
        assert result == pipeline_state

    @pytest.mark.asyncio
    async def test_execute_gate_closed(self, stage: PdfGateStage) -> None:
        """Test gate execution when gate is closed."""
        from Medical_KG_rev.orchestration.stages.types import PdfGateStatus

        pipeline_state = PipelineState(
            job_id="test-job",
            tenant_id="test-tenant",
            document_id="test-doc",
            pdf_gate=PdfGateStatus(ir_ready=False),
        )

        result = await stage.execute(pipeline_state)

        # Gate should block and return state with gate closed
        assert result.pdf_gate.ir_ready is False

    @pytest.mark.asyncio
    async def test_execute_no_gate_status(self, stage: PdfGateStage) -> None:
        """Test gate execution with no gate status."""
        pipeline_state = PipelineState(
            job_id="test-job",
            tenant_id="test-tenant",
            document_id="test-doc",
        )

        result = await stage.execute(pipeline_state)

        # Should handle missing gate status gracefully
        assert result == pipeline_state


class TestSimplePdfGateStage:
    """Test simple PDF gate stage."""

    @pytest.fixture
    def stage(self) -> SimplePdfGateStage:
        """Simple PDF gate stage."""
        return SimplePdfGateStage()

    def test_stage_name(self, stage: SimplePdfGateStage) -> None:
        """Test stage name."""
        assert stage._gate_name == "pdf-ir-gate"

    @pytest.mark.asyncio
    async def test_execute_gate_open(self, stage: SimplePdfGateStage) -> None:
        """Test simple gate execution when gate is open."""
        from Medical_KG_rev.orchestration.stages.types import PdfGateStatus

        pipeline_state = PipelineState(
            job_id="test-job",
            tenant_id="test-tenant",
            document_id="test-doc",
            pdf_gate=PdfGateStatus(ir_ready=True),
        )

        result = await stage.execute(pipeline_state)

        # Gate should pass through unchanged
        assert result == pipeline_state

    @pytest.mark.asyncio
    async def test_execute_gate_closed(self, stage: SimplePdfGateStage) -> None:
        """Test simple gate execution when gate is closed."""
        from Medical_KG_rev.orchestration.stages.types import PdfGateStatus

        pipeline_state = PipelineState(
            job_id="test-job",
            tenant_id="test-tenant",
            document_id="test-doc",
            pdf_gate=PdfGateStatus(ir_ready=False),
        )

        result = await stage.execute(pipeline_state)

        # Gate should block
        assert result.pdf_gate.ir_ready is False
