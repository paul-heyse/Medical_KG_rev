"""Unit tests for storage helper mixins."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from Medical_KG_rev.adapters.mixins.storage_helpers import StorageHelperMixin
from Medical_KG_rev.storage.clients import PdfAsset, PdfStorageClient


class TestStorageHelperMixin:
    """Test storage helper mixin functionality."""

    @pytest.fixture
    def mock_pdf_storage(self) -> PdfStorageClient:
        """Mock PDF storage client."""
        storage = MagicMock(spec=PdfStorageClient)
        storage.store_pdf = AsyncMock()
        return storage

    @pytest.fixture
    def adapter_with_storage(self, mock_pdf_storage: PdfStorageClient) -> StorageHelperMixin:
        """Adapter with storage client."""
        adapter = StorageHelperMixin()
        adapter._pdf_storage = mock_pdf_storage
        return adapter

    @pytest.fixture
    def adapter_without_storage(self) -> StorageHelperMixin:
        """Adapter without storage client."""
        adapter = StorageHelperMixin()
        adapter._pdf_storage = None
        return adapter

    def test_init_with_storage(self, mock_pdf_storage: PdfStorageClient) -> None:
        """Test mixin initialization with storage client."""
        adapter = StorageHelperMixin()
        adapter._pdf_storage = mock_pdf_storage

        assert adapter._pdf_storage == mock_pdf_storage

    def test_init_without_storage(self) -> None:
        """Test mixin initialization without storage client."""
        adapter = StorageHelperMixin()

        assert adapter._pdf_storage is None

    @pytest.mark.asyncio
    async def test_upload_pdf_if_available_success(
        self, adapter_with_storage: StorageHelperMixin, mock_pdf_storage: PdfStorageClient
    ) -> None:
        """Test successful PDF upload when storage is available."""
        pdf_data = b"fake pdf content"
        tenant_id = "test-tenant"
        document_id = "test-doc"

        # Mock successful storage
        mock_asset = PdfAsset(
            uri="s3://bucket/pdf/test-tenant/test-doc/abc123.pdf",
            checksum="abc123",
            content_type="application/pdf",
            size=1024,
        )
        mock_pdf_storage.store_pdf.return_value = mock_asset

        result = await adapter_with_storage.upload_pdf_if_available(
            tenant_id=tenant_id,
            document_id=document_id,
            pdf_data=pdf_data,
        )

        assert result == "s3://bucket/pdf/test-tenant/test-doc/abc123.pdf"
        mock_pdf_storage.store_pdf.assert_called_once_with(
            tenant_id=tenant_id,
            document_id=document_id,
            pdf_data=pdf_data,
            content_type="application/pdf",
        )

    @pytest.mark.asyncio
    async def test_upload_pdf_if_available_no_storage(
        self, adapter_without_storage: StorageHelperMixin
    ) -> None:
        """Test PDF upload when storage is not available."""
        pdf_data = b"fake pdf content"
        tenant_id = "test-tenant"
        document_id = "test-doc"

        result = await adapter_without_storage.upload_pdf_if_available(
            tenant_id=tenant_id,
            document_id=document_id,
            pdf_data=pdf_data,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_upload_pdf_if_available_no_data(
        self, adapter_with_storage: StorageHelperMixin
    ) -> None:
        """Test PDF upload with no data."""
        tenant_id = "test-tenant"
        document_id = "test-doc"

        result = await adapter_with_storage.upload_pdf_if_available(
            tenant_id=tenant_id,
            document_id=document_id,
            pdf_data=b"",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_upload_pdf_if_available_storage_error(
        self, adapter_with_storage: StorageHelperMixin, mock_pdf_storage: PdfStorageClient
    ) -> None:
        """Test PDF upload when storage raises an error."""
        pdf_data = b"fake pdf content"
        tenant_id = "test-tenant"
        document_id = "test-doc"

        # Mock storage error
        mock_pdf_storage.store_pdf.side_effect = Exception("Storage error")

        result = await adapter_with_storage.upload_pdf_if_available(
            tenant_id=tenant_id,
            document_id=document_id,
            pdf_data=pdf_data,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_upload_pdf_if_available_none_data(
        self, adapter_with_storage: StorageHelperMixin
    ) -> None:
        """Test PDF upload with None data."""
        tenant_id = "test-tenant"
        document_id = "test-doc"

        result = await adapter_with_storage.upload_pdf_if_available(
            tenant_id=tenant_id,
            document_id=document_id,
            pdf_data=None,
        )

        assert result is None


class TestStorageHelperMixinIntegration:
    """Integration tests for storage helper mixin with real adapters."""

    @pytest.mark.asyncio
    async def test_openalex_adapter_with_storage(self) -> None:
        """Test OpenAlex adapter with storage mixin."""
        from Medical_KG_rev.adapters.openalex.adapter import OpenAlexAdapter

        # Create adapter with storage
        mock_storage = MagicMock(spec=PdfStorageClient)
        mock_storage.store_pdf = AsyncMock()

        adapter = OpenAlexAdapter()
        adapter._pdf_storage = mock_storage

        # Test that adapter has storage mixin functionality
        assert hasattr(adapter, "upload_pdf_if_available")
        assert hasattr(adapter, "fetch_and_upload_pdf")

        # Test PDF upload
        pdf_data = b"fake pdf content"
        tenant_id = "test-tenant"
        document_id = "test-doc"

        mock_asset = PdfAsset(
            uri="s3://bucket/pdf/test-tenant/test-doc/abc123.pdf",
            checksum="abc123",
            content_type="application/pdf",
            size=1024,
        )
        mock_storage.store_pdf.return_value = mock_asset

        result = await adapter.upload_pdf_if_available(
            tenant_id=tenant_id,
            document_id=document_id,
            pdf_data=pdf_data,
        )

        assert result == "s3://bucket/pdf/test-tenant/test-doc/abc123.pdf"

    def test_mixin_inheritance(self) -> None:
        """Test that mixin can be inherited by adapter classes."""
        from Medical_KG_rev.adapters.openalex.adapter import OpenAlexAdapter

        # Check that OpenAlexAdapter inherits from StorageHelperMixin
        assert issubclass(OpenAlexAdapter, StorageHelperMixin)

        # Check that mixin methods are available
        adapter = OpenAlexAdapter()
        assert hasattr(adapter, "upload_pdf_if_available")
        assert hasattr(adapter, "_pdf_storage")

    def test_mixin_protocol_compliance(self) -> None:
        """Test that mixin implements the expected protocol."""
        from Medical_KG_rev.adapters.mixins.storage_helpers import HasPdfStorage

        # Test adapter with storage
        adapter_with_storage = StorageHelperMixin()
        adapter_with_storage._pdf_storage = MagicMock(spec=PdfStorageClient)

        assert isinstance(adapter_with_storage, HasPdfStorage)

        # Test adapter without storage
        adapter_without_storage = StorageHelperMixin()
        adapter_without_storage._pdf_storage = None

        assert not isinstance(adapter_without_storage, HasPdfStorage)
