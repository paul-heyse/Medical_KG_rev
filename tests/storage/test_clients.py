"""Unit tests for storage client factories and typed helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from Medical_KG_rev.config.settings import ObjectStorageSettings, RedisCacheSettings
from Medical_KG_rev.storage.base import ObjectStore
from Medical_KG_rev.storage.clients import (
    DocumentStorageClient,
    PdfAsset,
    PdfStorageClient,
    create_cache_backend,
    create_object_store,
    create_storage_clients,
)


class TestPdfStorageClient:
    """Test PDF storage client functionality."""

    @pytest.fixture
    def mock_store(self) -> ObjectStore:
        """Mock object store for testing."""
        store = MagicMock(spec=ObjectStore)
        store.put = AsyncMock()
        store.get = AsyncMock()
        store.generate_presigned_url = MagicMock(return_value="https://example.com/presigned")
        return store

    @pytest.fixture
    def settings(self) -> ObjectStorageSettings:
        """Test storage settings."""
        return ObjectStorageSettings(
            bucket="test-bucket",
            region="us-east-1",
            key_prefix="pdf",
        )

    @pytest.fixture
    def client(self, mock_store: ObjectStore, settings: ObjectStorageSettings) -> PdfStorageClient:
        """PDF storage client with mocked dependencies."""
        return PdfStorageClient(mock_store, settings)

    @pytest.mark.asyncio
    async def test_store_pdf_success(
        self, client: PdfStorageClient, mock_store: ObjectStore
    ) -> None:
        """Test successful PDF storage."""
        pdf_data = b"fake pdf content"
        tenant_id = "tenant-123"
        document_id = "doc-456"

        result = await client.store_pdf(
            tenant_id=tenant_id,
            document_id=document_id,
            pdf_data=pdf_data,
            content_type="application/pdf",
        )

        assert isinstance(result, PdfAsset)
        assert result.uri.startswith("s3://test-bucket/pdf/")
        assert result.checksum is not None
        assert result.size == len(pdf_data)
        assert result.content_type == "application/pdf"

        # Verify store.put was called
        mock_store.put.assert_called_once()
        call_args = mock_store.put.call_args
        assert call_args[0][0].startswith("pdf/tenant-123/doc-456/")
        assert call_args[0][1] == pdf_data

    @pytest.mark.asyncio
    async def test_get_pdf_asset_from_cache(self, client: PdfStorageClient) -> None:
        """Test retrieving PDF asset metadata from cache."""
        tenant_id = "tenant-123"
        document_id = "doc-456"
        checksum = "abc123"

        # Mock cache to return asset metadata
        mock_asset = PdfAsset(
            uri="s3://test-bucket/pdf/tenant-123/doc-456/abc123.pdf",
            checksum=checksum,
            content_type="application/pdf",
            size=1024,
        )

        with patch.object(client._cache, "get", return_value=mock_asset):
            result = await client.get_pdf_asset(tenant_id, document_id, checksum)
            assert result == mock_asset

    @pytest.mark.asyncio
    async def test_get_pdf_data_from_store(
        self, client: PdfStorageClient, mock_store: ObjectStore
    ) -> None:
        """Test retrieving PDF data from object store."""
        tenant_id = "tenant-123"
        document_id = "doc-456"
        checksum = "abc123"
        expected_data = b"fake pdf content"

        mock_store.get.return_value = expected_data

        result = await client.get_pdf_data(tenant_id, document_id, checksum)

        assert result == expected_data
        mock_store.get.assert_called_once_with("pdf/tenant-123/doc-456/abc123.pdf")

    def test_get_presigned_url(self, client: PdfStorageClient, mock_store: ObjectStore) -> None:
        """Test generating presigned URLs."""
        tenant_id = "tenant-123"
        document_id = "doc-456"
        checksum = "abc123"

        result = client.get_presigned_url(tenant_id, document_id, checksum, expires_in=3600)

        assert result == "https://example.com/presigned"
        mock_store.generate_presigned_url.assert_called_once_with(
            "pdf/tenant-123/doc-456/abc123.pdf", expires_in=3600
        )


class TestDocumentStorageClient:
    """Test document storage client functionality."""

    @pytest.fixture
    def mock_store(self) -> ObjectStore:
        """Mock object store for testing."""
        store = MagicMock(spec=ObjectStore)
        store.put = AsyncMock()
        store.get = AsyncMock()
        return store

    @pytest.fixture
    def settings(self) -> ObjectStorageSettings:
        """Test storage settings."""
        return ObjectStorageSettings(
            bucket="test-bucket",
            region="us-east-1",
        )

    @pytest.fixture
    def client(
        self, mock_store: ObjectStore, settings: ObjectStorageSettings
    ) -> DocumentStorageClient:
        """Document storage client with mocked dependencies."""
        return DocumentStorageClient(mock_store, settings)

    @pytest.mark.asyncio
    async def test_upload_document_artifact(
        self, client: DocumentStorageClient, mock_store: ObjectStore
    ) -> None:
        """Test uploading document artifacts."""
        artifact_data = b"fake artifact content"
        tenant_id = "tenant-123"
        document_id = "doc-456"
        artifact_type = "mineru_output"
        file_extension = "json"

        result = await client.upload_document_artifact(
            tenant_id=tenant_id,
            document_id=document_id,
            artifact_type=artifact_type,
            data=artifact_data,
            file_extension=file_extension,
        )

        assert result.startswith("s3://test-bucket/documents/")
        assert artifact_type in result
        assert file_extension in result

        # Verify store.put was called
        mock_store.put.assert_called_once()
        call_args = mock_store.put.call_args
        assert call_args[0][0].startswith("documents/tenant-123/doc-456/mineru_output/")
        assert call_args[0][1] == artifact_data

    @pytest.mark.asyncio
    async def test_get_document_artifact(
        self, client: DocumentStorageClient, mock_store: ObjectStore
    ) -> None:
        """Test retrieving document artifacts."""
        tenant_id = "tenant-123"
        document_id = "doc-456"
        artifact_type = "mineru_output"
        checksum = "abc123"
        file_extension = "json"
        expected_data = b"fake artifact content"

        mock_store.get.return_value = expected_data

        result = await client.get_document_artifact(
            tenant_id, document_id, artifact_type, checksum, file_extension
        )

        assert result == expected_data
        mock_store.get.assert_called_once_with(
            f"documents/tenant-123/doc-456/mineru_output/abc123.{file_extension}"
        )


class TestStorageClientFactories:
    """Test storage client factory functions."""

    def test_create_object_store_minio(self) -> None:
        """Test creating object store with MinIO endpoint."""
        settings = ObjectStorageSettings(
            endpoint_url="http://minio:9000",
            access_key_id="minioadmin",
            secret_access_key="minioadmin",
            bucket="test-bucket",
        )

        with patch("Medical_KG_rev.storage.clients.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_session.client.return_value = mock_client
            mock_boto3.session.Session.return_value = mock_session

            result = create_object_store(settings)

            assert result is not None
            mock_session.client.assert_called_once_with("s3", endpoint_url=settings.endpoint_url)

    def test_create_object_store_aws(self) -> None:
        """Test creating object store with AWS credentials."""
        settings = ObjectStorageSettings(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            bucket="test-bucket",
            region="us-east-1",
        )

        with patch("Medical_KG_rev.storage.clients.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_session.client.return_value = mock_client
            mock_boto3.session.Session.return_value = mock_session

            result = create_object_store(settings)

            assert result is not None
            mock_session.client.assert_called_once_with("s3", use_ssl=True, verify=False)

    def test_create_object_store_in_memory(self) -> None:
        """Test creating in-memory object store when no credentials provided."""
        settings = ObjectStorageSettings(bucket="test-bucket")

        result = create_object_store(settings)

        from Medical_KG_rev.storage.object_store import InMemoryObjectStore

        assert isinstance(result, InMemoryObjectStore)

    def test_create_cache_backend_redis(self) -> None:
        """Test creating Redis cache backend."""
        settings = RedisCacheSettings(url="redis://localhost:6379/0")

        with patch("Medical_KG_rev.storage.clients.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client

            result = create_cache_backend(settings)

            assert result is not None
            mock_redis.assert_called_once()

    def test_create_cache_backend_in_memory(self) -> None:
        """Test creating in-memory cache backend when no URL provided."""
        settings = RedisCacheSettings()

        result = create_cache_backend(settings)

        from Medical_KG_rev.storage.cache import InMemoryCache

        assert isinstance(result, InMemoryCache)

    def test_create_storage_clients(self) -> None:
        """Test creating all storage clients together."""
        object_settings = ObjectStorageSettings(bucket="test-bucket")
        redis_settings = RedisCacheSettings()

        with (
            patch("Medical_KG_rev.storage.clients.create_object_store") as mock_create_store,
            patch("Medical_KG_rev.storage.clients.create_cache_backend") as mock_create_cache,
        ):

            mock_store = MagicMock()
            mock_cache = MagicMock()
            mock_create_store.return_value = mock_store
            mock_create_cache.return_value = mock_cache

            result = create_storage_clients(object_settings, redis_settings)

            assert result.object_store == mock_store
            assert result.cache_backend == mock_cache
            assert isinstance(result.pdf_storage_client, PdfStorageClient)
            assert isinstance(result.document_storage_client, DocumentStorageClient)


class TestPdfAsset:
    """Test PdfAsset dataclass."""

    def test_pdf_asset_creation(self) -> None:
        """Test creating PdfAsset instances."""
        asset = PdfAsset(
            uri="s3://bucket/key.pdf",
            checksum="abc123",
            content_type="application/pdf",
            size=1024,
        )

        assert asset.uri == "s3://bucket/key.pdf"
        assert asset.checksum == "abc123"
        assert asset.content_type == "application/pdf"
        assert asset.size == 1024
        assert asset.metadata == {}

    def test_pdf_asset_with_metadata(self) -> None:
        """Test creating PdfAsset with custom metadata."""
        metadata = {"tenant_id": "tenant-123", "document_id": "doc-456"}
        asset = PdfAsset(
            uri="s3://bucket/key.pdf",
            checksum="abc123",
            metadata=metadata,
        )

        assert asset.metadata == metadata
