"""Storage client factories and helpers for S3 and Redis integration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import structlog

from Medical_KG_rev.config.settings import ObjectStorageSettings, RedisCacheSettings
from Medical_KG_rev.storage.base import CacheBackend, ObjectStore
from Medical_KG_rev.storage.cache import InMemoryCache, RedisCache
from Medical_KG_rev.storage.object_store import InMemoryObjectStore, S3ObjectStore

logger = structlog.get_logger(__name__)


@dataclass
class PdfAsset:
    """Represents a PDF asset stored in object storage."""

    tenant_id: str
    document_id: str
    checksum: str
    s3_key: str
    size: int
    content_type: str
    upload_timestamp: float
    cache_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tenant_id": self.tenant_id,
            "document_id": self.document_id,
            "checksum": self.checksum,
            "s3_key": self.s3_key,
            "size": self.size,
            "content_type": self.content_type,
            "upload_timestamp": self.upload_timestamp,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PdfAsset:
        """Create from dictionary."""
        return cls(**data)


class PdfStorageClient:
    """Client for PDF-specific storage operations."""

    def __init__(
        self,
        object_store: ObjectStore,
        cache: CacheBackend,
        settings: ObjectStorageSettings,
    ) -> None:
        self._object_store = object_store
        self._cache = cache
        self._settings = settings

    def _generate_key(self, tenant_id: str, document_id: str, checksum: str) -> str:
        """Generate S3 key for PDF storage."""
        return f"{self._settings.key_prefix}/{tenant_id}/{document_id}/{checksum}.pdf"

    def _generate_cache_key(self, tenant_id: str, document_id: str, checksum: str) -> str:
        """Generate cache key for PDF metadata."""
        return f"pdf:{tenant_id}:{document_id}:{checksum}"

    async def store_pdf(
        self,
        tenant_id: str,
        document_id: str,
        pdf_data: bytes,
        content_type: str = "application/pdf",
    ) -> PdfAsset:
        """Store PDF data and return asset metadata."""
        import time

        # Compute checksum
        checksum = hashlib.sha256(pdf_data).hexdigest()

        # Generate keys
        s3_key = self._generate_key(tenant_id, document_id, checksum)
        cache_key = self._generate_cache_key(tenant_id, document_id, checksum)

        # Store in S3
        await self._object_store.put(
            s3_key,
            pdf_data,
            metadata={
                "content-type": content_type,
                "tenant-id": tenant_id,
                "document-id": document_id,
                "checksum": checksum,
            },
        )

        # Create asset metadata
        asset = PdfAsset(
            tenant_id=tenant_id,
            document_id=document_id,
            checksum=checksum,
            s3_key=s3_key,
            size=len(pdf_data),
            content_type=content_type,
            upload_timestamp=time.time(),
            cache_key=cache_key,
        )

        # Cache metadata
        await self._cache.set(
            cache_key,
            json.dumps(asset.to_dict()).encode(),
            ttl=getattr(self._settings, "default_ttl_seconds", 3600),
        )

        logger.info(
            "pdf.stored",
            tenant_id=tenant_id,
            document_id=document_id,
            checksum=checksum,
            s3_key=s3_key,
            size=len(pdf_data),
        )

        return asset

    async def get_pdf_asset(
        self, tenant_id: str, document_id: str, checksum: str
    ) -> PdfAsset | None:
        """Retrieve PDF asset metadata from cache."""
        cache_key = self._generate_cache_key(tenant_id, document_id, checksum)

        cached_data = await self._cache.get(cache_key)
        if cached_data:
            try:
                data = json.loads(cached_data.decode())
                return PdfAsset.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "pdf.cache.corrupted",
                    cache_key=cache_key,
                    error=str(e),
                )
                return None

        return None

    async def get_pdf_data(self, asset: PdfAsset) -> bytes:
        """Retrieve PDF data from S3."""
        return await self._object_store.get(asset.s3_key)

    def get_presigned_url(
        self, tenant_id: str, document_id: str, checksum: str, expires_in: int = 3600
    ) -> str | None:
        """Generate a presigned URL for a PDF asset."""
        s3_key = self._generate_key(tenant_id, document_id, checksum)
        return self._object_store.get_presigned_url(s3_key, expires_in)

    async def delete_pdf(self, asset: PdfAsset) -> None:
        """Delete PDF from both S3 and cache."""
        # Delete from S3
        await self._object_store.delete(asset.s3_key)

        # Delete from cache
        if asset.cache_key:
            await self._cache.delete(asset.cache_key)

        logger.info(
            "pdf.deleted",
            tenant_id=asset.tenant_id,
            document_id=asset.document_id,
            checksum=asset.checksum,
            s3_key=asset.s3_key,
        )


class DocumentStorageClient:
    """Client for document metadata storage operations."""

    def __init__(
        self,
        object_store: ObjectStore,
        cache: CacheBackend,
        settings: ObjectStorageSettings,
    ) -> None:
        self._object_store = object_store
        self._cache = cache
        self._settings = settings

    def _generate_key(self, tenant_id: str, document_id: str, suffix: str) -> str:
        """Generate S3 key for document storage."""
        return f"{self._settings.key_prefix}/{tenant_id}/{document_id}/{suffix}"

    def _generate_cache_key(self, tenant_id: str, document_id: str, suffix: str) -> str:
        """Generate cache key for document metadata."""
        return f"doc:{tenant_id}:{document_id}:{suffix}"

    async def store_document_metadata(
        self,
        tenant_id: str,
        document_id: str,
        metadata: dict[str, Any],
        suffix: str = "metadata.json",
    ) -> str:
        """Store document metadata and return S3 key."""
        # Generate keys
        s3_key = self._generate_key(tenant_id, document_id, suffix)
        cache_key = self._generate_cache_key(tenant_id, document_id, suffix)

        # Serialize metadata
        metadata_json = json.dumps(metadata, indent=2)
        metadata_bytes = metadata_json.encode()

        # Store in S3
        await self._object_store.put(
            s3_key,
            metadata_bytes,
            metadata={
                "content-type": "application/json",
                "tenant-id": tenant_id,
                "document-id": document_id,
            },
        )

        # Cache metadata
        await self._cache.set(
            cache_key,
            metadata_bytes,
            ttl=getattr(self._settings, "default_ttl_seconds", 3600),
        )

        logger.info(
            "document.metadata.stored",
            tenant_id=tenant_id,
            document_id=document_id,
            s3_key=s3_key,
            cache_key=cache_key,
        )

        return s3_key

    async def get_document_metadata(
        self, tenant_id: str, document_id: str, suffix: str = "metadata.json"
    ) -> dict[str, Any] | None:
        """Retrieve document metadata from cache or S3."""
        cache_key = self._generate_cache_key(tenant_id, document_id, suffix)

        # Try cache first
        cached_data = await self._cache.get(cache_key)
        if cached_data:
            try:
                return json.loads(cached_data.decode())
            except json.JSONDecodeError as e:
                logger.warning(
                    "document.metadata.cache.corrupted",
                    cache_key=cache_key,
                    error=str(e),
                )

        # Fallback to S3
        s3_key = self._generate_key(tenant_id, document_id, suffix)
        try:
            data = await self._object_store.get(s3_key)
            metadata = json.loads(data.decode())

            # Re-cache the data
            await self._cache.set(
                cache_key,
                data,
                ttl=getattr(self._settings, "default_ttl_seconds", 3600),
            )

            return metadata
        except Exception as e:
            logger.warning(
                "document.metadata.not_found",
                tenant_id=tenant_id,
                document_id=document_id,
                s3_key=s3_key,
                error=str(e),
            )
            return None

    async def upload_document_artifact(
        self,
        tenant_id: str,
        document_id: str,
        artifact_type: str,
        data: bytes,
        file_extension: str,
    ) -> str:
        """Upload a document-related artifact to object storage."""
        s3_key = self._generate_key(
            tenant_id, document_id, f"artifacts/{artifact_type}.{file_extension}"
        )
        await self._object_store.put(
            s3_key,
            data,
            metadata={
                "content-type": f"application/{file_extension}",
                "tenant-id": tenant_id,
                "document-id": document_id,
                "artifact-type": artifact_type,
            },
        )
        return s3_key

    async def get_document_artifact(
        self,
        tenant_id: str,
        document_id: str,
        artifact_type: str,
        file_extension: str,
    ) -> bytes | None:
        """Retrieve a document-related artifact from object storage."""
        s3_key = self._generate_key(
            tenant_id, document_id, f"artifacts/{artifact_type}.{file_extension}"
        )
        return await self._object_store.get(s3_key)


def create_object_store(settings: ObjectStorageSettings) -> ObjectStore:
    """Create object store instance based on settings."""
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not available, using in-memory object store")
        return InMemoryObjectStore()

    # Check if we have credentials or endpoint URL for S3
    if settings.endpoint_url or settings.access_key_id:
        # Create S3 client with custom configuration
        s3_config = {}

        if settings.endpoint_url:
            s3_config["endpoint_url"] = settings.endpoint_url
            s3_config["use_ssl"] = settings.use_tls

        if settings.access_key_id and settings.secret_access_key:
            s3_config["aws_access_key_id"] = settings.access_key_id.get_secret_value()
            s3_config["aws_secret_access_key"] = settings.secret_access_key.get_secret_value()

            if settings.session_token:
                s3_config["aws_session_token"] = settings.session_token.get_secret_value()

        if settings.region:
            s3_config["region_name"] = settings.region

        client = boto3.client("s3", **s3_config)
        return S3ObjectStore(settings.bucket, client=client)
    else:
        # Use default AWS credentials (IAM role, environment, etc.)
        client = boto3.client("s3", region_name=settings.region)
        return S3ObjectStore(settings.bucket, client=client)


def create_cache_backend(settings: RedisCacheSettings) -> CacheBackend:
    """Create cache backend instance based on settings."""
    try:
        from redis.asyncio import Redis
    except ImportError:
        logger.warning("redis not available, using in-memory cache")
        return InMemoryCache()

    # Parse Redis URL
    redis_config = {
        "decode_responses": False,  # We want bytes for consistency
        "max_connections": settings.max_connections,
    }

    if settings.password:
        redis_config["password"] = settings.password.get_secret_value()

    if settings.use_tls and settings.tls_cert_path:
        redis_config["ssl_cert_reqs"] = "required"
        redis_config["ssl_ca_certs"] = settings.tls_cert_path

    # Create Redis client
    client = Redis.from_url(settings.url, **redis_config)
    return RedisCache(client=client)


def create_storage_clients(
    object_storage_settings: ObjectStorageSettings,
    redis_cache_settings: RedisCacheSettings,
) -> tuple[PdfStorageClient, DocumentStorageClient]:
    """Create storage client instances."""
    object_store = create_object_store(object_storage_settings)
    cache_backend = create_cache_backend(redis_cache_settings)

    pdf_client = PdfStorageClient(object_store, cache_backend, object_storage_settings)
    doc_client = DocumentStorageClient(object_store, cache_backend, object_storage_settings)

    return pdf_client, doc_client
