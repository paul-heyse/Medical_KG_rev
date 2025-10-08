from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from Medical_KG_rev.observability.metrics import BUSINESS_EVENTS
from Medical_KG_rev.storage.base import ObjectStore, StorageError
from Medical_KG_rev.storage.object_store import InMemoryObjectStore

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class PdfStorageConfig:
    """Configuration for storing downloaded PDFs."""

    base_prefix: str = "pdf"
    enable_access_logging: bool = True


class PdfStorageClient:
    """Store downloaded PDFs using the configured object store backend."""

    def __init__(
        self,
        backend: ObjectStore | None = None,
        *,
        config: PdfStorageConfig | None = None,
    ) -> None:
        self._backend = backend or InMemoryObjectStore()
        self._config = config or PdfStorageConfig()

    async def store(
        self,
        *,
        tenant_id: str,
        document_id: str,
        data: bytes,
        content_type: str | None,
        metadata: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        checksum = hashlib.sha256(data).hexdigest()
        key = self._object_key(tenant_id, document_id, checksum)
        storage_metadata = {
            "tenant-id": tenant_id,
            "document-id": document_id,
            "checksum": checksum,
        }
        if content_type:
            storage_metadata["content-type"] = content_type
        if metadata:
            storage_metadata.update({str(k): str(v) for k, v in metadata.items()})
        await self._backend.put(key, data, metadata=storage_metadata)
        if self._config.enable_access_logging:
            logger.info(
                "pdf.storage.store",
                tenant_id=tenant_id,
                document_id=document_id,
                key=key,
                size=len(data),
                content_type=content_type,
            )
        BUSINESS_EVENTS.labels("pdf_downloaded").inc()
        return key, checksum

    async def fetch(self, key: str) -> bytes:
        data = await self._backend.get(key)
        if self._config.enable_access_logging:
            logger.debug("pdf.storage.fetch", key=key, size=len(data))
        return data

    async def delete(self, key: str) -> None:
        await self._backend.delete(key)
        if self._config.enable_access_logging:
            logger.info("pdf.storage.delete", key=key)

    async def store_processing_state(
        self,
        tenant_id: str,
        document_id: str,
        state: dict[str, Any],
    ) -> str:
        payload = json.dumps(state, separators=(",", ":"), default=str).encode("utf-8")
        key = self._processing_state_key(tenant_id, document_id)
        await self._backend.put(
            key,
            payload,
            metadata={
                "tenant-id": tenant_id,
                "document-id": document_id,
                "content-type": "application/json",
            },
        )
        if self._config.enable_access_logging:
            logger.info(
                "pdf.storage.state.store",
                tenant_id=tenant_id,
                document_id=document_id,
                key=key,
            )
        return key

    async def fetch_processing_state(
        self, tenant_id: str, document_id: str
    ) -> dict[str, Any] | None:
        key = self._processing_state_key(tenant_id, document_id)
        try:
            payload = await self._backend.get(key)
        except StorageError:
            return None
        try:
            state = json.loads(payload.decode("utf-8"))
        except Exception:
            logger.warning(
                "pdf.storage.state.invalid",
                tenant_id=tenant_id,
                document_id=document_id,
                key=key,
            )
            return None
        return state

    async def delete_processing_state(self, tenant_id: str, document_id: str) -> None:
        key = self._processing_state_key(tenant_id, document_id)
        await self._backend.delete(key)
        if self._config.enable_access_logging:
            logger.info(
                "pdf.storage.state.delete",
                tenant_id=tenant_id,
                document_id=document_id,
                key=key,
            )

    async def cleanup_document(self, tenant_id: str, document_id: str) -> None:
        prefix = self._document_prefix(tenant_id, document_id)
        if hasattr(self._backend, "list_prefix"):
            keys = await getattr(self._backend, "list_prefix")(prefix)
            for key in keys:
                await self._backend.delete(key)
                if self._config.enable_access_logging:
                    logger.info("pdf.storage.cleanup", key=key)

    def _object_key(self, tenant_id: str, document_id: str, checksum: str) -> str:
        prefix = self._document_prefix(tenant_id, document_id)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{prefix}{timestamp}-{checksum}.pdf"

    def _processing_state_key(self, tenant_id: str, document_id: str) -> str:
        prefix = self._document_prefix(tenant_id, document_id)
        return f"{prefix}processing-state.json"

    def _document_prefix(self, tenant_id: str, document_id: str) -> str:
        tenant_segment = self._safe_segment(tenant_id)
        doc_segment = self._safe_segment(document_id)
        base = self._config.base_prefix.strip("/")
        return f"{base}/{tenant_segment}/{doc_segment}/"

    def _safe_segment(self, value: str) -> str:
        cleaned = [ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value]
        segment = "".join(cleaned).strip("-")
        return segment or "unknown"

    def _list_keys(self, prefix: str) -> Iterable[str]:
        if hasattr(self._backend, "list_prefix"):
            return self._run(self._backend.list_prefix(prefix))
        return []

    def _audit(self, action: str, **kwargs: object) -> None:
        payload = {"action": action, **kwargs}
        correlation_id = get_correlation_id()
        if correlation_id:
            payload["correlation_id"] = correlation_id
        logger.info("pdf.storage.audit", **payload)

    def _run(self, awaitable):
        try:
            return asyncio.run(awaitable)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(awaitable)
            finally:
                loop.close()


__all__ = ["PdfStorageClient", "PdfStorageConfig"]
