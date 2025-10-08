"""Storage helpers for persisting downloaded PDFs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable

import structlog

from Medical_KG_rev.storage.base import ObjectStore, StorageError
from Medical_KG_rev.utils.logging import get_correlation_id

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class StoredPdf:
    """Metadata describing a stored PDF object."""

    key: str
    size_bytes: int
    content_type: str | None
    checksum: str


class PdfStorageClient:
    """Synchronously interact with an underlying :class:`ObjectStore` for PDFs."""

    def __init__(
        self,
        backend: ObjectStore,
        *,
        base_prefix: str = "pdfs",
    ) -> None:
        self._backend = backend
        self._base_prefix = base_prefix.strip("/") or "pdfs"

    def store_pdf(
        self,
        tenant_id: str,
        document_id: str,
        payload: bytes,
        *,
        checksum: str,
        content_type: str | None,
    ) -> StoredPdf:
        key = self._object_key(tenant_id, document_id, checksum)
        metadata = {
            "tenant-id": tenant_id,
            "document-id": document_id,
            "checksum": checksum,
        }
        if content_type:
            metadata["content-type"] = content_type
        self._run(self._backend.put(key, payload, metadata=metadata))
        self._audit(
            "store",
            key=key,
            tenant_id=tenant_id,
            document_id=document_id,
            bytes=len(payload),
            content_type=content_type,
        )
        return StoredPdf(
            key=key,
            size_bytes=len(payload),
            content_type=content_type,
            checksum=checksum,
        )

    def fetch_pdf(self, key: str) -> bytes:
        try:
            data = self._run(self._backend.get(key))
        except StorageError as exc:
            raise StorageError(f"Stored PDF '{key}' could not be retrieved") from exc
        self._audit("fetch", key=key, bytes=len(data))
        return data

    def delete_pdf(self, key: str) -> None:
        try:
            self._run(self._backend.delete(key))
        except StorageError:
            logger.warning("pdf.storage.delete_missing", key=key)
        else:
            self._audit("delete", key=key)

    def cleanup_document(self, tenant_id: str, document_id: str) -> int:
        """Remove all stored PDFs for the supplied document.

        Returns the number of artefacts removed which is useful for auditing
        cleanup operations triggered after failed downloads.
        """

        prefix = self._document_prefix(tenant_id, document_id)
        removed = 0
        for key in self._list_keys(prefix):
            self.delete_pdf(key)
            removed += 1
        if removed:
            self._audit(
                "cleanup",
                tenant_id=tenant_id,
                document_id=document_id,
                removed=removed,
                prefix=prefix,
            )
        return removed

    def _object_key(self, tenant_id: str, document_id: str, checksum: str) -> str:
        tenant_segment = self._safe_segment(tenant_id)
        document_segment = self._safe_segment(document_id)
        return f"{self._base_prefix}/{tenant_segment}/{document_segment}/{checksum}.pdf"

    def _document_prefix(self, tenant_id: str, document_id: str) -> str:
        tenant_segment = self._safe_segment(tenant_id)
        document_segment = self._safe_segment(document_id)
        return f"{self._base_prefix}/{tenant_segment}/{document_segment}/"

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
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(awaitable)
            finally:
                asyncio.set_event_loop(None)
                loop.close()


__all__ = ["PdfStorageClient", "StoredPdf"]
