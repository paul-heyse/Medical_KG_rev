"""Storage helper mixins for adapters."""

from __future__ import annotations

import hashlib
from typing import Any

import structlog
from Medical_KG_rev.storage.clients import PdfStorageClient

logger = structlog.get_logger(__name__)


class StorageHelperMixin:
    """Mixin providing storage helper methods for adapters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pdf_storage: PdfStorageClient | None = None

    def set_pdf_storage(self, pdf_storage: PdfStorageClient) -> None:
        """Set the PDF storage client for this adapter."""
        self._pdf_storage = pdf_storage

    async def upload_pdf_if_available(
        self,
        tenant_id: str,
        document_id: str,
        pdf_data: bytes | None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Upload PDF data to storage if available and storage is configured."""
        if not pdf_data or not self._pdf_storage:
            return None

        try:
            asset = await self._pdf_storage.store_pdf(
                tenant_id=tenant_id,
                document_id=document_id,
                pdf_data=pdf_data,
                content_type="application/pdf",
            )

            logger.info(
                "adapter.pdf_uploaded",
                adapter=self.__class__.__name__,
                tenant_id=tenant_id,
                document_id=document_id,
                s3_key=asset.s3_key,
                size=len(pdf_data),
                checksum=asset.checksum,
            )

            return f"s3://{asset.s3_key}"
        except Exception as e:
            logger.warning(
                "adapter.pdf_upload_failed",
                adapter=self.__class__.__name__,
                tenant_id=tenant_id,
                document_id=document_id,
                error=str(e),
            )
            return None

    def calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum for data."""
        return hashlib.sha256(data).hexdigest()

    def extract_pdf_urls(self, metadata: dict[str, Any]) -> list[str]:
        """Extract PDF URLs from metadata."""
        pdf_urls = []

        # Common PDF URL field names
        url_fields = [
            "pdf_url",
            "download_url",
            "open_access_url",
            "oa_url",
            "full_text_url",
            "url",
        ]

        for field in url_fields:
            value = metadata.get(field)
            if value:
                if isinstance(value, str):
                    pdf_urls.append(value)
                elif isinstance(value, list):
                    pdf_urls.extend([url for url in value if isinstance(url, str)])

        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in pdf_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    def is_pdf_url(self, url: str) -> bool:
        """Check if URL likely points to a PDF."""
        if not url:
            return False

        url_lower = url.lower()

        # Check file extension
        if url_lower.endswith('.pdf'):
            return True

        # Check common PDF URL patterns
        pdf_patterns = [
            '/pdf',
            '/download',
            '/fulltext',
            'format=pdf',
            'type=pdf',
        ]

        return any(pattern in url_lower for pattern in pdf_patterns)

    def normalize_pdf_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Normalize PDF-related metadata."""
        normalized = metadata.copy()

        # Extract and normalize PDF URLs
        pdf_urls = self.extract_pdf_urls(metadata)
        if pdf_urls:
            normalized["pdf_urls"] = pdf_urls

        # Add document type if PDF URLs are present
        if pdf_urls:
            normalized["document_type"] = "pdf"

        # Normalize open access fields
        if metadata.get("is_open_access") or metadata.get("open_access"):
            normalized["is_open_access"] = True

        # Normalize license information
        license_info = metadata.get("license") or metadata.get("licence")
        if license_info:
            normalized["license"] = license_info

        return normalized
