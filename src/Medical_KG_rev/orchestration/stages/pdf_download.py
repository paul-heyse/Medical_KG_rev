"""Storage-aware PDF download stage implementation."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass

import httpx

import structlog
from Medical_KG_rev.config.settings import get_settings
from Medical_KG_rev.orchestration.stages.contracts import (
    DownloadArtifact,
    DownloadStage,
    PipelineState,
    StageContext,
)
from Medical_KG_rev.orchestration.stages.plugin_manager import StagePluginContext
from Medical_KG_rev.storage import PdfStorageClient

logger = structlog.get_logger(__name__)


@dataclass
class PdfDownloadConfig:
    """Configuration for PDF download stage."""

    max_file_size: int = 100 * 1024 * 1024  # 100MB
    timeout_seconds: float = 300.0
    retry_attempts: int = 3
    retry_backoff_multiplier: float = 1.0
    allowed_content_types: tuple[str, ...] = ("application/pdf",)
    max_redirects: int = 5


class StorageAwarePdfDownloadStage(DownloadStage):
    """PDF download stage that stores files in S3 and caches metadata in Redis."""

    def __init__(self, config: PdfDownloadConfig | None = None) -> None:
        self._config = config or PdfDownloadConfig()
        self._pdf_storage: PdfStorageClient | None = None

    def initialise(self, context: StagePluginContext) -> None:
        """Initialize stage with storage resources."""
        self._pdf_storage = context.get("pdf_storage")
        if self._pdf_storage is None:
            logger.warning(
                "pdf_download_stage.no_storage",
                message="PDF storage not available, using in-memory only",
            )

    async def execute(self, ctx: StageContext, state: PipelineState) -> list[DownloadArtifact]:
        """Download PDFs and store them in object storage."""
        pdf_urls = state.metadata.get("pdf_urls", [])
        if not pdf_urls:
            logger.debug("pdf_download_stage.no_urls", tenant_id=state.tenant_id)
            return []

        artifacts: list[DownloadArtifact] = []

        for pdf_url in pdf_urls:
            try:
                artifact = await self._download_and_store_pdf(
                    ctx=ctx,
                    state=state,
                    pdf_url=pdf_url,
                )
                if artifact:
                    artifacts.append(artifact)
            except Exception as e:
                logger.error(
                    "pdf_download_stage.error",
                    tenant_id=state.tenant_id,
                    pdf_url=pdf_url,
                    error=str(e),
                )
                # Continue with other URLs instead of failing the entire stage
                continue

        logger.info(
            "pdf_download_stage.completed",
            tenant_id=state.tenant_id,
            total_urls=len(pdf_urls),
            successful_downloads=len(artifacts),
        )

        if artifacts and get_settings().feature_flags.pdf_processing_backend == "docling_vlm":
            state.mark_pdf_vlm_ready(metadata={"backend": "docling_vlm"})

        return artifacts

    async def _download_and_store_pdf(
        self,
        ctx: StageContext,
        state: PipelineState,
        pdf_url: str,
    ) -> DownloadArtifact | None:
        """Download a single PDF and store it in object storage."""
        # Generate document ID if not present
        document_id = state.document_id or f"doc-{int(time.time())}"

        # Download PDF with retries
        pdf_data = self._download_with_retries(pdf_url)
        if not pdf_data:
            return None

        # Validate content type
        content_type = self._detect_content_type(pdf_data)
        if content_type not in self._config.allowed_content_types:
            logger.warning(
                "pdf_download_stage.invalid_content_type",
                tenant_id=state.tenant_id,
                pdf_url=pdf_url,
                content_type=content_type,
            )
            return None

        # Store in object storage if available
        if self._pdf_storage:
            try:
                asset = await self._pdf_storage.store_pdf(
                    tenant_id=state.tenant_id,
                    document_id=document_id,
                    pdf_data=pdf_data,
                    content_type=content_type,
                )

                logger.info(
                    "pdf_download_stage.stored",
                    tenant_id=state.tenant_id,
                    document_id=document_id,
                    pdf_url=pdf_url,
                    s3_key=asset.s3_key,
                    size=len(pdf_data),
                    checksum=asset.checksum,
                )

                return DownloadArtifact(
                    document_id=document_id,
                    tenant_id=state.tenant_id,
                    uri=f"s3://{asset.s3_key}",
                    metadata={
                        "checksum": asset.checksum,
                        "size": len(pdf_data),
                        "content_type": content_type,
                        "pdf_url": pdf_url,
                        "s3_key": asset.s3_key,
                        "cache_key": asset.cache_key,
                        "upload_timestamp": asset.upload_timestamp,
                        "vlm_ready": True,
                        "backends": ["docling_vlm"],
                    },
                )
            except Exception as e:
                logger.error(
                    "pdf_download_stage.storage_error",
                    tenant_id=state.tenant_id,
                    pdf_url=pdf_url,
                    error=str(e),
                )
                # Fall back to in-memory storage
                return self._create_in_memory_artifact(
                    pdf_url, pdf_data, content_type, document_id, state.tenant_id
                )
        else:
            # No storage available, use in-memory only
            return self._create_in_memory_artifact(
                pdf_url, pdf_data, content_type, document_id, state.tenant_id
            )

    def _download_with_retries(self, pdf_url: str) -> bytes | None:
        """Download PDF with retry logic."""
        for attempt in range(self._config.retry_attempts):
            try:
                with httpx.Client(
                    timeout=self._config.timeout_seconds,
                    follow_redirects=True,
                    max_redirects=self._config.max_redirects,
                ) as client:
                    response = client.get(pdf_url)
                    response.raise_for_status()

                    # Check content length
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > self._config.max_file_size:
                        logger.warning(
                            "pdf_download_stage.file_too_large",
                            pdf_url=pdf_url,
                            content_length=int(content_length),
                            max_size=self._config.max_file_size,
                        )
                        return None

                    pdf_data = response.content

                    # Check actual size
                    if len(pdf_data) > self._config.max_file_size:
                        logger.warning(
                            "pdf_download_stage.file_too_large",
                            pdf_url=pdf_url,
                            actual_size=len(pdf_data),
                            max_size=self._config.max_file_size,
                        )
                        return None

                    return pdf_data

            except httpx.TimeoutException:
                logger.warning(
                    "pdf_download_stage.timeout",
                    pdf_url=pdf_url,
                    attempt=attempt + 1,
                    timeout=self._config.timeout_seconds,
                )
            except httpx.HTTPStatusError as e:
                logger.warning(
                    "pdf_download_stage.http_error",
                    pdf_url=pdf_url,
                    attempt=attempt + 1,
                    status_code=e.response.status_code,
                )
            except Exception as e:
                logger.warning(
                    "pdf_download_stage.download_error",
                    pdf_url=pdf_url,
                    attempt=attempt + 1,
                    error=str(e),
                )

            # Wait before retry
            if attempt < self._config.retry_attempts - 1:
                wait_time = self._config.retry_backoff_multiplier * (2**attempt)
                time.sleep(wait_time)

        logger.error(
            "pdf_download_stage.download_failed",
            pdf_url=pdf_url,
            attempts=self._config.retry_attempts,
        )
        return None

    def _detect_content_type(self, data: bytes) -> str:
        """Detect content type from PDF data."""
        # Check PDF magic bytes
        if data.startswith(b"%PDF-"):
            return "application/pdf"

        # Fallback to generic binary
        return "application/octet-stream"

    def _create_in_memory_artifact(
        self,
        pdf_url: str,
        pdf_data: bytes,
        content_type: str,
        document_id: str | None = None,
        tenant_id: str | None = None,
    ) -> DownloadArtifact:
        """Create artifact for in-memory storage."""
        checksum = hashlib.sha256(pdf_data).hexdigest()
        resolved_document = document_id or f"doc-{int(time.time())}"
        resolved_tenant = tenant_id or "default"

        return DownloadArtifact(
            document_id=resolved_document,
            tenant_id=resolved_tenant,
            uri=f"in-memory://{checksum}",
            metadata={
                "checksum": checksum,
                "size": len(pdf_data),
                "content_type": content_type,
                "pdf_url": pdf_url,
                "in_memory": True,
                "vlm_ready": True,
                "backends": ["docling_vlm"],
            },
        )


# Async version for compatibility with existing code
class AsyncStorageAwarePdfDownloadStage(StorageAwarePdfDownloadStage):
    """Async version of the storage-aware PDF download stage."""

    async def execute(self, ctx: StageContext, state: PipelineState) -> list[DownloadArtifact]:
        """Async version of execute method."""
        pdf_urls = state.metadata.get("pdf_urls", [])
        if not pdf_urls:
            logger.debug("pdf_download_stage.no_urls", tenant_id=state.tenant_id)
            return []

        artifacts: list[DownloadArtifact] = []

        for pdf_url in pdf_urls:
            try:
                artifact = await self._download_and_store_pdf_async(
                    ctx=ctx,
                    state=state,
                    pdf_url=pdf_url,
                )
                if artifact:
                    artifacts.append(artifact)
            except Exception as e:
                logger.error(
                    "pdf_download_stage.error",
                    tenant_id=state.tenant_id,
                    pdf_url=pdf_url,
                    error=str(e),
                )
                continue

        logger.info(
            "pdf_download_stage.completed",
            tenant_id=state.tenant_id,
            total_urls=len(pdf_urls),
            successful_downloads=len(artifacts),
        )

        if artifacts and get_settings().feature_flags.pdf_processing_backend == "docling_vlm":
            state.mark_pdf_vlm_ready(metadata={"backend": "docling_vlm"})

        return artifacts

    async def _download_and_store_pdf_async(
        self,
        ctx: StageContext,
        state: PipelineState,
        pdf_url: str,
    ) -> DownloadArtifact | None:
        """Async version of download and store."""
        # Generate document ID if not present
        document_id = state.document_id or f"doc-{int(time.time())}"

        # Download PDF with retries
        pdf_data = await self._download_with_retries_async(pdf_url)
        if not pdf_data:
            return None

        # Validate content type
        content_type = self._detect_content_type(pdf_data)
        if content_type not in self._config.allowed_content_types:
            logger.warning(
                "pdf_download_stage.invalid_content_type",
                tenant_id=state.tenant_id,
                pdf_url=pdf_url,
                content_type=content_type,
            )
            return None

        # Store in object storage if available
        if self._pdf_storage:
            try:
                asset = await self._pdf_storage.store_pdf(
                    tenant_id=state.tenant_id,
                    document_id=document_id,
                    pdf_data=pdf_data,
                    content_type=content_type,
                )

                logger.info(
                    "pdf_download_stage.stored",
                    tenant_id=state.tenant_id,
                    document_id=document_id,
                    pdf_url=pdf_url,
                    s3_key=asset.s3_key,
                    size=len(pdf_data),
                    checksum=asset.checksum,
                )

                return DownloadArtifact(
                    document_id=document_id,
                    tenant_id=state.tenant_id,
                    uri=f"s3://{asset.s3_key}",
                    metadata={
                        "checksum": asset.checksum,
                        "size": len(pdf_data),
                        "content_type": content_type,
                        "pdf_url": pdf_url,
                    },
                )
            except Exception as e:
                logger.error(
                    "pdf_download_stage.storage_error",
                    tenant_id=state.tenant_id,
                    pdf_url=pdf_url,
                    error=str(e),
                )
                return self._create_in_memory_artifact(
                    pdf_url, pdf_data, content_type, document_id, state.tenant_id
                )
        else:
            return self._create_in_memory_artifact(pdf_url, pdf_data, content_type)

    async def _download_with_retries_async(self, pdf_url: str) -> bytes | None:
        """Async version of download with retries."""
        for attempt in range(self._config.retry_attempts):
            try:
                async with httpx.AsyncClient(
                    timeout=self._config.timeout_seconds,
                    follow_redirects=True,
                    max_redirects=self._config.max_redirects,
                ) as client:
                    response = await client.get(pdf_url)
                    response.raise_for_status()

                    # Check content length
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > self._config.max_file_size:
                        logger.warning(
                            "pdf_download_stage.file_too_large",
                            pdf_url=pdf_url,
                            content_length=int(content_length),
                            max_size=self._config.max_file_size,
                        )
                        return None

                    pdf_data = response.content

                    # Check actual size
                    if len(pdf_data) > self._config.max_file_size:
                        logger.warning(
                            "pdf_download_stage.file_too_large",
                            pdf_url=pdf_url,
                            actual_size=len(pdf_data),
                            max_size=self._config.max_file_size,
                        )
                        return None

                    return pdf_data

            except httpx.TimeoutException:
                logger.warning(
                    "pdf_download_stage.timeout",
                    pdf_url=pdf_url,
                    attempt=attempt + 1,
                    timeout=self._config.timeout_seconds,
                )
            except httpx.HTTPStatusError as e:
                logger.warning(
                    "pdf_download_stage.http_error",
                    pdf_url=pdf_url,
                    attempt=attempt + 1,
                    status_code=e.response.status_code,
                )
            except Exception as e:
                logger.warning(
                    "pdf_download_stage.download_error",
                    pdf_url=pdf_url,
                    attempt=attempt + 1,
                    error=str(e),
                )

            # Wait before retry
            if attempt < self._config.retry_attempts - 1:
                wait_time = self._config.retry_backoff_multiplier * (2**attempt)
                await asyncio.sleep(wait_time)

        logger.error(
            "pdf_download_stage.download_failed",
            pdf_url=pdf_url,
            attempts=self._config.retry_attempts,
        )
        return None
