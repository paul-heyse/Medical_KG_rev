"""Pipeline stage for running Docling's VLM processing over downloaded PDFs."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Iterable

import structlog

from Medical_KG_rev.models.ir import BlockType, Document
from Medical_KG_rev.orchestration.stages.contracts import (
    DownloadArtifact,
    PipelineState,
    StageContext,
)
from Medical_KG_rev.orchestration.stages.plugin_manager import StagePluginContext
from Medical_KG_rev.services.parsing.docling import DoclingVLMOutputParser
from Medical_KG_rev.services.parsing.docling_vlm_service import (
    DoclingVLMResult,
    DoclingVLMService,
)
from Medical_KG_rev.storage.clients import PdfAsset, PdfStorageClient

logger = structlog.get_logger(__name__)


class DoclingVLMProcessingStage:
    """Stage that converts downloaded PDFs into the internal Document IR via Docling."""

    def __init__(
        self,
        service: DoclingVLMService | None = None,
        parser: DoclingVLMOutputParser | None = None,
        *,
        storage_client: PdfStorageClient | None = None,
    ) -> None:
        self._service = service or DoclingVLMService()
        self._parser = parser or DoclingVLMOutputParser()
        self._storage = storage_client

    def initialise(self, context: StagePluginContext) -> None:
        if self._storage is None:
            candidate = context.get("pdf_storage")
            if isinstance(candidate, PdfStorageClient):
                self._storage = candidate

    async def _fetch_from_storage(self, artifact: DownloadArtifact) -> Path | None:
        if not self._storage:
            return None
        checksum = artifact.metadata.get("checksum")
        s3_key = artifact.metadata.get("s3_key")
        asset: PdfAsset | None = None
        if checksum:
            try:
                asset = await self._storage.get_pdf_asset(
                    artifact.tenant_id,
                    artifact.document_id,
                    checksum,
                )
            except Exception as exc:  # pragma: no cover - cache miss or storage issue
                logger.debug(
                    "docling_vlm_stage.asset_lookup_failed",
                    tenant_id=artifact.tenant_id,
                    document_id=artifact.document_id,
                    error=str(exc),
                )
        if asset is None:
            if not s3_key:
                return None
            inferred_checksum = checksum or self._infer_checksum(s3_key)
            if not inferred_checksum:
                return None
            asset = PdfAsset(
                tenant_id=artifact.tenant_id,
                document_id=artifact.document_id,
                checksum=inferred_checksum,
                s3_key=s3_key,
                size=int(artifact.metadata.get("size", 0)),
                content_type=str(artifact.metadata.get("content_type", "application/pdf")),
                upload_timestamp=float(artifact.metadata.get("upload_timestamp", 0.0)),
                cache_key=artifact.metadata.get("cache_key"),
            )
        try:
            payload = await self._storage.get_pdf_data(asset)
        except Exception as exc:  # pragma: no cover - storage failures are unlikely in tests
            logger.warning(
                "docling_vlm_stage.storage_fetch_failed",
                tenant_id=artifact.tenant_id,
                document_id=artifact.document_id,
                error=str(exc),
            )
            return None
        tmp_dir = Path(tempfile.mkdtemp(prefix="docling-vlm-"))
        tmp_path = tmp_dir / f"{artifact.document_id}.pdf"
        tmp_path.write_bytes(payload)
        return tmp_path

    def _resolve_local_path(self, artifact: DownloadArtifact) -> Path | None:
        path_value = artifact.metadata.get("local_path") or artifact.metadata.get("path")
        if path_value:
            candidate = Path(str(path_value))
            if candidate.exists():
                return candidate
        uri = artifact.uri
        if uri.startswith("file://"):
            candidate = Path(uri[7:])
            if candidate.exists():
                return candidate
        candidate = Path(uri)
        if candidate.exists():
            return candidate
        return None

    def _load_pdf(self, artifact: DownloadArtifact) -> Path | None:
        local = self._resolve_local_path(artifact)
        if local is not None:
            return local
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._fetch_from_storage(artifact))
        finally:
            loop.close()

    def _process_artifact(self, artifact: DownloadArtifact) -> DoclingVLMResult | None:
        pdf_path = self._load_pdf(artifact)
        if pdf_path is None:
            logger.error(
                "docling_vlm_stage.pdf_missing",
                tenant_id=artifact.tenant_id,
                document_id=artifact.document_id,
                uri=artifact.uri,
            )
            return None
        return self._service.process_pdf(
            str(pdf_path),
            document_id=artifact.document_id,
        )

    def _ensure_list(self, downloads: Iterable[DownloadArtifact]) -> list[DownloadArtifact]:
        return list(downloads)

    def execute(self, ctx: StageContext, state: PipelineState) -> Document | None:
        downloads = self._ensure_list(state.downloads)
        if not downloads:
            logger.info(
                "docling_vlm_stage.no_downloads",
                tenant_id=state.tenant_id,
                job_id=ctx.job_id,
            )
            return None

        processed: list[Document] = []
        for artifact in downloads:
            try:
                result = self._process_artifact(artifact)
            except Exception as exc:  # pragma: no cover - GPU/runtime issues
                logger.error(
                    "docling_vlm_stage.processing_failed",
                    tenant_id=state.tenant_id,
                    document_id=artifact.document_id,
                    error=str(exc),
                )
                continue
            if result is None:
                continue
            document = self._parser.parse(result)
            processed.append(document)

        if not processed:
            logger.error(
                "docling_vlm_stage.no_documents",
                tenant_id=state.tenant_id,
                job_id=ctx.job_id,
            )
            return None

        primary = processed[0]
        state.set_document(primary)
        state.mark_pdf_vlm_ready(metadata={"backend": "docling_vlm"})
        metadata = state.metadata.setdefault("docling_vlm", {})
        metadata["documents"] = [doc.model_dump(mode="json") for doc in processed]
        metadata["structured_context"] = self._build_structured_context(primary)
        return primary

    @staticmethod
    def _build_structured_context(document: Document) -> list[str]:
        context: list[str] = []
        for block in document.iter_blocks():
            if block.type == BlockType.TABLE and block.table is not None:
                caption = block.table.caption or "Table"
                context.append(f"{caption}\n{block.table.to_markdown()}")
            elif block.type == BlockType.FIGURE and block.figure is not None:
                caption = block.figure.caption or "Figure"
                context.append(caption)
        return context

    @staticmethod
    def _infer_checksum(s3_key: str | None) -> str | None:
        if not s3_key:
            return None
        candidate = Path(s3_key).name
        if candidate.endswith(".pdf"):
            return candidate[:-4]
        return candidate or None
