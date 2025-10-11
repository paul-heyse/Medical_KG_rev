"""Built-in stage plugins for orchestration pipeline."""

from __future__ import annotations

import logging
from typing import Any

import structlog

from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stages.plugins import (
    StagePlugin,
    StagePluginHealth,
    StagePluginRegistration,
)

logger = structlog.get_logger(__name__)


class PdfTwoPhasePlugin(StagePlugin):
    """Plugin providing download and gate stages for the pdf-two-phase pipeline.

    This plugin specializes in PDF processing workflows, providing stages for
    downloading PDFs and gating pipeline execution based on backend readiness.
    It's designed for two-phase PDF processing where the first phase handles
    PDF acquisition and the second phase waits for Docling processing
    completion depending on the active feature flag.

    Attributes
    ----------
    _ledger: Job ledger for tracking PDF processing state

    Thread Safety:
    Thread-safe once initialized. Stage instances created by this plugin
    should be stateless or thread-safe.
    """

    def __init__(self, ledger: JobLedger | None = None) -> None:
        """Initialize the PDF two-phase plugin."""
        super().__init__("pdf-two-phase", dependencies=())
        self._ledger = ledger or JobLedger()
        self.logger = logger

    def registrations(self, resources: Any) -> tuple[StagePluginRegistration, ...]:
        """Return stage plugin registrations."""
        return (
            self._create_download_registration(),
            self._create_gate_registration(),
        )

    def _create_download_registration(self) -> StagePluginRegistration:
        """Create download stage registration."""
        def download_builder(config: dict[str, Any]) -> Any:
            """Build download stage."""
            return DownloadStage(config)

        return StagePluginRegistration(
            stage_type="download",
            builder=download_builder,
            capabilities=("pdf_download", "url_processing"),
        )

    def _create_gate_registration(self) -> StagePluginRegistration:
        """Create gate stage registration."""
        def gate_builder(config: dict[str, Any]) -> Any:
            """Build gate stage."""
            return GateStage(config)

        return StagePluginRegistration(
            stage_type="gate",
            builder=gate_builder,
            capabilities=("pipeline_gating", "backend_readiness"),
        )

    def health_check(self) -> StagePluginHealth:
        """Check plugin health."""
        try:
            # Check ledger health
            ledger_status = self._ledger.get_status()

            return StagePluginHealth(
                status="ok" if ledger_status.get("healthy", False) else "degraded",
                detail=f"PDF two-phase plugin - Ledger: {ledger_status.get('status', 'unknown')}",
                timestamp=ledger_status.get("timestamp", 0.0),
            )
        except Exception as exc:
            return StagePluginHealth(
                status="error",
                detail=f"PDF two-phase plugin health check failed: {exc}",
                timestamp=0.0,
            )


class DownloadStage:
    """Stage for downloading PDFs."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the download stage."""
        self.config = config
        self.logger = logger

    def execute(self, context: Any) -> Any:
        """Execute download stage."""
        try:
            self.logger.info("Executing download stage")

            # Mock download implementation
            url = context.get("url", "https://example.com/document.pdf")
            document_id = context.get("document_id", "doc-123")

            # Simulate download
            download_result = {
                "document_id": document_id,
                "url": url,
                "status": "downloaded",
                "size": 1024,
                "content_type": "application/pdf",
            }

            # Update context
            context["download_result"] = download_result

            return context

        except Exception as exc:
            self.logger.error(f"Download stage failed: {exc}")
            raise exc


class GateStage:
    """Stage for gating pipeline execution."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the gate stage."""
        self.config = config
        self.logger = logger

    def execute(self, context: Any) -> Any:
        """Execute gate stage."""
        try:
            self.logger.info("Executing gate stage")

            # Check backend readiness
            backend_ready = self._check_backend_readiness()

            if not backend_ready:
                self.logger.warning("Backend not ready, gating pipeline")
                context["gated"] = True
                context["gate_reason"] = "backend_not_ready"
                return context

            # Backend is ready, continue
            context["gated"] = False
            context["gate_reason"] = "backend_ready"

            return context

        except Exception as exc:
            self.logger.error(f"Gate stage failed: {exc}")
            raise exc

    def _check_backend_readiness(self) -> bool:
        """Check if backend is ready."""
        # Mock implementation
        return True


class EmbeddingPlugin(StagePlugin):
    """Plugin providing embedding stages for the orchestration pipeline."""

    def __init__(self) -> None:
        """Initialize the embedding plugin."""
        super().__init__("embedding", dependencies=())

    def registrations(self, resources: Any) -> tuple[StagePluginRegistration, ...]:
        """Return stage plugin registrations."""
        return (
            self._create_embedding_registration(),
        )

    def _create_embedding_registration(self) -> StagePluginRegistration:
        """Create embedding stage registration."""
        def embedding_builder(config: dict[str, Any]) -> Any:
            """Build embedding stage."""
            return EmbeddingStage(config)

        return StagePluginRegistration(
            stage_type="embedding",
            builder=embedding_builder,
            capabilities=("text_embedding", "vector_generation"),
        )

    def health_check(self) -> StagePluginHealth:
        """Check plugin health."""
        return StagePluginHealth(
            status="ok",
            detail="Embedding plugin healthy",
            timestamp=0.0,
        )


class EmbeddingStage:
    """Stage for generating embeddings."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the embedding stage."""
        self.config = config
        self.logger = logger

    def execute(self, context: Any) -> Any:
        """Execute embedding stage."""
        try:
            self.logger.info("Executing embedding stage")

            # Get documents from context
            documents = context.get("documents", [])
            if not documents:
                raise ValueError("No documents found in context")

            # Generate embeddings
            embeddings = self._generate_embeddings(documents)

            # Update context
            context["embeddings"] = embeddings

            return context

        except Exception as exc:
            self.logger.error(f"Embedding stage failed: {exc}")
            raise exc

    def _generate_embeddings(self, documents: list[Any]) -> list[Any]:
        """Generate embeddings for documents."""
        # Mock implementation
        embeddings = []
        for i, doc in enumerate(documents):
            embedding = {
                "id": f"emb-{i}",
                "document_id": getattr(doc, "id", f"doc-{i}"),
                "vector": [0.1] * 384,  # Mock 384-dimensional vector
                "model": "mock-model",
            }
            embeddings.append(embedding)
        return embeddings


class ChunkingPlugin(StagePlugin):
    """Plugin providing chunking stages for the orchestration pipeline."""

    def __init__(self) -> None:
        """Initialize the chunking plugin."""
        super().__init__("chunking", dependencies=())

    def registrations(self, resources: Any) -> tuple[StagePluginRegistration, ...]:
        """Return stage plugin registrations."""
        return (
            self._create_chunking_registration(),
        )

    def _create_chunking_registration(self) -> StagePluginRegistration:
        """Create chunking stage registration."""
        def chunking_builder(config: dict[str, Any]) -> Any:
            """Build chunking stage."""
            return ChunkingStage(config)

        return StagePluginRegistration(
            stage_type="chunking",
            builder=chunking_builder,
            capabilities=("text_chunking", "document_segmentation"),
        )

    def health_check(self) -> StagePluginHealth:
        """Check plugin health."""
        return StagePluginHealth(
            status="ok",
            detail="Chunking plugin healthy",
            timestamp=0.0,
        )


class ChunkingStage:
    """Stage for chunking documents."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the chunking stage."""
        self.config = config
        self.logger = logger

    def execute(self, context: Any) -> Any:
        """Execute chunking stage."""
        try:
            self.logger.info("Executing chunking stage")

            # Get documents from context
            documents = context.get("documents", [])
            if not documents:
                raise ValueError("No documents found in context")

            # Generate chunks
            chunks = self._generate_chunks(documents)

            # Update context
            context["chunks"] = chunks

            return context

        except Exception as exc:
            self.logger.error(f"Chunking stage failed: {exc}")
            raise exc

    def _generate_chunks(self, documents: list[Any]) -> list[Any]:
        """Generate chunks for documents."""
        # Mock implementation
        chunks = []
        for i, doc in enumerate(documents):
            for j in range(3):  # 3 chunks per document
                chunk = {
                    "id": f"chunk-{i}-{j}",
                    "document_id": getattr(doc, "id", f"doc-{i}"),
                    "text": f"Chunk {j} of document {i}",
                    "position": j,
                }
                chunks.append(chunk)
        return chunks


def get_builtin_plugins() -> list[StagePlugin]:
    """Get all built-in stage plugins."""
    return [
        PdfTwoPhasePlugin(),
        EmbeddingPlugin(),
        ChunkingPlugin(),
    ]
