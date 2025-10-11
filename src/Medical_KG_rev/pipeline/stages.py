"""Pipeline stages for document processing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from Medical_KG_rev.chunking.hybrid_chunker import HybridChunker
from Medical_KG_rev.chunking.registry import get_chunker
from Medical_KG_rev.services.parsing.docling_vlm_service import (
    DoclingVLMService,
)
from Medical_KG_rev.services.retrieval.bm25_service import BM25Service
from Medical_KG_rev.services.retrieval.qwen3_service import Qwen3Service
from Medical_KG_rev.services.retrieval.splade_service import SPLADEService
from Medical_KG_rev.services.vector_store.stores.bm25_index import BM25Index
from Medical_KG_rev.services.vector_store.stores.qwen3_index import Qwen3Index
from Medical_KG_rev.services.vector_store.stores.splade_index import SPLADEIndex
from Medical_KG_rev.storage.chunk_store import ChunkRecord, ChunkStore

logger = structlog.get_logger(__name__)


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""

    status: StageStatus
    data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None

    def is_success(self) -> bool:
        """Check if the stage completed successfully."""
        return self.status == StageStatus.COMPLETED

    def is_failure(self) -> bool:
        """Check if the stage failed."""
        return self.status == StageStatus.FAILED


class BaseStage(ABC):
    """Base class for pipeline stages."""

    def __init__(self) -> None:
        """Initialize the stage."""
        self.logger = logger

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> StageResult:
        """Execute the stage."""
        pass

    def health_check(self) -> dict[str, Any]:
        """Check stage health."""
        return {"stage": self.__class__.__name__, "status": "healthy"}

    def get_config(self) -> dict[str, Any]:
        """Get stage configuration."""
        return {}

    def update_config(self, config: dict[str, Any]) -> None:
        """Update stage configuration."""
        pass


class ChunkStage(BaseStage):
    """Stage for chunking documents."""

    def __init__(self, chunker_name: str = "hybrid") -> None:
        """Initialize the chunk stage."""
        super().__init__()
        self.chunker_name = chunker_name
        self.chunker = get_chunker(chunker_name)

    def execute(self, context: dict[str, Any]) -> StageResult:
        """Execute the chunk stage."""
        try:
            self.logger.info("Executing chunk stage")

            # Get documents from context
            documents = context.get("documents", [])
            if not documents:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No documents found in context",
                    data=context,
                )

            # Chunk documents
            chunks = []
            for doc in documents:
                doc_chunks = self.chunker.chunk(
                    doc,
                    tenant_id=context.get("tenant_id", "default"),
                    granularity="paragraph",
                )
                chunks.extend(doc_chunks)

            # Update context
            context["chunks"] = chunks
            context["chunk_count"] = len(chunks)

            return StageResult(
                status=StageStatus.COMPLETED,
                data=context,
                metadata={
                    "chunker": self.chunker_name,
                    "chunk_count": len(chunks),
                },
            )

        except Exception as exc:
            self.logger.error(f"Chunk stage failed: {exc}")
            return StageResult(
                status=StageStatus.FAILED,
                error=str(exc),
                data=context,
            )


class ConvertStage(BaseStage):
    """Stage for converting documents."""

    def __init__(self) -> None:
        """Initialize the convert stage."""
        super().__init__()
        self.docling_service = DoclingVLMService()

    def execute(self, context: dict[str, Any]) -> StageResult:
        """Execute the convert stage."""
        try:
            self.logger.info("Executing convert stage")

            # Get documents from context
            documents = context.get("documents", [])
            if not documents:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No documents found in context",
                    data=context,
                )

            # Convert documents
            converted_documents = []
            for doc in documents:
                try:
                    converted_doc = self.docling_service.process_document(doc)
                    converted_documents.append(converted_doc)
                except Exception as exc:
                    self.logger.warning(f"Failed to convert document: {exc}")
                    continue

            # Update context
            context["converted_documents"] = converted_documents
            context["conversion_count"] = len(converted_documents)

            return StageResult(
                status=StageStatus.COMPLETED,
                data=context,
                metadata={
                    "converted_count": len(converted_documents),
                    "total_count": len(documents),
                },
            )

        except Exception as exc:
            self.logger.error(f"Convert stage failed: {exc}")
            return StageResult(
                status=StageStatus.FAILED,
                error=str(exc),
                data=context,
            )


class EmbeddingStage(BaseStage):
    """Stage for generating embeddings."""

    def __init__(self, model_name: str = "qwen3") -> None:
        """Initialize the embedding stage."""
        super().__init__()
        self.model_name = model_name
        self.qwen3_service = Qwen3Service()

    def execute(self, context: dict[str, Any]) -> StageResult:
        """Execute the embedding stage."""
        try:
            self.logger.info("Executing embedding stage")

            # Get chunks from context
            chunks = context.get("chunks", [])
            if not chunks:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No chunks found in context",
                    data=context,
                )

            # Generate embeddings
            embeddings = []
            for chunk in chunks:
                try:
                    embedding = self.qwen3_service.generate_embedding(chunk.text)
                    embeddings.append({
                        "chunk_id": chunk.id,
                        "embedding": embedding,
                        "model": self.model_name,
                    })
                except Exception as exc:
                    self.logger.warning(f"Failed to generate embedding for chunk {chunk.id}: {exc}")
                    continue

            # Update context
            context["embeddings"] = embeddings
            context["embedding_count"] = len(embeddings)

            return StageResult(
                status=StageStatus.COMPLETED,
                data=context,
                metadata={
                    "model": self.model_name,
                    "embedding_count": len(embeddings),
                },
            )

        except Exception as exc:
            self.logger.error(f"Embedding stage failed: {exc}")
            return StageResult(
                status=StageStatus.FAILED,
                error=str(exc),
                data=context,
            )


class RetrievalStage(BaseStage):
    """Stage for document retrieval."""

    def __init__(self, retrieval_type: str = "hybrid") -> None:
        """Initialize the retrieval stage."""
        super().__init__()
        self.retrieval_type = retrieval_type
        self.bm25_service = BM25Service()
        self.splade_service = SPLADEService()

    def execute(self, context: dict[str, Any]) -> StageResult:
        """Execute the retrieval stage."""
        try:
            self.logger.info("Executing retrieval stage")

            # Get query from context
            query = context.get("query", "")
            if not query:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No query found in context",
                    data=context,
                )

            # Perform retrieval
            if self.retrieval_type == "bm25":
                results = self.bm25_service.search(query, top_k=10)
            elif self.retrieval_type == "splade":
                results = self.splade_service.search(query, top_k=10)
            else:  # hybrid
                bm25_results = self.bm25_service.search(query, top_k=10)
                splade_results = self.splade_service.search(query, top_k=10)
                results = self._combine_results(bm25_results, splade_results)

            # Update context
            context["retrieval_results"] = results
            context["result_count"] = len(results)

            return StageResult(
                status=StageStatus.COMPLETED,
                data=context,
                metadata={
                    "retrieval_type": self.retrieval_type,
                    "result_count": len(results),
                },
            )

        except Exception as exc:
            self.logger.error(f"Retrieval stage failed: {exc}")
            return StageResult(
                status=StageStatus.FAILED,
                error=str(exc),
                data=context,
            )

    def _combine_results(self, bm25_results: list[Any], splade_results: list[Any]) -> list[Any]:
        """Combine BM25 and SPLADE results."""
        # Simple combination - in practice, you'd want more sophisticated fusion
        combined = bm25_results + splade_results
        return combined[:10]  # Return top 10


class StorageStage(BaseStage):
    """Stage for storing results."""

    def __init__(self) -> None:
        """Initialize the storage stage."""
        super().__init__()
        self.chunk_store = ChunkStore()

    def execute(self, context: dict[str, Any]) -> StageResult:
        """Execute the storage stage."""
        try:
            self.logger.info("Executing storage stage")

            # Get chunks from context
            chunks = context.get("chunks", [])
            if not chunks:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No chunks found in context",
                    data=context,
                )

            # Store chunks
            stored_count = 0
            for chunk in chunks:
                try:
                    chunk_record = ChunkRecord(
                        id=chunk.id,
                        text=chunk.text,
                        metadata=chunk.metadata,
                        tenant_id=context.get("tenant_id", "default"),
                    )
                    self.chunk_store.store(chunk_record)
                    stored_count += 1
                except Exception as exc:
                    self.logger.warning(f"Failed to store chunk {chunk.id}: {exc}")
                    continue

            # Update context
            context["stored_count"] = stored_count

            return StageResult(
                status=StageStatus.COMPLETED,
                data=context,
                metadata={
                    "stored_count": stored_count,
                    "total_count": len(chunks),
                },
            )

        except Exception as exc:
            self.logger.error(f"Storage stage failed: {exc}")
            return StageResult(
                status=StageStatus.FAILED,
                error=str(exc),
                data=context,
            )


def create_stage(stage_type: str, config: dict[str, Any] | None = None) -> BaseStage:
    """Create a stage instance."""
    config = config or {}

    if stage_type == "chunk":
        return ChunkStage(config.get("chunker_name", "hybrid"))
    elif stage_type == "convert":
        return ConvertStage()
    elif stage_type == "embedding":
        return EmbeddingStage(config.get("model_name", "qwen3"))
    elif stage_type == "retrieval":
        return RetrievalStage(config.get("retrieval_type", "hybrid"))
    elif stage_type == "storage":
        return StorageStage()
    else:
        raise ValueError(f"Unknown stage type: {stage_type}")


def get_available_stages() -> list[str]:
    """Get list of available stage types."""
    return ["chunk", "convert", "embedding", "retrieval", "storage"]
