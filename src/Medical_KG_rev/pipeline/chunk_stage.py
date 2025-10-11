"""Chunk stage for pipeline processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import structlog

from Medical_KG_rev.chunking.models import ChunkerConfig
from Medical_KG_rev.pipeline.stages import ChunkStage, StageResult, StageStatus

logger = structlog.get_logger(__name__)


@dataclass
class ChunkStageConfig:
    """Configuration for chunk stage."""

    chunker_name: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    tokenizer_name: str = "bert-base-uncased"
    max_chunks: int = 1000


class ChunkStageImpl(ChunkStage):
    """Implementation of chunk stage."""

    def __init__(self, config: ChunkStageConfig) -> None:
        """Initialize the chunk stage."""
        super().__init__()
        self.config = config
        self.logger = logger
        self._tokenizer = None

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

            # Initialize tokenizer if needed
            if not self._tokenizer:
                self._initialize_tokenizer()

            # Process documents
            chunks = []
            for doc in documents:
                doc_chunks = self._chunk_document(doc)
                chunks.extend(doc_chunks)

            # Limit chunks if configured
            if self.config.max_chunks and len(chunks) > self.config.max_chunks:
                chunks = chunks[:self.config.max_chunks]
                self.logger.warning(f"Limited chunks to {self.config.max_chunks}")

            # Update context
            context["chunks"] = chunks
            context["chunk_count"] = len(chunks)

            return StageResult(
                status=StageStatus.COMPLETED,
                data=context,
                metadata={
                    "chunker": self.config.chunker_name,
                    "chunk_count": len(chunks),
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                },
            )

        except Exception as exc:
            self.logger.error(f"Chunk stage failed: {exc}")
            return StageResult(
                status=StageStatus.FAILED,
                error=str(exc),
                data=context,
            )

    def _initialize_tokenizer(self) -> None:
        """Initialize tokenizer."""
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            self.logger.info(f"Tokenizer initialized: {self.config.tokenizer_name}")
        except ImportError:
            self.logger.warning("Transformers not available, using simple tokenizer")
            self._tokenizer = "simple"
        except Exception as exc:
            self.logger.warning(f"Failed to initialize tokenizer: {exc}")
            self._tokenizer = "simple"

    def _chunk_document(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        """Chunk a document."""
        try:
            content = document.get("content", "")
            if not content:
                return []

            # Simple chunking implementation
            chunks = []
            words = content.split()

            current_chunk = []
            current_size = 0

            for word in words:
                word_size = self._count_tokens(word)

                if current_size + word_size > self.config.chunk_size and current_chunk:
                    # Create chunk
                    chunk_text = " ".join(current_chunk)
                    chunk = {
                        "id": f"chunk-{len(chunks)}",
                        "text": chunk_text,
                        "metadata": {
                            "document_id": document.get("id", "unknown"),
                            "chunk_size": len(chunk_text),
                            "token_count": current_size,
                            "chunker": self.config.chunker_name,
                        },
                    }
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    overlap_size = int(self.config.chunk_overlap * 0.1)  # 10% of chunk size
                    current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                    current_size = sum(self._count_tokens(w) for w in current_chunk)

                current_chunk.append(word)
                current_size += word_size

            # Add final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk = {
                    "id": f"chunk-{len(chunks)}",
                    "text": chunk_text,
                    "metadata": {
                        "document_id": document.get("id", "unknown"),
                        "chunk_size": len(chunk_text),
                        "token_count": current_size,
                        "chunker": self.config.chunker_name,
                    },
                }
                chunks.append(chunk)

            return chunks

        except Exception as exc:
            self.logger.error(f"Failed to chunk document: {exc}")
            return []

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer == "simple":
            return len(text.split())
        elif self._tokenizer and hasattr(self._tokenizer, 'tokenize'):
            try:
                tokens = self._tokenizer.tokenize(text)
                return len(tokens)
            except Exception:
                return len(text.split())
        else:
            return len(text.split())

    def health_check(self) -> dict[str, Any]:
        """Check stage health."""
        health = {
            "stage": "chunk",
            "config": {
                "chunker_name": self.config.chunker_name,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "tokenizer_name": self.config.tokenizer_name,
                "max_chunks": self.config.max_chunks,
            },
        }

        # Check tokenizer health
        try:
            if self._tokenizer and hasattr(self._tokenizer, 'tokenize'):
                # Test tokenizer
                test_text = "test"
                tokens = self._tokenizer.tokenize(test_text)
                health["tokenizer_status"] = "healthy"
                health["tokenizer_name"] = self.config.tokenizer_name
            else:
                health["tokenizer_status"] = "simple"
                health["tokenizer_name"] = "simple"
        except Exception as e:
            health["tokenizer_status"] = "unhealthy"
            health["tokenizer_error"] = str(e)

        return health

    def get_config(self) -> ChunkStageConfig:
        """Get stage configuration."""
        return self.config

    def update_config(self, config: ChunkStageConfig) -> None:
        """Update stage configuration."""
        self.config = config
        self.logger.info("Chunk stage configuration updated")


def create_chunk_stage(config: ChunkStageConfig) -> ChunkStageImpl:
    """Create a chunk stage instance."""
    return ChunkStageImpl(config)


def create_default_chunk_stage_config() -> ChunkStageConfig:
    """Create default chunk stage configuration."""
    return ChunkStageConfig(
        chunker_name="default",
        chunk_size=1000,
        chunk_overlap=200,
        tokenizer_name="bert-base-uncased",
        max_chunks=1000,
    )
