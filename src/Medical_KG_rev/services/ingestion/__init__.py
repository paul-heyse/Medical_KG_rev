"""Document ingestion pipeline integrating chunking, embeddings, and storage."""

from .service import (
    EmbeddingBatchMetrics,
    IngestionOptions,
    IngestionResult,
    IngestionService,
)

__all__ = [
    "EmbeddingBatchMetrics",
    "IngestionOptions",
    "IngestionResult",
    "IngestionService",
]
