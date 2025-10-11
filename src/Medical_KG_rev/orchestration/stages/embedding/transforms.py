"""Result transformation utilities for embedding operations."""

from __future__ import annotations

from typing import Any

from Medical_KG_rev.orchestration.stages.embedding.contracts import EmbeddingResult


def result_to_context_data(result: EmbeddingResult) -> dict[str, Any]:
    """Convert EmbeddingResult to legacy context.data format.

    For backward compatibility with stages expecting old format.
    """
    return {
        "embeddings": [
            {
                "chunk_id": vec.chunk_id,
                "vector": list(vec.vector),
                "model_id": vec.model_id,
                "namespace": vec.namespace,
                "metadata": vec.metadata,
            }
            for vec in result.vectors
        ],
        "metrics": {
            "embedding": {
                "vectors": result.vector_count,
                "processing_time_ms": result.processing_time_ms,
            }
        },
        "embedding_summary": {
            "vectors": result.vector_count,
            "per_namespace": result.per_namespace_counts,
            "model_id": result.model_id,
            "timestamp": result.timestamp.isoformat(),
        }
    }


def context_data_to_result(data: dict[str, Any]) -> EmbeddingResult:
    """Convert legacy context.data format to EmbeddingResult.

    For migrating existing pipeline state.
    """
    from Medical_KG_rev.orchestration.stages.embedding.contracts import (
        EmbeddingResult,
        EmbeddingVector,
    )

    vectors = tuple(
        EmbeddingVector(
            chunk_id=emb["chunk_id"],
            vector=tuple(emb["vector"]),
            model_id=emb["model_id"],
            namespace=emb["namespace"],
            metadata=emb.get("metadata", {}),
        )
        for emb in data["embeddings"]
    )

    return EmbeddingResult(
        vectors=vectors,
        model_id=data["embedding_summary"]["model_id"],
        namespace=vectors[0].namespace if vectors else "default",
        processing_time_ms=data["metrics"]["embedding"]["processing_time_ms"],
    )
