"""Vector store GPU utilities - Torch-free version."""

from typing import Any

import structlog

from ..clients.embedding_client import EmbeddingClientManager

logger = structlog.get_logger(__name__)


class GPUResourceManager:
    """GPU resource manager for vector store operations."""

    def __init__(self):
        self._client_manager: EmbeddingClientManager | None = None

    async def initialize(self) -> None:
        """Initialize GPU resources."""
        pass

    async def cleanup(self) -> None:
        """Cleanup GPU resources."""
        pass


class VectorStoreGPU:
    """Vector store GPU operations via gRPC services."""

    def __init__(self):
        self._client_manager: EmbeddingClientManager | None = None

    async def _get_client_manager(self) -> EmbeddingClientManager:
        """Get or create embedding client manager."""
        if self._client_manager is None:
            self._client_manager = EmbeddingClientManager()
            await self._client_manager.initialize()
        return self._client_manager

    async def generate_embeddings(
        self, texts: list[str], model: str = "default"
    ) -> list[list[float]]:
        """Generate embeddings via gRPC service."""
        try:
            client_manager = await self._get_client_manager()
            embeddings = await client_manager.generate_embeddings_batch(texts, model)
            return embeddings
        except Exception as e:
            logger.error("vector_store.embedding.error", error=str(e))
            raise RuntimeError(f"Embedding generation failed: {e}")

    async def similarity_search(
        self, query_embedding: list[float], index_embeddings: list[list[float]], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Perform similarity search using embeddings."""
        try:
            # Simple cosine similarity implementation
            import numpy as np

            query_np = np.array(query_embedding)
            similarities = []

            for i, embedding in enumerate(index_embeddings):
                embedding_np = np.array(embedding)
                similarity = np.dot(query_np, embedding_np) / (
                    np.linalg.norm(query_np) * np.linalg.norm(embedding_np)
                )
                similarities.append(
                    {"index": i, "similarity": float(similarity), "embedding": embedding}
                )

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error("vector_store.similarity.error", error=str(e))
            raise RuntimeError(f"Similarity search failed: {e}")

    async def close(self) -> None:
        """Close the embedding client manager."""
        if self._client_manager:
            await self._client_manager.close()


# Legacy functions (replaced with gRPC service calls)
def generate_embeddings(texts: list[str], model: str = "default") -> list[list[float]]:
    """Legacy function - embedding generation moved to gRPC services."""
    raise NotImplementedError(
        "Embedding generation moved to gRPC services. Use VectorStoreGPU instead."
    )


def similarity_search(
    query_embedding: list[float], index_embeddings: list[list[float]], top_k: int = 10
) -> list[dict[str, Any]]:
    """Legacy function - similarity search moved to gRPC services."""
    raise NotImplementedError(
        "Similarity search moved to gRPC services. Use VectorStoreGPU instead."
    )
