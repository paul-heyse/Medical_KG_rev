"""Vector store GPU utilities."""

from typing import Any

import torch

import structlog

logger = structlog.get_logger(__name__)


class VectorStoreGPU:
    """Vector store GPU operations."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self, model_name: str) -> None:
        """Load a model for vector operations."""
        try:
            # This would load the actual model
            self.model = torch.load(model_name, map_location=self.device)
            logger.info("vector_store.model.loaded", model=model_name, device=self.device)
        except Exception as e:
            logger.error("vector_store.model.load_error", model=model_name, error=str(e))
            raise

    def generate_embeddings(self, texts: list[str], model: str = "default") -> list[list[float]]:
        """Generate embeddings using GPU."""
        try:
            if self.model is None:
                self.load_model(model)

            # Convert texts to tensors
            inputs = torch.tensor(texts, device=self.device)

            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(inputs)

            # Convert back to CPU and return as list
            return embeddings.cpu().numpy().tolist()

        except Exception as e:
            logger.error("vector_store.embedding.error", error=str(e))
            raise RuntimeError(f"Embedding generation failed: {e}")

    def similarity_search(
        self, query_embedding: list[float], index_embeddings: list[list[float]], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Perform similarity search using GPU."""
        try:
            # Convert to tensors
            query_tensor = torch.tensor(query_embedding, device=self.device)
            index_tensor = torch.tensor(index_embeddings, device=self.device)

            # Compute similarities
            with torch.no_grad():
                similarities = torch.cosine_similarity(
                    query_tensor.unsqueeze(0), index_tensor, dim=1
                )

            # Get top-k results
            top_indices = torch.topk(similarities, top_k).indices
            top_similarities = torch.topk(similarities, top_k).values

            results = []
            for idx, sim in zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy(), strict=False):
                results.append(
                    {
                        "index": int(idx),
                        "similarity": float(sim),
                        "embedding": index_embeddings[int(idx)],
                    }
                )

            return results

        except Exception as e:
            logger.error("vector_store.similarity.error", error=str(e))
            raise RuntimeError(f"Similarity search failed: {e}")


# Legacy functions
def generate_embeddings(texts: list[str], model: str = "default") -> list[list[float]]:
    """Legacy function for generating embeddings."""
    store = VectorStoreGPU()
    return store.generate_embeddings(texts, model)


def similarity_search(
    query_embedding: list[float], index_embeddings: list[list[float]], top_k: int = 10
) -> list[dict[str, Any]]:
    """Legacy function for similarity search."""
    store = VectorStoreGPU()
    return store.similarity_search(query_embedding, index_embeddings, top_k)
