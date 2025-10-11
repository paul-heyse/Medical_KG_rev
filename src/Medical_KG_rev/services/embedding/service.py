"""Embedding service - placeholder implementation removed to surface real issues.

Key Responsibilities:
    - Define the core embedding service interface and data models
    - Provide request/response structures for embedding operations
    - Serve as the foundation for embedding service implementations
    - Define the contract for embedding workers and clients

Collaborators:
    - Upstream: Embedding clients and coordinators request embeddings
    - Downstream: Concrete embedding implementations (gRPC services, local models)

Side Effects:
    - None: Pure data models and abstract interfaces

Thread Safety:
    - Thread-safe: All dataclasses are immutable

Performance Characteristics:
    - O(1) data structure operations
    - Memory usage proportional to number of texts and vectors

Example:
    >>> request = EmbeddingRequest(
    ...     tenant_id="tenant1",
    ...     texts=["Hello world", "How are you?"]
    ... )
    >>> # This would normally be implemented by a concrete service
    >>> # result = embedding_worker.run(request)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(slots=True)
class EmbeddingRequest:
    """Request for generating embeddings from text.

    Attributes:
        tenant_id: Unique identifier for the tenant requesting embeddings.
        texts: List of text strings to be embedded.
        normalize: Whether to normalize the resulting vectors (L2 normalization).
    """
    tenant_id: str
    texts: List[str]
    normalize: bool = False


@dataclass(slots=True)
class EmbeddingVector:
    """Single embedding vector result.

    Attributes:
        text: Original text that was embedded.
        model: Name of the embedding model used.
        values: Vector representation of the text.
    """
    text: str
    model: str
    values: List[float]


@dataclass(slots=True)
class EmbeddingResponse:
    """Response containing multiple embedding vectors.

    Attributes:
        vectors: List of embedding vectors for each input text.
    """
    vectors: List[EmbeddingVector]


class EmbeddingWorker:
    """Abstract base class for embedding service implementations.

    This class defines the interface that concrete embedding services must implement.
    It serves as the contract for generating embeddings from text inputs.

    Attributes:
        None (abstract base class)

    Invariants:
        - Implementations must be stateless or properly thread-safe
        - Must handle empty text lists gracefully
        - Must validate tenant_id and normalize embeddings

    Thread Safety:
        - Implementations must be thread-safe for concurrent requests

    Lifecycle:
        - No explicit lifecycle management required
        - Implementations may cache models or resources as needed

    Example:
        >>> class MyEmbeddingWorker(EmbeddingWorker):
        ...     def run(self, request):
        ...         # Implementation here
        ...         return EmbeddingResponse(vectors=[])
    """

    def run(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Process an embedding request and return vectorized results.

        Args:
            request: Embedding request containing tenant context and texts to embed.

        Returns:
            Response containing embedding vectors for each input text.

        Raises:
            NotImplementedError: Always raised by this abstract implementation.
            ValueError: If request contains invalid data.
            RuntimeError: If embedding service encounters an error.
        """
        raise NotImplementedError(
            "EmbeddingWorker.run() not implemented. "
            "This service requires a real embedding implementation. "
            "Please implement or configure a proper embedding service."
        )


__all__ = ["EmbeddingRequest", "EmbeddingVector", "EmbeddingResponse", "EmbeddingWorker"]
