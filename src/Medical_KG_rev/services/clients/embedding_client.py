"""Async embedding client abstractions and statistics tracking.

Key Responsibilities:
    - Define the abstract contract for asynchronous embedding service clients
    - Provide lightweight statistics tracking for downstream implementations

Collaborators:
    - Upstream: Retrieval services and orchestrators invoke embedding clients
    - Downstream: Concrete HTTP/gRPC embedding services satisfy embedding requests

Side Effects:
    - Concrete subclasses may perform network I/O and emit telemetry when handling requests

Thread Safety:
    - Thread-safe when subclasses avoid shared mutable state; the base class only maintains per-instance statistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingClientStats:
    """Statistics captured during embedding client operations.

    Attributes:
        total_requests: Number of embedding requests attempted.
        successful_requests: Number of requests that completed successfully.
        failed_requests: Number of requests that raised an error.
        total_processing_time: Aggregate time spent generating embeddings in seconds.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0

    @property
    def average_processing_time(self) -> float:
        """Calculate the average processing time per request.

        Returns:
            Average processing time in seconds, or 0.0 if no requests made.
        """
        if self.total_requests == 0:
            return 0.0
        return self.total_processing_time / self.total_requests


# ==============================================================================
# CLIENT INTERFACE
# ==============================================================================


class EmbeddingClient:
    """Abstract base class for async embedding service clients.

    This class defines the interface that concrete embedding clients must implement.
    It serves as the contract for communicating with embedding services.

    Attributes:
        service_endpoint: URL or endpoint of the embedding service.
        stats: Statistics tracking for client operations.

    Invariants:
        - service_endpoint must be a valid URL or service identifier
        - stats object must be initialized before use
        - Client must be initialized before generating embeddings

    Thread Safety:
        - Thread-safe: All async operations can be called concurrently

    Lifecycle:
        - Must call initialize() before use
        - Must call close() when finished to clean up resources
        - Can be reused after initialization

    Example:
        >>> class MyEmbeddingClient(EmbeddingClient):
        ...     async def initialize(self):
        ...         # Setup connection
        ...         pass
        ...
        ...     async def generate_embeddings(self, texts, **kwargs):
        ...         # Implementation here
        ...         return []
    """

    def __init__(self, service_endpoint: str) -> None:
        """Initialize the embedding client.

        Args:
            service_endpoint: URL or identifier for the embedding service.
        """
        self.service_endpoint = service_endpoint
        self.stats = EmbeddingClientStats()

    async def initialize(self) -> None:
        """Initialize the client connection and resources.

        Raises:
            NotImplementedError: Always raised by this abstract implementation.
            RuntimeError: If initialization fails.
        """
        raise NotImplementedError(
            "EmbeddingClient.initialize() not implemented. "
            "This client requires a real embedding service implementation. "
            "Please implement or configure a proper embedding service."
        )

    async def close(self) -> None:
        """Close the client and release resources.

        Raises:
            NotImplementedError: Always raised by this abstract implementation.
            RuntimeError: If cleanup fails.
        """
        raise NotImplementedError(
            "EmbeddingClient.close() not implemented. "
            "This client requires a real embedding service implementation."
        )

    async def generate_embeddings(
        self,
        texts: List[str],
        model_name: str = "default",
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> List[dict[str, Any]]:
        """Generate embeddings for the provided texts.

        Args:
            texts: List of text strings to embed.
            model_name: Name of the embedding model to use.
            config: Optional configuration for the embedding operation.
            options: Optional additional parameters.

        Returns:
            List of embedding results, one per input text.

        Raises:
            NotImplementedError: Always raised by this abstract implementation.
            ValueError: If input parameters are invalid.
            RuntimeError: If embedding service fails.
        """
        raise NotImplementedError(
            "EmbeddingClient.generate_embeddings() not implemented. "
            "This client requires a real embedding service implementation. "
            "Please implement or configure a proper embedding service."
        )

    def summary(self) -> dict[str, Any]:
        return {
            "endpoint": self.service_endpoint,
            "status": "not_implemented",
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "average_processing_time": self.stats.average_processing_time,
        }


__all__ = ["EmbeddingClient", "EmbeddingClientStats"]
