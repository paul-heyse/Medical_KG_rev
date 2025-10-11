"""Async reranking client abstractions and statistics tracking.

Key Responsibilities:
    - Define the abstract contract for asynchronous reranking clients
    - Provide simple statistics helpers that concrete clients can reuse

Collaborators:
    - Upstream: Retrieval orchestrators invoke reranking through this interface
    - Downstream: Concrete gRPC/HTTP reranking services satisfy requests

Side Effects:
    - Concrete subclasses may issue network I/O and emit telemetry

Thread Safety:
    - Thread-safe when implementations avoid shared mutable state; the base class
      itself only stores per-instance configuration and counters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RerankingClientStats:
    """Statistics captured while executing reranking requests.

    Attributes:
        total_requests: Number of reranking requests attempted.
        successful_requests: Number of requests that completed successfully.
        failed_requests: Number of requests that raised an error.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


class RerankingClient:
    """Abstract base class for reranking service clients.

    This class defines the interface that concrete reranking clients must implement.
    It serves as the contract for communicating with reranking services.

    Attributes:
        service_endpoint: URL or endpoint of the reranking service.
        stats: Statistics tracking for client operations.

    Invariants:
        - service_endpoint must be a valid URL or service identifier
        - stats object must be initialized before use
        - Client must be initialized before reranking documents

    Thread Safety:
        - Thread-safe: All async operations can be called concurrently

    Lifecycle:
        - Must call initialize() before use
        - Must call close() when finished to clean up resources
        - Can be reused after initialization

    Example:
        >>> class MyRerankingClient(RerankingClient):
        ...     async def initialize(self):
        ...         # Setup connection
        ...         pass
        ...
        ...     async def rerank_documents(self, query, documents, **kwargs):
        ...         # Implementation here
        ...         return []
    """

    def __init__(self, service_endpoint: str) -> None:
        """Initialize the reranking client.

        Args:
            service_endpoint: URL or identifier for the reranking service.
        """
        self.service_endpoint = service_endpoint
        self.stats = RerankingClientStats()

    async def initialize(self) -> None:
        """Initialize the client connection and resources.

        Raises:
            NotImplementedError: Always raised by this abstract implementation.
            RuntimeError: If initialization fails.
        """
        raise NotImplementedError(
            "RerankingClient.initialize() not implemented. "
            "This client requires a real reranking service implementation. "
            "Please implement or configure a proper reranking service."
        )

    async def close(self) -> None:
        """Close the client and release resources.

        Raises:
            NotImplementedError: Always raised by this abstract implementation.
            RuntimeError: If cleanup fails.
        """
        raise NotImplementedError(
            "RerankingClient.close() not implemented. "
            "This client requires a real reranking service implementation."
        )

    async def rerank_documents(
        self,
        query: str,
        documents: Iterable[str],
        *,
        model: str = "default",
        top_k: int = 10,
    ) -> List[dict[str, Any]]:
        """Rerank documents based on relevance to a query.

        Args:
            query: Search query to rank documents against.
            documents: Iterable of document texts to rerank.
            model: Name of the reranking model to use.
            top_k: Maximum number of results to return.

        Returns:
            List of reranked documents with scores and rankings.

        Raises:
            NotImplementedError: Always raised by this abstract implementation.
            ValueError: If input parameters are invalid.
            RuntimeError: If reranking service fails.
        """
        raise NotImplementedError(
            "RerankingClient.rerank_documents() not implemented. "
            "This client requires a real reranking service implementation. "
            "Please implement or configure a proper reranking service."
        )

    async def rerank_batch(
        self,
        queries: Iterable[str],
        documents: Iterable[str],
        *,
        model: str = "default",
        top_k: int = 10,
    ) -> List[List[dict[str, Any]]]:
        """Rerank documents for multiple queries in batch.

        Args:
            queries: Iterable of search queries.
            documents: Iterable of document texts to rerank.
            model: Name of the reranking model to use.
            top_k: Maximum number of results to return per query.

        Returns:
            List of reranked results, one per query.

        Raises:
            NotImplementedError: Always raised by this abstract implementation.
            ValueError: If input parameters are invalid.
            RuntimeError: If reranking service fails.
        """
        raise NotImplementedError(
            "RerankingClient.rerank_batch() not implemented. "
            "This client requires a real reranking service implementation. "
            "Please implement or configure a proper reranking service."
        )

    async def list_models(self) -> List[str]:
        """List available reranking models.

        Returns:
            List of available model names.

        Raises:
            NotImplementedError: Always raised by this abstract implementation.
            RuntimeError: If service fails to list models.
        """
        raise NotImplementedError(
            "RerankingClient.list_models() not implemented. "
            "This client requires a real reranking service implementation."
        )

    def summary(self) -> dict[str, Any]:
        """Return configuration and statistics for observability."""
        return {
            "endpoint": self.service_endpoint,
            "status": "not_implemented",
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
        }
