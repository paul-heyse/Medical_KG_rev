"""gRPC client for reranking service."""

import asyncio

from grpc import aio

from Medical_KG_rev.services.clients.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
)


class RerankingServiceClient:
    """gRPC client for reranking service."""

    def __init__(self, host: str = "localhost", port: int = 50053, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._channel: Optional[aio.Channel] = None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5, recovery_timeout=60.0, name="reranking_service"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Connect to the reranking service."""
        if self._channel is None:
            self._channel = aio.insecure_channel(f"{self.host}:{self.port}")

    async def close(self) -> None:
        """Close the connection."""
        if self._channel:
            await self._channel.close()
            self._channel = None

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Rerank documents based on query relevance."""
        if self._circuit_breaker.state == CircuitBreakerState.OPEN:
            raise Exception("Circuit breaker is OPEN for reranking service")

        # Simulate reranking processing
        await asyncio.sleep(0.1)  # Simulate processing time

        # Return dummy reranked results
        results = []
        for i, doc in enumerate(documents[:top_k]):
            score = 1.0 - (i * 0.1)  # Decreasing scores
            results.append((doc, score))

        return results

    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self._circuit_breaker.state != CircuitBreakerState.OPEN
