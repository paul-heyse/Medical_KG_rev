"""gRPC client for embedding service."""

from typing import Optional
import asyncio

from grpc import aio

from Medical_KG_rev.services.clients.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
)


class EmbeddingServiceClient:
    """gRPC client for embedding service."""

    def __init__(self, host: str = "localhost", port: int = 50051, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._channel: Optional[aio.Channel] = None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5, recovery_timeout=60.0, name="embedding_service"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Connect to the embedding service."""
        if self._channel is None:
            self._channel = aio.insecure_channel(f"{self.host}:{self.port}")

    async def close(self) -> None:
        """Close the connection."""
        if self._channel:
            await self._channel.close()
            self._channel = None

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        if self._circuit_breaker.state == CircuitBreakerState.OPEN:
            raise Exception("Circuit breaker is OPEN for embedding service")

        # Simulate embedding service call
        # In a real implementation, this would make actual gRPC calls
        await asyncio.sleep(0.1)  # Simulate network delay

        # Return dummy embeddings
        return [[0.1] * 384 for _ in texts]  # 384-dimensional embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]

    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self._circuit_breaker.state != CircuitBreakerState.OPEN
