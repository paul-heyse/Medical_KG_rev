"""gRPC client for GPU service."""

import asyncio
from typing import Any, Dict, List, Optional

import grpc
from grpc import aio

from Medical_KG_rev.services.clients.circuit_breaker import CircuitBreaker, CircuitBreakerState


class GPUServiceClient:
    """gRPC client for GPU service."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50052,
        timeout: float = 30.0
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._channel: Optional[aio.Channel] = None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            name="gpu_service"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Connect to the GPU service."""
        if self._channel is None:
            self._channel = aio.insecure_channel(f"{self.host}:{self.port}")

    async def close(self) -> None:
        """Close the connection."""
        if self._channel:
            await self._channel.close()
            self._channel = None

    async def process_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of data on GPU."""
        if self._circuit_breaker.state == CircuitBreakerState.OPEN:
            raise Exception("Circuit breaker is OPEN for GPU service")

        # Simulate GPU processing
        await asyncio.sleep(0.2)  # Simulate processing time

        # Return processed data
        return [{"processed": True, "result": f"processed_{i}"} for i in range(len(data))]

    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status information."""
        # Simulate GPU status check
        await asyncio.sleep(0.05)

        return {
            "available": True,
            "device_count": 1,
            "memory_used": 1024,
            "memory_total": 8192,
            "utilization": 45.0
        }

    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self._circuit_breaker.state != CircuitBreakerState.OPEN
