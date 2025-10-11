"""VLLM client for MinerU service."""

import asyncio
from typing import Any, Dict, List, Optional


class VLLMServerError(Exception):
    """Error raised when VLLM server encounters an issue."""
    pass


class VLLMTimeoutError(Exception):
    """Error raised when VLLM request times out."""
    pass


from Medical_KG_rev.services.mineru.circuit_breaker import CircuitBreaker, CircuitState


class VLLMClient:
    """VLLM client for text processing."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        timeout: float = 30.0
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            name="vllm_client"
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        """Generate text using VLLM."""
        if self.circuit_breaker.state == CircuitState.OPEN:
            raise Exception("Circuit breaker is OPEN for VLLM client")

        # Simulate VLLM generation
        await asyncio.sleep(0.5)  # Simulate generation time

        # Return mock generated text
        return f"Generated response for: {prompt[:50]}..."

    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> List[str]:
        """Generate text for multiple prompts."""
        if self.circuit_breaker.state == CircuitState.OPEN:
            raise Exception("Circuit breaker is OPEN for VLLM client")

        # Simulate batch generation
        await asyncio.sleep(1.0)  # Simulate batch processing

        # Return mock generated texts
        return [f"Generated response {i+1}: {prompt[:30]}..." for i, prompt in enumerate(prompts)]

    async def health_check(self) -> Dict[str, Any]:
        """Check VLLM service health."""
        await asyncio.sleep(0.1)  # Simulate health check

        return {
            "status": "healthy",
            "model_loaded": True,
            "gpu_available": True,
            "memory_usage": 0.6
        }

    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self.circuit_breaker.state != CircuitState.OPEN
