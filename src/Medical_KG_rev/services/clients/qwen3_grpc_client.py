"""gRPC client wrapper for the Qwen3 embedding service.

Key Responsibilities:
    - Establish a synchronous gRPC channel instrumented with metrics.
    - Provide convenience helpers for embedding, health checking, and metadata.
    - Translate gRPC failures into domain-specific errors.

Collaborators:
    - Downstream: Qwen3 embedding gRPC service.
    - Upstream: Retrieval services requesting embeddings.

Side Effects:
    - Performs network I/O via gRPC.
    - Emits structured logs for observability.

Thread Safety:
    - Not thread-safe: Instances should be dedicated to a single thread/event loop.

Performance Characteristics:
    - Latency dominated by remote gRPC calls; local overhead is minimal.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import logging
import time
from typing import Any

import grpc

from Medical_KG_rev.observability.grpc_interceptor import create_metrics_channel_sync
from Medical_KG_rev.proto.gen import embedding_pb2, embedding_pb2_grpc
from Medical_KG_rev.services.clients.errors import ServiceUnavailableError

logger = logging.getLogger(__name__)


class Qwen3ServiceUnavailableError(ServiceUnavailableError):
    """Raised when Qwen3 gRPC service is unavailable."""

    def __init__(self, message: str, endpoint: str | None = None) -> None:
        super().__init__(message)
        self.endpoint = endpoint


class Qwen3GRPCClient:
    """Synchronous gRPC client for Qwen3 embeddings.

    Attributes:
        endpoint: Target gRPC endpoint (``host:port``).
        timeout: Default unary request timeout in seconds.
        max_retries: Maximum number of retry attempts (not yet implemented).
        retry_delay: Delay between retries in seconds (not yet implemented).
        channel: Underlying gRPC channel instrumented with metrics.
        stub: Generated gRPC stub for the embedding service.
    """

    def __init__(
        self,
        endpoint: str = "localhost:50051",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize the Qwen3 gRPC client.

        Args:
            endpoint: Target gRPC endpoint (``host:port``).
            timeout: Unary RPC timeout in seconds.
            max_retries: Maximum number of retry attempts (reserved for future use).
            retry_delay: Delay between retries in seconds (reserved for future use).
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Create gRPC channel with metrics interceptor. The generated gRPC stubs
        # we use here are synchronous, so ensure we create a matching channel.
        self.channel = create_metrics_channel_sync(endpoint)
        self.stub = embedding_pb2_grpc.EmbeddingServiceStub(self.channel)

        logger.info(
            "Qwen3 gRPC client initialized",
            extra={
                "endpoint": endpoint,
                "timeout": timeout,
                "max_retries": max_retries,
            },
        )

    def embed_texts(
        self,
        texts: list[str],
        model_name: str = "qwen3",
        namespace: str = "default",
        normalize: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for input texts via gRPC.

        Args:
            texts: Input strings to embed.
            model_name: Remote model identifier to use for embedding.
            namespace: Embedding namespace used by the remote service.
            normalize: Whether to normalize vectors server-side.

        Returns:
            A list of embedding vectors (each vector is a list of floats).

        Raises:
            Qwen3ServiceUnavailableError: Raised when the remote service cannot be reached.
            grpc.RpcError: Propagated for unexpected gRPC failures.
        """
        if not texts:
            return []

        start_time = time.perf_counter()

        try:
            # Create gRPC request
            request = embedding_pb2.EmbedRequest(
                inputs=texts,
                namespace=namespace,
                normalize=normalize,
            )

            # Make gRPC call
            response = self.stub.Embed(request, timeout=self.timeout)

            # Extract embeddings
            embeddings = []
            for embedding_vector in response.embeddings:
                embeddings.append(list(embedding_vector.values))

            processing_time = time.perf_counter() - start_time

            logger.info(
                "Qwen3 embeddings generated via gRPC",
                extra={
                    "text_count": len(texts),
                    "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                    "processing_time_seconds": processing_time,
                    "model_name": model_name,
                    "namespace": namespace,
                },
            )

            return embeddings

        except grpc.RpcError as e:
            processing_time = time.perf_counter() - start_time

            logger.error(
                "Qwen3 gRPC embedding request failed",
                extra={
                    "text_count": len(texts),
                    "error_code": str(e.code()),
                    "error_details": e.details(),
                    "processing_time_seconds": processing_time,
                    "endpoint": self.endpoint,
                },
            )

            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise Qwen3ServiceUnavailableError(
                    f"Qwen3 service unavailable: {e.details()}",
                    endpoint=self.endpoint,
                ) from e

            raise

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            logger.error(
                "Unexpected error in Qwen3 gRPC client",
                extra={
                    "text_count": len(texts),
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                    "endpoint": self.endpoint,
                },
            )

            raise Qwen3ServiceUnavailableError(
                f"Unexpected error: {e}",
                endpoint=self.endpoint,
            ) from e

    def health_check(self) -> dict[str, Any]:
        """Check the health of the Qwen3 gRPC service.

        Returns:
            A dictionary describing service health and latency.

        Raises:
            Qwen3ServiceUnavailableError: If the health check RPC fails.
        """
        start_time = time.perf_counter()

        try:
            # Create a simple validation request
            request = embedding_pb2.ValidateTextsRequest(
                tenant_id="health_check",
                namespace="default",
                texts=["health check"],
            )

            # Make gRPC call
            response = self.stub.ValidateTexts(request, timeout=5.0)

            processing_time = time.perf_counter() - start_time

            health_status = {
                "status": "healthy",
                "endpoint": self.endpoint,
                "response_time_seconds": processing_time,
                "service_available": True,
            }

            logger.info(
                "Qwen3 gRPC service health check passed",
                extra=health_status,
            )

            return health_status

        except grpc.RpcError as e:
            processing_time = time.perf_counter() - start_time

            logger.warning(
                "Qwen3 gRPC service health check failed",
                extra={
                    "endpoint": self.endpoint,
                    "response_time_seconds": processing_time,
                    "error_code": str(e.code()),
                    "error_details": e.details(),
                },
            )
            raise Qwen3ServiceUnavailableError(
                f"Health check failed: {e.details()}",
                endpoint=self.endpoint,
            ) from e

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            logger.error(
                "Qwen3 gRPC service health check failed with unexpected error",
                extra={
                    "endpoint": self.endpoint,
                    "response_time_seconds": processing_time,
                    "error": str(e),
                },
            )

            raise Qwen3ServiceUnavailableError(
                f"Health check failed: {e}",
                endpoint=self.endpoint,
            ) from e

    def get_service_info(self) -> dict[str, Any]:
        """Retrieve metadata about the Qwen3 gRPC service.

        Returns:
            Service information such as available namespaces and timeouts.

        Raises:
            Qwen3ServiceUnavailableError: If the metadata RPC fails.
        """
        try:
            # List available namespaces
            request = embedding_pb2.ListNamespacesRequest(tenant_id="info")

            response = self.stub.ListNamespaces(request, timeout=5.0)

            namespaces = []
            for namespace_info in response.namespaces:
                namespaces.append({
                    "id": namespace_info.id,
                    "provider": namespace_info.provider,
                    "kind": namespace_info.kind,
                    "dimension": namespace_info.dimension,
                    "max_tokens": namespace_info.max_tokens,
                    "enabled": namespace_info.enabled,
                })

            service_info = {
                "endpoint": self.endpoint,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "available_namespaces": namespaces,
            }

            logger.info(
                "Qwen3 gRPC service info retrieved",
                extra=service_info,
            )

            return service_info

        except grpc.RpcError as e:
            logger.warning(
                "Failed to get Qwen3 gRPC service info",
                extra={
                    "endpoint": self.endpoint,
                    "error_code": str(e.code()),
                    "error_details": e.details(),
                },
            )
            raise Qwen3ServiceUnavailableError(
                f"Failed to get service info: {e.details()}",
                endpoint=self.endpoint,
            ) from e

        except Exception as e:
            logger.error(
                "Unexpected error getting Qwen3 gRPC service info",
                extra={
                    "endpoint": self.endpoint,
                    "error": str(e),
                },
            )
            raise Qwen3ServiceUnavailableError(
                f"Unexpected error getting service info: {e}",
                endpoint=self.endpoint,
            ) from e

    def close(self) -> None:
        """Close the gRPC channel."""
        if self.channel:
            try:
                # For async channels, we need to await the close
                import asyncio
                if asyncio.iscoroutine(self.channel.close()):
                    # This is an async channel, but we're in a sync context
                    # Just log a warning and don't close it
                    logger.warning("Cannot close async gRPC channel in sync context")
                else:
                    self.channel.close()
                    logger.info("Qwen3 gRPC client channel closed")
            except Exception as e:
                logger.warning(f"Error closing gRPC channel: {e}")

    def __enter__(self) -> Qwen3GRPCClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
