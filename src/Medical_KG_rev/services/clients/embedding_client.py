"""gRPC client for embedding service.

Handles communication with embedding gRPC service for embedding generation.
"""

import asyncio
import logging
import time
from typing import Any

import grpc
from grpc import aio

from ..clients.circuit_breaker import CircuitBreaker, CircuitBreakerState
from ..clients.errors import ServiceError, ServiceTimeoutError, ServiceUnavailableError
from ..registry import ServiceRegistry


class EmbeddingClientManager:
    """Manager for embedding service clients."""

    def __init__(self):
        self._clients = {}

    async def get_client(self, service_name: str = "embedding"):
        """Get embedding client for service."""
        if service_name not in self._clients:
            self._clients[service_name] = EmbeddingClient(service_name)
        return self._clients[service_name]

    async def close(self):
        """Close all clients."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

# Import generated gRPC stubs (will be generated from proto)
try:
    from ...proto.embedding_service_pb2 import (
        EmbeddingConfig,
        EmbeddingModel,
        EmbeddingResult,
        GenerateEmbeddingsBatchRequest,
        GenerateEmbeddingsBatchResponse,
        GenerateEmbeddingsRequest,
        GenerateEmbeddingsResponse,
        GetModelInfoRequest,
        GetModelInfoResponse,
        HealthRequest,
        HealthResponse,
        ListModelsRequest,
        ListModelsResponse,
        ModelCapabilities,
        ModelPerformance,
        ModelStatus,
        ProcessingMetadata,
        ProcessingOptions,
        ProcessingStatus,
        StatsRequest,
        StatsResponse,
    )
    from ...proto.embedding_service_pb2_grpc import EmbeddingServiceStub
except ImportError:
    # Fallback for development - will be replaced by generated stubs
    GenerateEmbeddingsRequest = None
    GenerateEmbeddingsResponse = None
    GenerateEmbeddingsBatchRequest = None
    GenerateEmbeddingsBatchResponse = None
    ListModelsRequest = None
    ListModelsResponse = None
    GetModelInfoRequest = None
    GetModelInfoResponse = None
    EmbeddingConfig = None
    ProcessingOptions = None
    EmbeddingResult = None
    ProcessingMetadata = None
    ProcessingStatus = None
    EmbeddingModel = None
    ModelCapabilities = None
    ModelPerformance = None
    ModelStatus = None
    HealthRequest = None
    HealthResponse = None
    StatsRequest = None
    StatsResponse = None
    EmbeddingServiceStub = None

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """gRPC client for embedding service.

    Handles:
    - gRPC communication with embedding service
    - Circuit breaker patterns for service resilience
    - Error handling and retry logic
    - Performance monitoring and metrics
    """

    def __init__(
        self,
        service_endpoint: str,
        service_registry: ServiceRegistry | None = None,
        circuit_breaker_config: dict[str, Any] | None = None,
    ):
        """Initialize the embedding client.

        Args:
            service_endpoint: gRPC endpoint for the embedding service
            service_registry: Optional service registry for discovery
            circuit_breaker_config: Circuit breaker configuration

        """
        self.service_endpoint = service_endpoint
        self.service_registry = service_registry
        self.channel: aio.Channel | None = None
        self.stub: EmbeddingServiceStub | None = None

        # Circuit breaker configuration
        circuit_config = circuit_breaker_config or {}
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_config.get("failure_threshold", 5),
            recovery_timeout=circuit_config.get("recovery_timeout", 60),
            expected_exception=ServiceError,
        )

        # Performance tracking
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "circuit_breaker_state": CircuitBreakerState.CLOSED,
        }

    async def initialize(self) -> None:
        """Initialize the gRPC client."""
        try:
            # Create gRPC channel
            self.channel = aio.insecure_channel(self.service_endpoint)

            # Create service stub
            if EmbeddingServiceStub:
                self.stub = EmbeddingServiceStub(self.channel)
            else:
                logger.warning("gRPC stubs not available - using mock implementation")

            # Test connection
            await self._test_connection()
            logger.info(f"Embedding client initialized for {self.service_endpoint}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding client: {e}")
            raise ServiceError(f"Client initialization failed: {e}") from e

    async def close(self) -> None:
        """Close the gRPC client."""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None

    async def _test_connection(self) -> None:
        """Test connection to embedding service."""
        try:
            if not self.stub:
                raise ServiceError("Service stub not available")

            # Test with health check
            request = HealthRequest(service_name="embedding_service")
            response = await self.stub.GetHealth(request)

            if response.status != "healthy":
                raise ServiceError(f"Service not healthy: {response.message}")

        except Exception as e:
            logger.error(f"Embedding service connection test failed: {e}")
            raise ServiceError(f"Connection test failed: {e}") from e

    async def generate_embeddings(
        self,
        texts: list[str],
        model_name: str = "qwen3-8b",
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate embeddings for text.

        Args:
            texts: List of texts to embed
            model_name: Name of the embedding model
            config: Embedding configuration options
            options: Processing options

        Returns:
            List of embedding results

        """
        start_time = time.time()

        try:
            if not self.stub:
                raise ServiceError("Service stub not available")

            # Prepare request
            request = self._create_embedding_request(texts, model_name, config, options)

            # Execute with circuit breaker
            response = await self.circuit_breaker.call(self._execute_embedding_request, request)

            # Process response
            result = self._process_embedding_response(response, start_time)

            # Update statistics
            self._update_stats(True, start_time)

            return result

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            self._update_stats(False, start_time)
            raise self._handle_embedding_error(e) from e

    async def generate_embeddings_batch(
        self,
        text_batches: list[list[str]],
        model_name: str = "qwen3-8b",
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Generate embeddings for multiple batches of text.

        Args:
            text_batches: List of text batches to embed
            model_name: Name of the embedding model
            config: Embedding configuration options
            options: Processing options

        Returns:
            List of embedding result batches

        """
        results = []

        # Process in batches to manage memory
        batch_size = options.get("batch_size", 5) if options else 5

        for i in range(0, len(text_batches), batch_size):
            batch = text_batches[i : i + batch_size]

            # Process batch concurrently
            batch_tasks = [
                self.generate_embeddings(texts, model_name, config, options) for texts in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch embedding failed for batch {i + j}: {result}")
                    # Create error result
                    error_result = [{"error": str(result)}]
                    results.append(error_result)
                else:
                    results.append(result)

        return results

    def _create_embedding_request(
        self,
        texts: list[str],
        model_name: str,
        config: dict[str, Any] | None,
        options: dict[str, Any] | None,
    ) -> GenerateEmbeddingsRequest:
        """Create GenerateEmbeddingsRequest from parameters."""
        if not GenerateEmbeddingsRequest:
            raise ServiceError("gRPC stubs not available")

        # Create EmbeddingConfig
        embedding_config = EmbeddingConfig()
        if config:
            embedding_config.max_sequence_length = config.get("max_sequence_length", 512)
            embedding_config.normalize_embeddings = config.get("normalize_embeddings", True)
            embedding_config.pooling_strategy = config.get("pooling_strategy", "mean")
            embedding_config.temperature = config.get("temperature", 0.1)
            embedding_config.enable_gpu = config.get("enable_gpu", True)
            embedding_config.batch_size = config.get("batch_size", 8)

        # Create ProcessingOptions
        processing_options = ProcessingOptions()
        if options:
            processing_options.enable_contextualization = options.get(
                "enable_contextualization", True
            )
            processing_options.enable_medical_normalization = options.get(
                "enable_medical_normalization", True
            )
            processing_options.enable_terminology_support = options.get(
                "enable_terminology_support", True
            )
            processing_options.min_confidence_threshold = options.get(
                "min_confidence_threshold", 0.7
            )
            processing_options.timeout_seconds = options.get("timeout_seconds", 300)

        return GenerateEmbeddingsRequest(
            texts=texts,
            model_name=model_name,
            config=embedding_config,
            options=processing_options,
        )

    async def _execute_embedding_request(
        self, request: GenerateEmbeddingsRequest
    ) -> GenerateEmbeddingsResponse:
        """Execute the GenerateEmbeddings gRPC request."""
        if not self.stub:
            raise ServiceError("Service stub not available")

        response = await self.stub.GenerateEmbeddings(request)
        return response

    def _process_embedding_response(
        self,
        response: GenerateEmbeddingsResponse,
        start_time: float,
    ) -> list[dict[str, Any]]:
        """Process the gRPC response."""
        try:
            if response.status != ProcessingStatus.PROCESSING_STATUS_SUCCESS:
                error_msg = response.error_message or "Unknown processing error"
                raise ServiceError(f"Embedding generation failed: {error_msg}")

            # Convert results to list of dictionaries
            results = []
            for result in response.results:
                result_dict = {
                    "text": result.text,
                    "embedding": list(result.embedding),
                    "embedding_dimension": result.embedding_dimension,
                    "model_name": result.model_name,
                    "model_version": result.model_version,
                    "confidence_score": result.confidence_score,
                    "generated_at": (
                        result.generated_at.ToDatetime() if result.generated_at else None
                    ),
                }
                results.append(result_dict)

            return results

        except Exception as e:
            logger.error(f"Error processing embedding response: {e}")
            raise ServiceError(f"Response processing error: {e}") from e

    def _handle_embedding_error(self, error: Exception) -> Exception:
        """Handle embedding errors and convert to appropriate exception types."""
        if isinstance(error, grpc.RpcError):
            if error.code() == grpc.StatusCode.UNAVAILABLE:
                return ServiceUnavailableError(f"Embedding service unavailable: {error.details()}")
            elif error.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                return ServiceTimeoutError(f"Embedding service timeout: {error.details()}")
            else:
                return ServiceError(f"gRPC error: {error.details()}")
        else:
            return ServiceError(f"Embedding error: {error!s}")

    def _update_stats(self, success: bool, start_time: float) -> None:
        """Update processing statistics."""
        self._stats["total_requests"] += 1
        processing_time = time.time() - start_time
        self._stats["total_processing_time"] += processing_time

        if success:
            self._stats["successful_requests"] += 1
        else:
            self._stats["failed_requests"] += 1

        # Calculate average processing time
        if self._stats["total_requests"] > 0:
            self._stats["average_processing_time"] = (
                self._stats["total_processing_time"] / self._stats["total_requests"]
            )

        # Update circuit breaker state
        self._stats["circuit_breaker_state"] = self.circuit_breaker.state

    async def list_models(self, include_model_info: bool = True) -> list[dict[str, Any]]:
        """List available embedding models.

        Args:
            include_model_info: Whether to include detailed model information

        Returns:
            List of embedding model information

        """
        start_time = time.time()

        try:
            if not self.stub:
                raise ServiceError("Service stub not available")

            request = ListModelsRequest(include_model_info=include_model_info)
            response = await self.stub.ListModels(request)

            # Convert models to list of dictionaries
            models = []
            for model in response.models:
                model_dict = {
                    "name": model.name,
                    "version": model.version,
                    "description": model.description,
                    "embedding_dimension": model.embedding_dimension,
                    "max_sequence_length": model.max_sequence_length,
                    "supported_languages": list(model.supported_languages),
                    "capabilities": list(model.capabilities),
                    "status": model.status.name if model.status else "UNKNOWN",
                    "last_updated": model.last_updated.ToDatetime() if model.last_updated else None,
                }
                models.append(model_dict)

            # Update statistics
            self._update_stats(True, start_time)

            return models

        except Exception as e:
            logger.error(f"Error listing embedding models: {e}")
            self._update_stats(False, start_time)
            raise self._handle_embedding_error(e) from e

    async def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get information about a specific embedding model.

        Args:
            model_name: Name of the embedding model

        Returns:
            Model information

        """
        start_time = time.time()

        try:
            if not self.stub:
                raise ServiceError("Service stub not available")

            request = GetModelInfoRequest(model_name=model_name)
            response = await self.stub.GetModelInfo(request)

            # Convert response to dictionary
            result = {
                "model": {
                    "name": response.model.name,
                    "version": response.model.version,
                    "description": response.model.description,
                    "embedding_dimension": response.model.embedding_dimension,
                    "max_sequence_length": response.model.max_sequence_length,
                    "supported_languages": list(response.model.supported_languages),
                    "capabilities": list(response.model.capabilities),
                    "status": response.model.status.name if response.model.status else "UNKNOWN",
                    "last_updated": (
                        response.model.last_updated.ToDatetime()
                        if response.model.last_updated
                        else None
                    ),
                },
                "capabilities": {
                    "supports_batch_processing": response.capabilities.supports_batch_processing,
                    "supports_gpu_acceleration": response.capabilities.supports_gpu_acceleration,
                    "supports_contextualization": response.capabilities.supports_contextualization,
                    "supports_medical_normalization": response.capabilities.supports_medical_normalization,
                    "supports_terminology_support": response.capabilities.supports_terminology_support,
                    "supported_pooling_strategies": list(
                        response.capabilities.supported_pooling_strategies
                    ),
                    "supported_normalization_methods": list(
                        response.capabilities.supported_normalization_methods
                    ),
                },
                "performance": {
                    "average_processing_time_ms": response.performance.average_processing_time_ms,
                    "throughput_texts_per_second": response.performance.throughput_texts_per_second,
                    "memory_usage_mb": response.performance.memory_usage_mb,
                    "gpu_memory_usage_mb": response.performance.gpu_memory_usage_mb,
                    "accuracy_score": response.performance.accuracy_score,
                },
            }

            # Update statistics
            self._update_stats(True, start_time)

            return result

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            self._update_stats(False, start_time)
            raise self._handle_embedding_error(e) from e

    async def health_check(self) -> dict[str, Any]:
        """Check embedding service health.

        Returns:
            Health status information

        """
        try:
            if not self.stub:
                return {"status": "unhealthy", "error": "Service stub not available"}

            request = HealthRequest(service_name="embedding_service")
            response = await self.stub.GetHealth(request)

            return {
                "status": response.status,
                "message": response.message,
                "timestamp": response.timestamp.ToDatetime() if response.timestamp else None,
                "service_info": {
                    "version": response.service_info.version,
                    "capabilities": list(response.service_info.capabilities),
                },
                "resource_usage": {
                    "cpu_usage_percent": response.resource_usage.cpu_usage_percent,
                    "memory_usage_mb": response.resource_usage.memory_usage_mb,
                    "gpu_usage_percent": response.resource_usage.gpu_usage_percent,
                    "gpu_memory_usage_mb": response.resource_usage.gpu_memory_usage_mb,
                    "active_models": response.resource_usage.active_models,
                },
                "circuit_breaker_state": self.circuit_breaker.state.value,
            }

        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state.value,
            }

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "client_stats": self._stats.copy(),
            "service_endpoint": self.service_endpoint,
            "circuit_breaker_state": self.circuit_breaker.state.value,
        }

    async def get_service_stats(self) -> dict[str, Any]:
        """Get service statistics from embedding service."""
        try:
            if not self.stub:
                return {"error": "Service stub not available"}

            request = StatsRequest()
            response = await self.stub.GetStats(request)

            return {
                "service_stats": [
                    {
                        "metric_name": metric.metric_name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp.ToDatetime() if metric.timestamp else None,
                        "labels": dict(metric.labels),
                    }
                    for metric in response.metrics
                ],
                "generated_at": (
                    response.generated_at.ToDatetime() if response.generated_at else None
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {"error": str(e)}


def create_embedding_client(service_endpoint: str) -> EmbeddingClient:
    """Create embedding client instance.

    Args:
        service_endpoint: gRPC endpoint for the embedding service

    Returns:
        EmbeddingClient instance

    """
    return EmbeddingClient(service_endpoint)
