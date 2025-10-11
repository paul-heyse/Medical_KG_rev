"""gRPC client for reranking service.

Handles communication with reranking gRPC service for reranking operations.
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

# Import generated gRPC stubs (will be generated from proto)
try:
    from ...proto.reranking_service_pb2 import (
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
        RerankBatchRequest,
        RerankBatchResponse,
        RerankingConfig,
        RerankingModel,
        RerankingResult,
        StatsRequest,
        StatsResponse,
    )
    from ...proto.reranking_service_pb2_grpc import RerankingServiceStub
except ImportError:
    # Fallback for development - will be replaced by generated stubs
    RerankBatchRequest = None
    RerankBatchResponse = None
    ListModelsRequest = None
    ListModelsResponse = None
    GetModelInfoRequest = None
    GetModelInfoResponse = None
    RerankingConfig = None
    ProcessingOptions = None
    RerankingResult = None
    ProcessingMetadata = None
    ProcessingStatus = None
    RerankingModel = None
    ModelCapabilities = None
    ModelPerformance = None
    ModelStatus = None
    HealthRequest = None
    HealthResponse = None
    StatsRequest = None
    StatsResponse = None
    RerankingServiceStub = None

logger = logging.getLogger(__name__)


class RerankingClient:
    """gRPC client for reranking service.

    Handles:
    - gRPC communication with reranking service
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
        """Initialize the reranking client.

        Args:
            service_endpoint: gRPC endpoint for the reranking service
            service_registry: Optional service registry for discovery
            circuit_breaker_config: Circuit breaker configuration

        """
        self.service_endpoint = service_endpoint
        self.service_registry = service_registry
        self.channel: aio.Channel | None = None
        self.stub: RerankingServiceStub | None = None

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
            if RerankingServiceStub:
                self.stub = RerankingServiceStub(self.channel)
            else:
                logger.warning("gRPC stubs not available - using mock implementation")

            # Test connection
            await self._test_connection()
            logger.info(f"Reranking client initialized for {self.service_endpoint}")

        except Exception as e:
            logger.error(f"Failed to initialize reranking client: {e}")
            raise ServiceError(f"Client initialization failed: {e}") from e

    async def close(self) -> None:
        """Close the gRPC client."""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None

    async def _test_connection(self) -> None:
        """Test connection to reranking service."""
        try:
            if not self.stub:
                raise ServiceError("Service stub not available")

            # Test with health check
            request = HealthRequest(service_name="reranking_service")
            response = await self.stub.GetHealth(request)

            if response.status != "healthy":
                raise ServiceError(f"Service not healthy: {response.message}")

        except Exception as e:
            logger.error(f"Reranking service connection test failed: {e}")
            raise ServiceError(f"Connection test failed: {e}") from e

    async def rerank_batch(
        self,
        query: str,
        documents: list[str],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank a batch of documents for a query.

        Args:
            query: Query text
            documents: List of document texts to rerank
            model_name: Name of the reranking model
            config: Reranking configuration options
            options: Processing options

        Returns:
            List of reranking results

        """
        start_time = time.time()

        try:
            if not self.stub:
                raise ServiceError("Service stub not available")

            # Prepare request
            request = self._create_rerank_request(query, documents, model_name, config, options)

            # Execute with circuit breaker
            response = await self.circuit_breaker.call(self._execute_rerank_request, request)

            # Process response
            result = self._process_rerank_response(response, start_time)

            # Update statistics
            self._update_stats(True, start_time)

            return result

        except Exception as e:
            logger.error(f"Error reranking batch: {e}")
            self._update_stats(False, start_time)
            raise self._handle_rerank_error(e) from e

    async def rerank_multiple_batches(
        self,
        queries: list[str],
        document_batches: list[list[str]],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Rerank multiple batches of documents for multiple queries.

        Args:
            queries: List of query texts
            document_batches: List of document batches to rerank
            model_name: Name of the reranking model
            config: Reranking configuration options
            options: Processing options

        Returns:
            List of reranking result batches

        """
        results = []

        # Process in batches to manage memory
        batch_size = options.get("batch_size", 5) if options else 5

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]
            batch_docs = document_batches[i : i + batch_size]

            # Process batch concurrently
            batch_tasks = [
                self.rerank_batch(query, docs, model_name, config, options)
                for query, docs in zip(batch_queries, batch_docs, strict=False)
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch reranking failed for batch {i + j}: {result}")
                    # Create error result
                    error_result = [{"error": str(result)}]
                    results.append(error_result)
                else:
                    results.append(result)

        return results

    def _create_rerank_request(
        self,
        query: str,
        documents: list[str],
        model_name: str,
        config: dict[str, Any] | None,
        options: dict[str, Any] | None,
    ) -> RerankBatchRequest:
        """Create RerankBatchRequest from parameters."""
        if not RerankBatchRequest:
            raise ServiceError("gRPC stubs not available")

        # Create RerankingConfig
        reranking_config = RerankingConfig()
        if config:
            reranking_config.max_sequence_length = config.get("max_sequence_length", 512)
            reranking_config.batch_size = config.get("batch_size", 16)
            reranking_config.score_threshold = config.get("score_threshold", 0.0)
            reranking_config.top_k = config.get("top_k", 100)
            reranking_config.enable_gpu = config.get("enable_gpu", True)
            reranking_config.temperature = config.get("temperature", 0.1)

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

        return RerankBatchRequest(
            query=query,
            documents=documents,
            model_name=model_name,
            config=reranking_config,
            options=processing_options,
        )

    async def _execute_rerank_request(self, request: RerankBatchRequest) -> RerankBatchResponse:
        """Execute the RerankBatch gRPC request."""
        if not self.stub:
            raise ServiceError("Service stub not available")

        response = await self.stub.RerankBatch(request)
        return response

    def _process_rerank_response(
        self,
        response: RerankBatchResponse,
        start_time: float,
    ) -> list[dict[str, Any]]:
        """Process the gRPC response."""
        try:
            if response.status != ProcessingStatus.PROCESSING_STATUS_SUCCESS:
                error_msg = response.error_message or "Unknown processing error"
                raise ServiceError(f"Reranking failed: {error_msg}")

            # Convert results to list of dictionaries
            results = []
            for result in response.results:
                result_dict = {
                    "document": result.document,
                    "score": result.score,
                    "rank": result.rank,
                    "model_name": result.model_name,
                    "model_version": result.model_version,
                    "confidence_score": result.confidence_score,
                    "reranked_at": result.reranked_at.ToDatetime() if result.reranked_at else None,
                }
                results.append(result_dict)

            return results

        except Exception as e:
            logger.error(f"Error processing rerank response: {e}")
            raise ServiceError(f"Response processing error: {e}") from e

    def _handle_rerank_error(self, error: Exception) -> Exception:
        """Handle reranking errors and convert to appropriate exception types."""
        if isinstance(error, grpc.RpcError):
            if error.code() == grpc.StatusCode.UNAVAILABLE:
                return ServiceUnavailableError(f"Reranking service unavailable: {error.details()}")
            elif error.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                return ServiceTimeoutError(f"Reranking service timeout: {error.details()}")
            else:
                return ServiceError(f"gRPC error: {error.details()}")
        else:
            return ServiceError(f"Reranking error: {error!s}")

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
        """List available reranking models.

        Args:
            include_model_info: Whether to include detailed model information

        Returns:
            List of reranking model information

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
            logger.error(f"Error listing reranking models: {e}")
            self._update_stats(False, start_time)
            raise self._handle_rerank_error(e) from e

    async def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get information about a specific reranking model.

        Args:
            model_name: Name of the reranking model

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
                    "supported_score_ranges": list(response.capabilities.supported_score_ranges),
                    "supported_ranking_strategies": list(
                        response.capabilities.supported_ranking_strategies
                    ),
                },
                "performance": {
                    "average_processing_time_ms": response.performance.average_processing_time_ms,
                    "throughput_pairs_per_second": response.performance.throughput_pairs_per_second,
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
            raise self._handle_rerank_error(e) from e

    async def health_check(self) -> dict[str, Any]:
        """Check reranking service health.

        Returns:
            Health status information

        """
        try:
            if not self.stub:
                return {"status": "unhealthy", "error": "Service stub not available"}

            request = HealthRequest(service_name="reranking_service")
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
            logger.error(f"Reranking service health check failed: {e}")
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
        """Get service statistics from reranking service."""
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


def create_reranking_client(service_endpoint: str) -> RerankingClient:
    """Create reranking client instance.

    Args:
        service_endpoint: gRPC endpoint for the reranking service

    Returns:
        RerankingClient instance

    """
    return RerankingClient(service_endpoint)
