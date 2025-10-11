"""gRPC client for Docling VLM service.

Handles communication with Docling VLM gRPC service for PDF processing.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import grpc
from grpc import aio

from ..caching.vlm_cache import CacheLevel, VLMCache, get_vlm_cache
from ..clients.circuit_breaker import CircuitBreaker, CircuitBreakerState
from ..clients.errors import ServiceError, ServiceTimeoutError, ServiceUnavailableError
from ..parsing.exceptions import (
    DoclingProcessingError,
    DoclingVLMError,
)
from ..registry import ServiceRegistry

# OpenTelemetry tracing
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    trace = None
    Status = None
    StatusCode = None

# Import generated gRPC stubs (will be generated from proto)
try:
    from ...proto.docling_vlm_service_pb2 import (
        DoclingConfig,
        DocTagsResult,
        HealthRequest,
        HealthResponse,
        ProcessingMetadata,
        ProcessingOptions,
        ProcessingStatus,
        ProcessPDFBatchRequest,
        ProcessPDFBatchResponse,
        ProcessPDFRequest,
        ProcessPDFResponse,
        StatsRequest,
        StatsResponse,
    )
    from ...proto.docling_vlm_service_pb2_grpc import DoclingVLMServiceStub
except ImportError:
    # Fallback for development - will be replaced by generated stubs
    ProcessPDFRequest = None
    ProcessPDFResponse = None
    ProcessPDFBatchRequest = None
    ProcessPDFBatchResponse = None
    DoclingConfig = None
    ProcessingOptions = None
    DocTagsResult = None
    ProcessingMetadata = None
    ProcessingStatus = None
    HealthRequest = None
    HealthResponse = None
    StatsRequest = None
    StatsResponse = None
    DoclingVLMServiceStub = None

logger = logging.getLogger(__name__)


class DoclingVLMClient:
    """gRPC client for Docling VLM service.

    Handles:
    - gRPC communication with Docling VLM service
    - Circuit breaker patterns for service resilience
    - Error handling and retry logic
    - Performance monitoring and metrics
    """

    def __init__(
        self,
        service_endpoint: str,
        service_registry: ServiceRegistry | None = None,
        circuit_breaker_config: dict[str, Any] | None = None,
        cache: VLMCache | None = None,
    ):
        """Initialize the Docling VLM client.

        Args:
            service_endpoint: gRPC endpoint for the VLM service
            service_registry: Optional service registry for discovery
            circuit_breaker_config: Circuit breaker configuration
            cache: Optional VLM cache instance

        """
        self.service_endpoint = service_endpoint
        self.service_registry = service_registry
        self.cache = cache or get_vlm_cache()
        self.channel: aio.Channel | None = None
        self.stub: DoclingVLMServiceStub | None = None

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
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def initialize(self) -> None:
        """Initialize the gRPC client."""
        try:
            # Create gRPC channel
            self.channel = aio.insecure_channel(self.service_endpoint)

            # Create service stub
            if DoclingVLMServiceStub:
                self.stub = DoclingVLMServiceStub(self.channel)
            else:
                logger.warning("gRPC stubs not available - using mock implementation")

            # Test connection
            await self._test_connection()
            logger.info(f"Docling VLM client initialized for {self.service_endpoint}")

        except Exception as e:
            logger.error(f"Failed to initialize Docling VLM client: {e}")
            raise DoclingVLMError(f"Client initialization failed: {e}")

    async def close(self) -> None:
        """Close the gRPC client."""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None

    async def _test_connection(self) -> None:
        """Test connection to VLM service."""
        try:
            if not self.stub:
                raise DoclingVLMError("Service stub not available")

            # Test with health check
            request = HealthRequest(service_name="docling_vlm")
            response = await self.stub.GetHealth(request)

            if response.status != "healthy":
                raise DoclingVLMError(f"Service not healthy: {response.message}")

        except Exception as e:
            logger.error(f"VLM service connection test failed: {e}")
            raise DoclingVLMError(f"Connection test failed: {e}")

    async def process_pdf(
        self,
        pdf_path: str,
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> DocTagsResult:
        """Process a single PDF using Docling VLM.

        Args:
            pdf_path: Path to the PDF file
            config: Docling configuration options
            options: Processing options

        Returns:
            DocTagsResult with processing results

        """
        start_time = time.time()

        # Create tracing span for VLM processing
        tracer = trace.get_tracer(__name__) if trace else None
        span = None
        if tracer:
            span = tracer.start_span("docling_vlm.process_pdf")
            span.set_attribute("pdf.path", pdf_path)
            span.set_attribute("service.name", "docling-vlm")
            span.set_attribute("processing.method", "single_pdf")

        try:
            # Validate PDF path
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Read PDF content
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()

            # Update span with PDF size
            if span:
                span.set_attribute("pdf.size_bytes", len(pdf_content))

            # Check cache first
            cached_result = await self.cache.get(
                pdf_content=pdf_content,
                config=config or {},
                options=options or {},
                level=CacheLevel.DOCTAGS_RESULT
            )

            if cached_result is not None:
                # Update span with cache hit
                if span:
                    span.set_attribute("cache.hit", True)
                    span.set_attribute("processing.duration_seconds", time.time() - start_time)
                    span.set_attribute("processing.success", True)
                    span.set_status(Status(StatusCode.OK))

                # Update cache stats
                self._stats["cache_hits"] += 1

                logger.info("Cache hit for PDF processing", pdf_path=pdf_path)
                return cached_result

            # Cache miss - update stats
            self._stats["cache_misses"] += 1
            if span:
                span.set_attribute("cache.hit", False)

            # Prepare request
            request = self._create_process_request(pdf_path, pdf_content, config, options)

            # Execute with circuit breaker
            response = await self.circuit_breaker.call(self._execute_process_request, request)

            # Process response
            result = self._process_response(response, pdf_path, start_time)

            # Cache the result
            await self.cache.set(
                pdf_content=pdf_content,
                config=config or {},
                options=options or {},
                level=CacheLevel.DOCTAGS_RESULT,
                value=result,
                ttl_seconds=3600  # Cache for 1 hour
            )

            # Update span with success metrics
            if span:
                processing_time = time.time() - start_time
                span.set_attribute("processing.duration_seconds", processing_time)
                span.set_attribute("processing.success", True)
                if hasattr(result, 'text_blocks'):
                    span.set_attribute("doctags.text_blocks_count", len(result.text_blocks))
                if hasattr(result, 'tables'):
                    span.set_attribute("doctags.tables_count", len(result.tables))
                if hasattr(result, 'figures'):
                    span.set_attribute("doctags.figures_count", len(result.figures))
                span.set_status(Status(StatusCode.OK))

            # Update statistics
            self._update_stats(result, start_time)

            return result

        except Exception as e:
            # Update span with error information
            if span:
                processing_time = time.time() - start_time
                span.set_attribute("processing.duration_seconds", processing_time)
                span.set_attribute("processing.success", False)
                span.set_attribute("error.message", str(e))
                span.set_status(Status(StatusCode.ERROR, str(e)))

            logger.error(f"Error processing PDF {pdf_path}: {e}")
            self._update_stats(None, start_time)
            raise self._handle_processing_error(e, pdf_path)
        finally:
            if span:
                span.end()

    async def process_pdf_batch(
        self,
        pdf_paths: list[str],
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[DocTagsResult]:
        """Process multiple PDFs in batch.

        Args:
            pdf_paths: List of PDF file paths
            config: Docling configuration options
            options: Processing options

        Returns:
            List of DocTagsResult objects

        """
        # Create tracing span for batch processing
        tracer = trace.get_tracer(__name__) if trace else None
        span = None
        if tracer:
            span = tracer.start_span("docling_vlm.process_pdf_batch")
            span.set_attribute("batch.total_pdfs", len(pdf_paths))
            span.set_attribute("service.name", "docling-vlm")
            span.set_attribute("processing.method", "batch")

        try:
            results = []

            # Process in batches to manage memory
            batch_size = options.get("batch_size", 5) if options else 5
            if span:
                span.set_attribute("batch.size", batch_size)

            for i in range(0, len(pdf_paths), batch_size):
                batch = pdf_paths[i : i + batch_size]

                # Process batch concurrently
                batch_tasks = [self.process_pdf(pdf_path, config, options) for pdf_path in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # Handle exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing failed for {batch[j]}: {result}")
                        # Create error result
                        error_result = self._create_error_result(batch[j], str(result))
                        results.append(error_result)
                    else:
                        results.append(result)

            # Update span with success metrics
            if span:
                successful_count = len([r for r in results if not isinstance(r, Exception)])
                span.set_attribute("batch.successful_count", successful_count)
                span.set_attribute("batch.failed_count", len(results) - successful_count)
                span.set_attribute("processing.success", True)
                span.set_status(Status(StatusCode.OK))

            return results

        except Exception as e:
            # Update span with error information
            if span:
                span.set_attribute("processing.success", False)
                span.set_attribute("error.message", str(e))
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            if span:
                span.end()

    def _create_process_request(
        self,
        pdf_path: str,
        pdf_content: bytes,
        config: dict[str, Any] | None,
        options: dict[str, Any] | None,
    ) -> ProcessPDFRequest:
        """Create ProcessPDFRequest from parameters."""
        if not ProcessPDFRequest:
            raise DoclingVLMError("gRPC stubs not available")

        # Create DoclingConfig
        docling_config = DoclingConfig()
        if config:
            docling_config.model_name = config.get("model_name", "gemma3-12b")
            docling_config.max_model_len = config.get("max_model_len", 4096)
            docling_config.temperature = config.get("temperature", 0.1)
            docling_config.enable_table_extraction = config.get("enable_table_extraction", True)
            docling_config.enable_figure_extraction = config.get("enable_figure_extraction", True)
            docling_config.enable_text_extraction = config.get("enable_text_extraction", True)

        # Create ProcessingOptions
        processing_options = ProcessingOptions()
        if options:
            processing_options.enable_medical_normalization = options.get(
                "enable_medical_normalization", True
            )
            processing_options.enable_table_fidelity = options.get("enable_table_fidelity", True)
            processing_options.enable_terminology_support = options.get(
                "enable_terminology_support", True
            )
            processing_options.min_confidence_threshold = options.get(
                "min_confidence_threshold", 0.7
            )
            processing_options.enable_quality_validation = options.get(
                "enable_quality_validation", True
            )
            processing_options.timeout_seconds = options.get("timeout_seconds", 300)

        return ProcessPDFRequest(
            pdf_content=pdf_content,
            pdf_path=pdf_path,
            config=docling_config,
            options=processing_options,
        )

    async def _execute_process_request(self, request: ProcessPDFRequest) -> ProcessPDFResponse:
        """Execute the ProcessPDF gRPC request."""
        if not self.stub:
            raise DoclingVLMError("Service stub not available")

        # Create tracing span for gRPC call
        tracer = trace.get_tracer(__name__) if trace else None
        span = None
        if tracer:
            span = tracer.start_span("docling_vlm.grpc_call")
            span.set_attribute("rpc.method", "ProcessPDF")
            span.set_attribute("rpc.service", "docling-vlm")
            span.set_attribute("pdf.size_bytes", len(request.pdf_content))
            span.set_attribute("pdf.path", request.pdf_path)

        try:
            response = await self.stub.ProcessPDF(request)

            # Update span with success metrics
            if span:
                span.set_attribute("grpc.status_code", "OK")
                span.set_attribute("processing.success", True)
                if hasattr(response, 'doctags') and response.doctags:
                    doctags = response.doctags
                    if hasattr(doctags, 'text_blocks'):
                        span.set_attribute("doctags.text_blocks_count", len(doctags.text_blocks))
                    if hasattr(doctags, 'tables'):
                        span.set_attribute("doctags.tables_count", len(doctags.tables))
                    if hasattr(doctags, 'figures'):
                        span.set_attribute("doctags.figures_count", len(doctags.figures))
                span.set_status(Status(StatusCode.OK))

            return response

        except grpc.RpcError as e:
            # Update span with error information
            if span:
                span.set_attribute("grpc.status_code", str(e.code()))
                span.set_attribute("grpc.error_message", e.details())
                span.set_attribute("processing.success", False)
                span.set_status(Status(StatusCode.ERROR, e.details()))
            raise
        except Exception as e:
            # Update span with error information
            if span:
                span.set_attribute("processing.success", False)
                span.set_attribute("error.message", str(e))
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            if span:
                span.end()

    def _process_response(
        self,
        response: ProcessPDFResponse,
        pdf_path: str,
        start_time: float,
    ) -> DocTagsResult:
        """Process the gRPC response."""
        try:
            if response.status != ProcessingStatus.PROCESSING_STATUS_SUCCESS:
                error_msg = response.error_message or "Unknown processing error"
                raise DoclingProcessingError(f"Processing failed: {error_msg}")

            # Extract DocTags result
            doctags = response.doctags
            if not doctags:
                raise DoclingProcessingError("No DocTags result in response")

            return doctags

        except Exception as e:
            logger.error(f"Error processing VLM response: {e}")
            raise DoclingProcessingError(f"Response processing error: {e}")

    def _create_error_result(self, pdf_path: str, error_message: str) -> DocTagsResult:
        """Create error result."""
        # This would create a proper error DocTagsResult
        # For now, raise an exception
        raise DoclingProcessingError(f"Error processing {pdf_path}: {error_message}")

    def _handle_processing_error(self, error: Exception, pdf_path: str) -> Exception:
        """Handle processing errors and convert to appropriate exception types."""
        if isinstance(error, grpc.RpcError):
            if error.code() == grpc.StatusCode.UNAVAILABLE:
                return ServiceUnavailableError(f"VLM service unavailable for {pdf_path}")
            elif error.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                return ServiceTimeoutError(f"VLM service timeout for {pdf_path}")
            else:
                return DoclingProcessingError(f"gRPC error for {pdf_path}: {error.details()}")
        elif isinstance(error, FileNotFoundError):
            return DoclingProcessingError(f"PDF file not found: {pdf_path}")
        else:
            return DoclingProcessingError(f"Processing error for {pdf_path}: {error!s}")

    def _update_stats(self, result: DocTagsResult | None, start_time: float) -> None:
        """Update processing statistics."""
        self._stats["total_requests"] += 1
        processing_time = time.time() - start_time
        self._stats["total_processing_time"] += processing_time

        if result:
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

    async def health_check(self) -> dict[str, Any]:
        """Check VLM service health.

        Returns:
            Health status information

        """
        try:
            if not self.stub:
                return {"status": "unhealthy", "error": "Service stub not available"}

            request = HealthRequest(service_name="docling_vlm")
            response = await self.stub.GetHealth(request)

            return {
                "status": response.status,
                "message": response.message,
                "timestamp": response.timestamp,
                "service_info": {
                    "version": response.service_info.version,
                    "capabilities": list(response.service_info.capabilities),
                },
                "resource_usage": {
                    "cpu_usage_percent": response.resource_usage.cpu_usage_percent,
                    "memory_usage_mb": response.resource_usage.memory_usage_mb,
                    "gpu_usage_percent": response.resource_usage.gpu_usage_percent,
                    "gpu_memory_usage_mb": response.resource_usage.gpu_memory_usage_mb,
                },
                "circuit_breaker_state": self.circuit_breaker.state.value,
            }

        except Exception as e:
            logger.error(f"VLM service health check failed: {e}")
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
        """Get service statistics from VLM service."""
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
                        "timestamp": metric.timestamp,
                        "labels": dict(metric.labels),
                    }
                    for metric in response.metrics
                ],
                "generated_at": response.generated_at,
            }

        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {"error": str(e)}


class DoclingVLMClientManager:
    """Manager for multiple Docling VLM client instances.

    Handles load balancing and failover between multiple VLM services.
    """

    def __init__(self, service_endpoints: list[str]):
        """Initialize the VLM client manager.

        Args:
            service_endpoints: List of gRPC endpoints for VLM services

        """
        self.service_endpoints = service_endpoints
        self.clients: list[DoclingVLMClient] = []
        self.current_client_index = 0

    async def initialize(self) -> None:
        """Initialize all VLM clients."""
        for endpoint in self.service_endpoints:
            client = DoclingVLMClient(endpoint)
            await client.initialize()
            self.clients.append(client)

    async def close(self) -> None:
        """Close all VLM clients."""
        for client in self.clients:
            await client.close()
        self.clients.clear()

    async def process_pdf(
        self,
        pdf_path: str,
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> DocTagsResult:
        """Process PDF using available VLM client.

        Args:
            pdf_path: Path to PDF file
            config: Docling configuration options
            options: Processing options

        Returns:
            DocTagsResult

        """
        if not self.clients:
            raise DoclingVLMError("No VLM clients available")

        # Try current client first
        client = self.clients[self.current_client_index]

        try:
            result = await client.process_pdf(pdf_path, config, options)
            return result

        except Exception as e:
            logger.warning(f"VLM client {self.current_client_index} failed: {e}")

        # Try other clients
        for i, client in enumerate(self.clients):
            if i == self.current_client_index:
                continue

            try:
                result = await client.process_pdf(pdf_path, config, options)
                # Update current client index on success
                self.current_client_index = i
                return result

            except Exception as e:
                logger.warning(f"VLM client {i} failed: {e}")

        # All clients failed
        raise DoclingVLMError("All VLM clients failed")

    async def health_check(self) -> dict[str, Any]:
        """Check health of all VLM clients."""
        health_results = []

        for i, client in enumerate(self.clients):
            try:
                health = await client.health_check()
                health_results.append(
                    {
                        "client_index": i,
                        "endpoint": client.service_endpoint,
                        "health": health,
                    }
                )
            except Exception as e:
                health_results.append(
                    {
                        "client_index": i,
                        "endpoint": client.service_endpoint,
                        "health": {"status": "unhealthy", "error": str(e)},
                    }
                )

        return {
            "total_clients": len(self.clients),
            "healthy_clients": len(
                [r for r in health_results if r.get("health", {}).get("status") == "healthy"]
            ),
            "client_health": health_results,
        }


def create_docling_vlm_client(service_endpoint: str) -> DoclingVLMClient:
    """Create Docling VLM client instance.

    Args:
        service_endpoint: gRPC endpoint for the VLM service

    Returns:
        DoclingVLMClient instance

    """
    return DoclingVLMClient(service_endpoint)


def create_docling_vlm_client_manager(service_endpoints: list[str]) -> DoclingVLMClientManager:
    """Create Docling VLM client manager.

    Args:
        service_endpoints: List of gRPC endpoints for VLM services

    Returns:
        DoclingVLMClientManager instance

    """
    return DoclingVLMClientManager(service_endpoints)
