"""Docling VLM service for biomedical document processing.

This module provides Docling Vision-Language Model integration including:
- HTTP client for communicating with Docker Docling VLM service
- PDF processing interface compatible with existing pipeline
- Error handling for Docker service communication
- Batch processing for multiple PDFs
- Performance monitoring and metrics for VLM processing
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import asyncio
import logging
import time

import httpx


logger = logging.getLogger(__name__)


class VLMProcessingStatus(Enum):
    """VLM processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class DoclingVLMResult:
    """Result from Docling VLM processing."""

    pdf_path: str | None = None
    status: VLMProcessingStatus = VLMProcessingStatus.COMPLETED
    text_content: str = ""
    tables: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    error_message: str | None = None
    model_version: str | None = None
    gpu_memory_used: float | None = None
    document_id: str | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        # Maintain backwards compatibility with older tests/fixtures that only provided a `text`
        # payload while the new implementation prefers `text_content`.
        if not self.text_content and self.text:
            self.text_content = self.text
        elif self.text_content and not self.text:
            self.text = self.text_content


@dataclass
class DoclingVLMConfig:
    """Configuration for Docling VLM service."""

    service_url: str = "http://docling-vlm:8000"
    timeout: int = 300
    retry_attempts: int = 3
    batch_size: int = 8
    model_name: str = "gemma3-12b"
    gpu_memory_fraction: float = 0.95
    max_model_len: int = 4096


class DoclingVLMService:
    """Docling VLM service for biomedical document processing.

    Handles:
    - HTTP client for communicating with Docker Docling VLM service
    - PDF processing interface compatible with existing pipeline
    - Error handling for Docker service communication
    - Batch processing for multiple PDFs
    - Performance monitoring and metrics for VLM processing
    """

    def __init__(self, config: DoclingVLMConfig):
        """Initialize the Docling VLM service.

        Args:
        ----
            config: Configuration for the VLM service

        """
        self.config = config
        self.session: httpx.AsyncClient | None = None
        self._processing_stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self) -> None:
        """Initialize the VLM service."""
        try:
            self.session = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )

            # Test connection to VLM service
            await self._test_connection()
            logger.info(f"Docling VLM service initialized at {self.config.service_url}")

        except Exception as e:
            logger.error(f"Failed to initialize Docling VLM service: {e}")
            raise

    async def close(self) -> None:
        """Close the VLM service."""
        if self.session:
            await self.session.aclose()
            self.session = None

    async def _test_connection(self) -> None:
        """Test connection to VLM service."""
        try:
            response = await self.session.get(f"{self.config.service_url}/health")
            response.raise_for_status()

            health_data = response.json()
            if not health_data.get("status") == "healthy":
                raise Exception(f"VLM service not healthy: {health_data}")

        except Exception as e:
            logger.error(f"VLM service connection test failed: {e}")
            raise

    async def process_pdf(self, pdf_path: str) -> DoclingVLMResult:
        """Process a single PDF using Docling VLM.

        Args:
        ----
            pdf_path: Path to the PDF file

        Returns:
        -------
            DoclingVLMResult with processing results

        """
        start_time = time.time()

        try:
            # Validate PDF path
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Prepare request data
            request_data = {
                "pdf_path": pdf_path,
                "model_name": self.config.model_name,
                "gpu_memory_fraction": self.config.gpu_memory_fraction,
                "max_model_len": self.config.max_model_len,
            }

            # Send request to VLM service
            response = await self._send_processing_request(request_data)

            # Process response
            result = self._process_response(response, pdf_path, start_time)

            # Update statistics
            self._update_stats(result)

            return result

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return self._create_error_result(pdf_path, str(e), start_time)

    async def process_pdf_batch(self, pdf_paths: list[str]) -> list[DoclingVLMResult]:
        """Process multiple PDFs in batch.

        Args:
        ----
            pdf_paths: List of PDF file paths

        Returns:
        -------
            List of DoclingVLMResult objects

        """
        results = []

        # Process in batches to manage memory
        for i in range(0, len(pdf_paths), self.config.batch_size):
            batch = pdf_paths[i : i + self.config.batch_size]

            # Process batch concurrently
            batch_tasks = [self.process_pdf(pdf_path) for pdf_path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    error_result = self._create_error_result(batch[j], str(result), time.time())
                    results.append(error_result)
                else:
                    results.append(result)

        return results

    async def _send_processing_request(self, request_data: dict[str, Any]) -> httpx.Response:
        """Send processing request to VLM service."""
        for attempt in range(self.config.retry_attempts):
            try:
                response = await self.session.post(
                    f"{self.config.service_url}/process", json=request_data
                )
                response.raise_for_status()
                return response

            except httpx.TimeoutException:
                if attempt == self.config.retry_attempts - 1:
                    raise Exception("VLM service timeout after all retries")
                logger.warning(f"VLM service timeout, retrying... (attempt {attempt + 1})")
                await asyncio.sleep(2**attempt)  # Exponential backoff

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:  # Server error
                    if attempt == self.config.retry_attempts - 1:
                        raise Exception(f"VLM service error: {e.response.status_code}")
                    logger.warning(f"VLM service error, retrying... (attempt {attempt + 1})")
                    await asyncio.sleep(2**attempt)
                else:  # Client error
                    raise Exception(f"VLM service client error: {e.response.status_code}")

    def _process_response(
        self, response: httpx.Response, pdf_path: str, start_time: float
    ) -> DoclingVLMResult:
        """Process VLM service response."""
        try:
            data = response.json()

            return DoclingVLMResult(
                pdf_path=pdf_path,
                status=VLMProcessingStatus.COMPLETED,
                text_content=data.get("text_content", ""),
                tables=data.get("tables", []),
                figures=data.get("figures", []),
                metadata=data.get("metadata", {}),
                processing_time=time.time() - start_time,
                model_version=data.get("model_version"),
                gpu_memory_used=data.get("gpu_memory_used"),
            )

        except Exception as e:
            logger.error(f"Error processing VLM response: {e}")
            return self._create_error_result(
                pdf_path, f"Response processing error: {e}", start_time
            )

    def _create_error_result(
        self, pdf_path: str, error_message: str, start_time: float
    ) -> DoclingVLMResult:
        """Create error result."""
        return DoclingVLMResult(
            pdf_path=pdf_path,
            status=VLMProcessingStatus.FAILED,
            text_content="",
            tables=[],
            figures=[],
            metadata={},
            processing_time=time.time() - start_time,
            error_message=error_message,
        )

    def _update_stats(self, result: DoclingVLMResult) -> None:
        """Update processing statistics."""
        self._processing_stats["total_processed"] += 1
        self._processing_stats["total_processing_time"] += result.processing_time

        if result.status == VLMProcessingStatus.COMPLETED:
            self._processing_stats["successful_processed"] += 1
        else:
            self._processing_stats["failed_processed"] += 1

        # Calculate average processing time
        if self._processing_stats["total_processed"] > 0:
            self._processing_stats["average_processing_time"] = (
                self._processing_stats["total_processing_time"]
                / self._processing_stats["total_processed"]
            )

    async def health_check(self) -> dict[str, Any]:
        """Check VLM service health.

        Returns
        -------
            Health status information

        """
        try:
            if not self.session:
                return {"status": "unhealthy", "error": "Service not initialized"}

            response = await self.session.get(f"{self.config.service_url}/health")
            response.raise_for_status()

            health_data = response.json()
            return {
                "status": "healthy",
                "service_url": self.config.service_url,
                "model_name": self.config.model_name,
                "processing_stats": self._processing_stats,
                **health_data,
            }

        except Exception as e:
            logger.error(f"VLM service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_url": self.config.service_url,
                "processing_stats": self._processing_stats,
            }

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "config": asdict(self.config),
            "processing_stats": self._processing_stats.copy(),
            "service_initialized": self.session is not None,
        }

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.config.model_name,
            "gpu_memory_fraction": self.config.gpu_memory_fraction,
            "max_model_len": self.config.max_model_len,
            "batch_size": self.config.batch_size,
        }


class DoclingVLMServiceManager:
    """Manager for multiple Docling VLM service instances.

    Handles load balancing and failover between multiple VLM services.
    """

    def __init__(self, configs: list[DoclingVLMConfig]):
        """Initialize the VLM service manager.

        Args:
        ----
            configs: List of VLM service configurations

        """
        self.configs = configs
        self.services: list[DoclingVLMService] = []
        self.current_service_index = 0

    async def initialize(self) -> None:
        """Initialize all VLM services."""
        for config in self.configs:
            service = DoclingVLMService(config)
            await service.initialize()
            self.services.append(service)

    async def close(self) -> None:
        """Close all VLM services."""
        for service in self.services:
            await service.close()
        self.services.clear()

    async def process_pdf(self, pdf_path: str) -> DoclingVLMResult:
        """Process PDF using available VLM service.

        Args:
        ----
            pdf_path: Path to PDF file

        Returns:
        -------
            DoclingVLMResult

        """
        if not self.services:
            raise Exception("No VLM services available")

        # Try current service first
        service = self.services[self.current_service_index]

        try:
            result = await service.process_pdf(pdf_path)
            if result.status == VLMProcessingStatus.COMPLETED:
                return result
        except Exception as e:
            logger.warning(f"VLM service {self.current_service_index} failed: {e}")

        # Try other services
        for i, service in enumerate(self.services):
            if i == self.current_service_index:
                continue

            try:
                result = await service.process_pdf(pdf_path)
                if result.status == VLMProcessingStatus.COMPLETED:
                    self.current_service_index = i  # Switch to working service
                    return result
            except Exception as e:
                logger.warning(f"VLM service {i} failed: {e}")

        # All services failed
        raise Exception("All VLM services failed")

    async def health_check(self) -> dict[str, Any]:
        """Check health of all VLM services."""
        health_results = []

        for i, service in enumerate(self.services):
            try:
                health = await service.health_check()
                health_results.append(
                    {"service_index": i, "config": asdict(service.config), **health}
                )
            except Exception as e:
                health_results.append(
                    {
                        "service_index": i,
                        "config": asdict(service.config),
                        "status": "unhealthy",
                        "error": str(e),
                    }
                )

        return {
            "total_services": len(self.services),
            "healthy_services": len([h for h in health_results if h["status"] == "healthy"]),
            "services": health_results,
        }


def create_docling_vlm_service(config: DoclingVLMConfig) -> DoclingVLMService:
    """Create Docling VLM service instance.

    Args:
    ----
        config: VLM service configuration

    Returns:
    -------
        DoclingVLMService instance

    """
    return DoclingVLMService(config)


def create_docling_vlm_service_manager(configs: list[DoclingVLMConfig]) -> DoclingVLMServiceManager:
    """Create Docling VLM service manager.

    Args:
    ----
        configs: List of VLM service configurations

    Returns:
    -------
        DoclingVLMServiceManager instance

    """
    return DoclingVLMServiceManager(configs)
