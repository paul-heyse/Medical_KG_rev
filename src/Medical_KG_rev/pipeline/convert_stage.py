"""Convert stage implementation for document processing.

This module implements the Convert stage for processing PDF documents
using Docling VLM.
"""

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from Medical_KG_rev.pipeline.stages import ConvertStage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class ConvertStageConfig(BaseModel):
    """Configuration for Convert stage."""

    enable_docling: bool = Field(default=True, description="Enable Docling VLM processing")
    output_format: str = Field(default="doctags", description="Output format")
    docling_service_url: str | None = Field(default=None, description="Docling service URL")
    max_processing_time_seconds: int = Field(default=300, description="Maximum processing time")
    enable_retry: bool = Field(default=True, description="Enable retry on failure")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_delay_seconds: int = Field(default=5, description="Delay between retries")


class ConvertStageImpl(ConvertStage):
    """Enhanced Convert stage implementation."""

    def __init__(self, config: ConvertStageConfig | None = None):
        """Initialize Convert stage implementation.

        Args:
            config: Configuration for the stage

        """
        self.stage_config = config or ConvertStageConfig()

        # Convert to dict for base class
        base_config = self.stage_config.dict()
        super().__init__(base_config)

        # Stage-specific attributes
        self.docling_service_url = self.stage_config.docling_service_url
        self.max_processing_time = self.stage_config.max_processing_time_seconds
        self.enable_retry = self.stage_config.enable_retry
        self.max_retries = self.stage_config.max_retries
        self.retry_delay = self.stage_config.retry_delay_seconds

        self.logger.info(
            "Initialized Convert stage implementation",
            extra={
                "enable_docling": self.enable_docling,
                "output_format": self.output_format,
                "docling_service_url": self.docling_service_url,
                "max_processing_time": self.max_processing_time,
                "enable_retry": self.enable_retry,
                "max_retries": self.max_retries,
            },
        )

    def execute(self, input_data: dict[str, Any]) -> StageResult:
        """Execute Convert stage with enhanced functionality.

        Args:
            input_data: Input data containing document information

        Returns:
            Stage execution result

        """
        start_time = time.perf_counter()

        try:
            self.logger.info("Starting Convert stage execution")

            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data for Convert stage")

            # Extract input parameters
            document_id = input_data.get("document_id")
            pdf_path = input_data.get("pdf_path")
            docling_service_url = input_data.get("docling_service_url") or self.docling_service_url

            if not document_id or not pdf_path:
                raise ValueError("Missing required input: document_id or pdf_path")

            # Process document with retry logic
            result_data = self._process_document_with_retry(
                document_id, pdf_path, docling_service_url
            )

            # Calculate duration
            end_time = time.perf_counter()
            duration = end_time - start_time

            self.logger.info(
                "Convert stage completed successfully",
                extra={
                    "document_id": document_id,
                    "duration_seconds": duration,
                    "output_keys": list(result_data.keys()),
                },
            )

            return StageResult(
                stage_name=self.name,
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_data=input_data,
                output_data=result_data,
                metadata={
                    "enable_docling": self.enable_docling,
                    "output_format": self.output_format,
                    "docling_service_url": docling_service_url,
                    "processing_method": result_data.get("processing_method"),
                    "retry_enabled": self.enable_retry,
                },
            )

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time

            self.logger.error(
                "Convert stage failed",
                extra={
                    "error": str(e),
                    "duration_seconds": duration,
                },
            )

            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_data=input_data,
                error_message=str(e),
            )

    def _process_document_with_retry(
        self, document_id: str, pdf_path: str, docling_service_url: str | None = None
    ) -> dict[str, Any]:
        """Process document with retry logic.

        Args:
            document_id: Document identifier
            pdf_path: Path to PDF file
            docling_service_url: URL to Docling service

        Returns:
            Processing result data

        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(
                    f"Processing document (attempt {attempt + 1}/{self.max_retries + 1})",
                    extra={"document_id": document_id, "pdf_path": pdf_path},
                )

                # Process document
                result = self._process_document(document_id, pdf_path, docling_service_url)

                # Add retry metadata
                result["retry_attempts"] = attempt
                result["retry_successful"] = True

                return result

            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Document processing attempt {attempt + 1} failed",
                    extra={
                        "document_id": document_id,
                        "error": str(e),
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                    },
                )

                # Check if we should retry
                if not self.enable_retry or attempt >= self.max_retries:
                    break

                # Wait before retry
                if attempt < self.max_retries:
                    self.logger.info(f"Waiting {self.retry_delay} seconds before retry")
                    time.sleep(self.retry_delay)

        # All retries failed
        self.logger.error(
            f"Document processing failed after {self.max_retries + 1} attempts",
            extra={
                "document_id": document_id,
                "pdf_path": pdf_path,
                "last_error": str(last_error),
            },
        )

        raise last_error or Exception("Document processing failed")

    def _process_document(
        self, document_id: str, pdf_path: str, docling_service_url: str | None = None
    ) -> dict[str, Any]:
        """Process document using Docling VLM.

        Args:
            document_id: Document identifier
            pdf_path: Path to PDF file
            docling_service_url: URL to Docling service

        Returns:
            Processing result data

        """
        try:
            # Check processing time limit
            start_time = time.perf_counter()

            if self.enable_docling and docling_service_url:
                # Use Docling VLM service
                result = self._process_with_docling(document_id, pdf_path, docling_service_url)
            else:
                raise ValueError("Docling VLM service URL required for processing")

            # Check processing time
            processing_time = time.perf_counter() - start_time
            if processing_time > self.max_processing_time:
                self.logger.warning(
                    f"Processing time exceeded limit: {processing_time:.2f}s > {self.max_processing_time}s",
                    extra={"document_id": document_id},
                )

            return result

        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            raise

    def _process_with_docling(
        self, document_id: str, pdf_path: str, service_url: str
    ) -> dict[str, Any]:
        """Process document with Docling VLM.

        Args:
            document_id: Document identifier
            pdf_path: Path to PDF file
            service_url: Docling service URL

        Returns:
            Processing result

        """
        try:
            # Import here to avoid circular imports
            from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMService

            # Initialize Docling service
            docling_service = DoclingVLMService(docker_service_url=service_url)

            # Process document
            result = docling_service.process_pdf(pdf_path, document_id)

            return {
                "document_id": document_id,
                "processing_method": "docling",
                "service_url": service_url,
                "result": result,
                "metadata": {
                    "model_name": "google/gemma-3-12b-it",
                    "processing_time": result.processing_time_seconds,
                    "gpu_memory_used": result.gpu_memory_used_mb,
                    "docling_version": "1.0.0",
                },
            }

        except Exception as e:
            self.logger.error(f"Docling processing failed: {e}")
            raise


    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for Convert stage.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise

        """
        required_fields = ["document_id", "pdf_path"]

        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"Missing required field: {field}")
                return False

        # Check if PDF file exists
        pdf_path = input_data.get("pdf_path")
        if pdf_path and not Path(pdf_path).exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return False

        # Check PDF file size
        if pdf_path:
            file_size = Path(pdf_path).stat().st_size
            if file_size == 0:
                self.logger.error(f"PDF file is empty: {pdf_path}")
                return False

            # Check if file is too large (e.g., > 100MB)
            max_size = 100 * 1024 * 1024  # 100MB
            if file_size > max_size:
                self.logger.warning(
                    f"PDF file is large: {file_size / (1024 * 1024):.1f}MB",
                    extra={"pdf_path": pdf_path},
                )

        return True

    def get_stage_info(self) -> dict[str, Any]:
        """Get stage information.

        Returns:
            Stage information

        """
        return {
            "stage_name": self.name,
            "config": self.stage_config.dict(),
            "capabilities": {
                "docling_processing": self.enable_docling,
                "retry_logic": self.enable_retry,
                "output_formats": [self.output_format],
            },
            "limits": {
                "max_processing_time_seconds": self.max_processing_time,
                "max_retries": self.max_retries,
                "retry_delay_seconds": self.retry_delay,
            },
        }

    def health_check(self) -> dict[str, Any]:
        """Check stage health.

        Returns:
            Health status information

        """
        try:
            health = {
                "status": "healthy",
                "stage_name": self.name,
                "config": self.stage_config.dict(),
                "capabilities": {
                    "docling_processing": self.enable_docling,
                    "retry_logic": self.enable_retry,
                },
                "limits": {
                    "max_processing_time_seconds": self.max_processing_time,
                    "max_retries": self.max_retries,
                    "retry_delay_seconds": self.retry_delay,
                },
            }

            # Check if Docling service is available
            if self.enable_docling and self.docling_service_url:
                try:
                    import requests

                    response = requests.get(f"{self.docling_service_url}/health", timeout=5)
                    health["docling_service_status"] = (
                        "healthy" if response.status_code == 200 else "unhealthy"
                    )
                except Exception as e:
                    health["docling_service_status"] = "unhealthy"
                    health["docling_service_error"] = str(e)

            return health

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }
