"""Doctags processing pipeline for Docling VLM.

Handles the complete pipeline from PDF to processed Doctags results.
"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .doctags_interface import (
    DoctagsConfig,
    DoctagsInterface,
    DoctagsProcessingStatus,
    DoctagsResult,
)

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stages for Doctags processing."""

    INITIALIZATION = "initialization"
    VLM_PROCESSING = "vlm_processing"
    ENHANCEMENT = "enhancement"
    QUALITY_VALIDATION = "quality_validation"
    FINALIZATION = "finalization"


class PipelineStatus(Enum):
    """Pipeline processing status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Configuration for Doctags processing pipeline."""

    # Pipeline settings
    max_concurrent_pdfs: int = 5
    pipeline_timeout: int = 1800  # 30 minutes
    enable_stage_checkpointing: bool = True
    enable_progress_tracking: bool = True

    # VLM service configuration
    vlm_service_endpoints: list[str]
    vlm_model_name: str = "gemma3-12b"
    vlm_max_model_len: int = 4096
    vlm_temperature: float = 0.1

    # Processing options
    enable_table_extraction: bool = True
    enable_figure_extraction: bool = True
    enable_text_extraction: bool = True

    # Enhancement options
    enable_medical_normalization: bool = True
    enable_table_fidelity: bool = True
    enable_terminology_support: bool = True

    # Quality settings
    min_confidence_threshold: float = 0.7
    enable_quality_validation: bool = True

    # Performance settings
    batch_size: int = 5
    timeout_seconds: int = 300
    enable_caching: bool = True
    cache_ttl: int = 3600


@dataclass
class PipelineResult:
    """Result from pipeline processing."""

    # Pipeline identification
    pipeline_id: str
    status: PipelineStatus

    # Processing results
    results: list[DoctagsResult]
    failed_pdfs: list[str]

    # Pipeline metadata
    start_time: float
    end_time: float | None = None
    total_processing_time: float = 0.0

    # Stage information
    completed_stages: list[PipelineStage] = None
    current_stage: PipelineStage | None = None

    # Statistics
    total_pdfs: int = 0
    successful_pdfs: int = 0
    failed_pdfs_count: int = 0

    # Error information
    error_message: str | None = None
    stage_errors: dict[str, str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.completed_stages is None:
            self.completed_stages = []
        if self.stage_errors is None:
            self.stage_errors = {}


@dataclass
class StageResult:
    """Result from a pipeline stage."""

    stage: PipelineStage
    status: PipelineStatus
    start_time: float
    end_time: float | None = None
    processing_time: float = 0.0
    result_data: dict[str, Any] | None = None
    error_message: str | None = None


class DoctagsPipeline:
    """Doctags processing pipeline for Docling VLM.

    Handles:
    - Complete pipeline from PDF to processed Doctags results
    - Stage-based processing with checkpointing
    - Progress tracking and error handling
    - Batch processing with concurrency control
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the Doctags pipeline.

        Args:
            config: Configuration for the pipeline

        """
        self.config = config
        self.doctags_interface: DoctagsInterface | None = None
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._stage_results: dict[str, list[StageResult]] = {}

    async def initialize(self) -> None:
        """Initialize the pipeline."""
        try:
            # Create Doctags interface configuration
            doctags_config = DoctagsConfig(
                vlm_service_endpoints=self.config.vlm_service_endpoints,
                vlm_model_name=self.config.vlm_model_name,
                vlm_max_model_len=self.config.vlm_max_model_len,
                vlm_temperature=self.config.vlm_temperature,
                enable_table_extraction=self.config.enable_table_extraction,
                enable_figure_extraction=self.config.enable_figure_extraction,
                enable_text_extraction=self.config.enable_text_extraction,
                enable_medical_normalization=self.config.enable_medical_normalization,
                enable_table_fidelity=self.config.enable_table_fidelity,
                enable_terminology_support=self.config.enable_terminology_support,
                min_confidence_threshold=self.config.min_confidence_threshold,
                enable_quality_validation=self.config.enable_quality_validation,
                batch_size=self.config.batch_size,
                timeout_seconds=self.config.timeout_seconds,
                enable_caching=self.config.enable_caching,
                cache_ttl=self.config.cache_ttl,
            )

            # Initialize Doctags interface
            self.doctags_interface = DoctagsInterface(doctags_config)
            await self.doctags_interface.initialize()

            logger.info("Doctags pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Doctags pipeline: {e}")
            raise

    async def close(self) -> None:
        """Close the pipeline."""
        # Cancel active tasks
        for task_id, task in self._active_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled pipeline task {task_id}")

        # Close Doctags interface
        if self.doctags_interface:
            await self.doctags_interface.close()

    async def process_pdfs(
        self,
        pdf_paths: list[str],
        pipeline_id: str | None = None,
    ) -> PipelineResult:
        """Process multiple PDFs through the pipeline.

        Args:
            pdf_paths: List of PDF file paths
            pipeline_id: Optional pipeline ID

        Returns:
            PipelineResult with processing results

        """
        if not pipeline_id:
            pipeline_id = f"pipeline_{int(time.time())}"

        start_time = time.time()

        try:
            # Initialize pipeline result
            result = PipelineResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.RUNNING,
                results=[],
                failed_pdfs=[],
                start_time=start_time,
                total_pdfs=len(pdf_paths),
            )

            # Stage 1: Initialization
            await self._execute_stage(
                pipeline_id,
                PipelineStage.INITIALIZATION,
                self._stage_initialization,
                pdf_paths,
                result,
            )

            # Stage 2: VLM Processing
            await self._execute_stage(
                pipeline_id,
                PipelineStage.VLM_PROCESSING,
                self._stage_vlm_processing,
                pdf_paths,
                result,
            )

            # Stage 3: Enhancement
            await self._execute_stage(
                pipeline_id, PipelineStage.ENHANCEMENT, self._stage_enhancement, result
            )

            # Stage 4: Quality Validation
            await self._execute_stage(
                pipeline_id,
                PipelineStage.QUALITY_VALIDATION,
                self._stage_quality_validation,
                result,
            )

            # Stage 5: Finalization
            await self._execute_stage(
                pipeline_id, PipelineStage.FINALIZATION, self._stage_finalization, result
            )

            # Update final result
            result.status = PipelineStatus.COMPLETED
            result.end_time = time.time()
            result.total_processing_time = result.end_time - result.start_time
            result.successful_pdfs = len(result.results)
            result.failed_pdfs_count = len(result.failed_pdfs)

            logger.info(f"Pipeline {pipeline_id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
            result.status = PipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = time.time()
            result.total_processing_time = result.end_time - result.start_time
            return result

    async def _execute_stage(
        self, pipeline_id: str, stage: PipelineStage, stage_func, *args, **kwargs
    ) -> StageResult:
        """Execute a pipeline stage."""
        stage_start_time = time.time()

        try:
            logger.info(f"Pipeline {pipeline_id}: Starting stage {stage.value}")

            # Execute stage
            stage_result = await stage_func(*args, **kwargs)

            # Create stage result
            result = StageResult(
                stage=stage,
                status=PipelineStatus.COMPLETED,
                start_time=stage_start_time,
                end_time=time.time(),
                processing_time=time.time() - stage_start_time,
                result_data=stage_result,
            )

            # Store stage result
            if pipeline_id not in self._stage_results:
                self._stage_results[pipeline_id] = []
            self._stage_results[pipeline_id].append(result)

            logger.info(
                f"Pipeline {pipeline_id}: Completed stage {stage.value} in {result.processing_time:.2f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Pipeline {pipeline_id}: Stage {stage.value} failed: {e}")

            # Create error stage result
            result = StageResult(
                stage=stage,
                status=PipelineStatus.FAILED,
                start_time=stage_start_time,
                end_time=time.time(),
                processing_time=time.time() - stage_start_time,
                error_message=str(e),
            )

            # Store stage result
            if pipeline_id not in self._stage_results:
                self._stage_results[pipeline_id] = []
            self._stage_results[pipeline_id].append(result)

            raise

    async def _stage_initialization(
        self, pdf_paths: list[str], result: PipelineResult
    ) -> dict[str, Any]:
        """Initialize stage - validate PDFs and prepare for processing."""
        validated_pdfs = []
        invalid_pdfs = []

        for pdf_path in pdf_paths:
            try:
                # Validate PDF path
                if not Path(pdf_path).exists():
                    invalid_pdfs.append(pdf_path)
                    continue

                # Check file size (basic validation)
                file_size = Path(pdf_path).stat().st_size
                if file_size == 0:
                    invalid_pdfs.append(pdf_path)
                    continue

                validated_pdfs.append(pdf_path)

            except Exception as e:
                logger.warning(f"PDF validation failed for {pdf_path}: {e}")
                invalid_pdfs.append(pdf_path)

        # Update result
        result.failed_pdfs.extend(invalid_pdfs)

        return {
            "validated_pdfs": validated_pdfs,
            "invalid_pdfs": invalid_pdfs,
            "total_validated": len(validated_pdfs),
            "total_invalid": len(invalid_pdfs),
        }

    async def _stage_vlm_processing(
        self, pdf_paths: list[str], result: PipelineResult
    ) -> dict[str, Any]:
        """VLM processing stage - process PDFs with Docling VLM."""
        if not self.doctags_interface:
            raise Exception("Doctags interface not initialized")

        # Process PDFs in batches
        processed_results = []
        failed_pdfs = []

        for i in range(0, len(pdf_paths), self.config.max_concurrent_pdfs):
            batch = pdf_paths[i : i + self.config.max_concurrent_pdfs]

            # Process batch concurrently
            batch_tasks = [self.doctags_interface.process_pdf(pdf_path) for pdf_path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle results
            for j, batch_result in enumerate(batch_results):
                if isinstance(batch_result, Exception):
                    logger.error(f"VLM processing failed for {batch[j]}: {batch_result}")
                    failed_pdfs.append(batch[j])
                else:
                    processed_results.append(batch_result)

        # Update result
        result.results.extend(processed_results)
        result.failed_pdfs.extend(failed_pdfs)

        return {
            "processed_results": len(processed_results),
            "failed_pdfs": len(failed_pdfs),
            "total_processed": len(processed_results) + len(failed_pdfs),
        }

    async def _stage_enhancement(self, result: PipelineResult) -> dict[str, Any]:
        """Enhancement stage - apply medical normalization, table fidelity, terminology."""
        enhanced_results = []
        enhancement_errors = []

        for doctags_result in result.results:
            try:
                # Apply enhancements (already done in DoctagsInterface)
                # This stage could add additional enhancements if needed
                enhanced_results.append(doctags_result)

            except Exception as e:
                logger.error(f"Enhancement failed for {doctags_result.pdf_path}: {e}")
                enhancement_errors.append(
                    {
                        "pdf_path": doctags_result.pdf_path,
                        "error": str(e),
                    }
                )

        return {
            "enhanced_results": len(enhanced_results),
            "enhancement_errors": len(enhancement_errors),
            "total_enhanced": len(enhanced_results) + len(enhancement_errors),
        }

    async def _stage_quality_validation(self, result: PipelineResult) -> dict[str, Any]:
        """Quality validation stage - validate processing quality."""
        validated_results = []
        quality_issues = []

        for doctags_result in result.results:
            try:
                # Quality validation (already done in DoctagsInterface)
                # This stage could add additional validation if needed
                validated_results.append(doctags_result)

                # Collect quality issues
                if doctags_result.quality_issues:
                    quality_issues.extend(doctags_result.quality_issues)

            except Exception as e:
                logger.error(f"Quality validation failed for {doctags_result.pdf_path}: {e}")

        return {
            "validated_results": len(validated_results),
            "quality_issues": len(quality_issues),
            "total_validated": len(validated_results),
        }

    async def _stage_finalization(self, result: PipelineResult) -> dict[str, Any]:
        """Finalization stage - finalize results and prepare output."""
        finalized_results = []

        for doctags_result in result.results:
            try:
                # Finalize result
                doctags_result.status = DoctagsProcessingStatus.COMPLETED
                finalized_results.append(doctags_result)

            except Exception as e:
                logger.error(f"Finalization failed for {doctags_result.pdf_path}: {e}")

        return {
            "finalized_results": len(finalized_results),
            "total_finalized": len(finalized_results),
        }

    async def get_pipeline_status(self, pipeline_id: str) -> dict[str, Any]:
        """Get status of a pipeline."""
        if pipeline_id not in self._stage_results:
            return {"status": "not_found"}

        stage_results = self._stage_results[pipeline_id]

        # Determine overall status
        if not stage_results:
            status = "pending"
        elif all(stage.status == PipelineStatus.COMPLETED for stage in stage_results):
            status = "completed"
        elif any(stage.status == PipelineStatus.FAILED for stage in stage_results):
            status = "failed"
        else:
            status = "running"

        return {
            "pipeline_id": pipeline_id,
            "status": status,
            "stages": [
                {
                    "stage": stage.stage.value,
                    "status": stage.status.value,
                    "processing_time": stage.processing_time,
                    "error_message": stage.error_message,
                }
                for stage in stage_results
            ],
            "total_stages": len(stage_results),
            "completed_stages": len(
                [s for s in stage_results if s.status == PipelineStatus.COMPLETED]
            ),
        }

    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel a running pipeline."""
        if pipeline_id in self._active_tasks:
            task = self._active_tasks[pipeline_id]
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled pipeline {pipeline_id}")
                return True

        return False

    async def health_check(self) -> dict[str, Any]:
        """Check pipeline health."""
        health_info = {
            "pipeline_status": "healthy",
            "config": asdict(self.config),
        }

        # Check Doctags interface
        if self.doctags_interface:
            try:
                interface_health = await self.doctags_interface.health_check()
                health_info["doctags_interface"] = interface_health
            except Exception as e:
                health_info["doctags_interface"] = {"status": "unhealthy", "error": str(e)}
                health_info["pipeline_status"] = "unhealthy"
        else:
            health_info["doctags_interface"] = {"status": "not_initialized"}
            health_info["pipeline_status"] = "unhealthy"

        # Active tasks
        health_info["active_tasks"] = len(self._active_tasks)
        health_info["stage_results"] = len(self._stage_results)

        return health_info

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "pipeline_stats": {
                "active_tasks": len(self._active_tasks),
                "stage_results": len(self._stage_results),
                "max_concurrent_pdfs": self.config.max_concurrent_pdfs,
                "pipeline_timeout": self.config.pipeline_timeout,
            },
        }


def create_doctags_pipeline(config: PipelineConfig) -> DoctagsPipeline:
    """Create Doctags pipeline instance.

    Args:
        config: Pipeline configuration

    Returns:
        DoctagsPipeline instance

    """
    return DoctagsPipeline(config)


def create_default_doctags_pipeline(vlm_service_endpoints: list[str]) -> DoctagsPipeline:
    """Create Doctags pipeline with default configuration.

    Args:
        vlm_service_endpoints: List of VLM service endpoints

    Returns:
        DoctagsPipeline instance

    """
    config = PipelineConfig(
        max_concurrent_pdfs=5,
        pipeline_timeout=1800,
        enable_stage_checkpointing=True,
        enable_progress_tracking=True,
        vlm_service_endpoints=vlm_service_endpoints,
        vlm_model_name="gemma3-12b",
        vlm_max_model_len=4096,
        vlm_temperature=0.1,
        enable_table_extraction=True,
        enable_figure_extraction=True,
        enable_text_extraction=True,
        enable_medical_normalization=True,
        enable_table_fidelity=True,
        enable_terminology_support=True,
        min_confidence_threshold=0.7,
        enable_quality_validation=True,
        batch_size=5,
        timeout_seconds=300,
        enable_caching=True,
        cache_ttl=3600,
    )

    return DoctagsPipeline(config)
