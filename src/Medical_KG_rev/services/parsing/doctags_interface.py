"""Doctags-based processing interface for Docling VLM.

Handles Doctags output format from Docling VLM.
"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .docling_vlm_client import DoclingVLMClientManager
from .medical_normalization import MedicalNormalizer, NormalizationLevel
from .medical_terminology import MedicalTerminologyProcessor, TerminologyType
from .table_fidelity import TableFidelityPreserver

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for Doctags interface."""

    VLM_ONLY = "vlm_only"
    VLM_WITH_ENHANCEMENT = "vlm_with_enhancement"
    FULL_PIPELINE = "full_pipeline"


class DoctagsProcessingStatus(Enum):
    """Status of Doctags processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class DoctagsConfig:
    """Configuration for Doctags processing."""

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
class DoctagsResult:
    """Result from Doctags processing."""

    # Document identification
    document_id: str
    pdf_path: str

    # Document structure
    title: str | None = None
    sections: list[dict[str, Any]] = None
    pages: list[dict[str, Any]] = None

    # Extracted content
    tables: list[dict[str, Any]] = None
    figures: list[dict[str, Any]] = None
    text_blocks: list[dict[str, Any]] = None

    # Document metadata
    document_type: str | None = None
    language: str | None = None
    author: str | None = None
    keywords: list[str] = None
    abstract: str | None = None

    # Processing information
    processing_time: float = 0.0
    status: DoctagsProcessingStatus = DoctagsProcessingStatus.PENDING
    error_message: str | None = None

    # VLM-specific information
    model_version: str | None = None
    gpu_memory_used: float | None = None

    # Provenance tracking
    processing_method: str = "docling_vlm"
    processing_config: dict[str, Any] = None

    # Enhanced processing results
    normalized_content: dict[str, Any] | None = None
    table_preservation: dict[str, Any] | None = None
    terminology_processing: dict[str, Any] | None = None

    # Quality metrics
    confidence_score: float = 1.0
    completeness_score: float = 1.0
    accuracy_score: float = 1.0
    quality_issues: list[dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.sections is None:
            self.sections = []
        if self.pages is None:
            self.pages = []
        if self.tables is None:
            self.tables = []
        if self.figures is None:
            self.figures = []
        if self.text_blocks is None:
            self.text_blocks = []
        if self.keywords is None:
            self.keywords = []
        if self.processing_config is None:
            self.processing_config = {}
        if self.normalized_content is None:
            self.normalized_content = {}
        if self.table_preservation is None:
            self.table_preservation = {}
        if self.terminology_processing is None:
            self.terminology_processing = {}
        if self.quality_issues is None:
            self.quality_issues = []


class DoctagsInterface:
    """Doctags-based processing interface for Docling VLM.

    Handles:
    - Doctags output format from Docling VLM
    - Clean interface focused on Docling VLM
    - Enhanced processing with medical normalization, table fidelity, terminology
    - Quality validation and provenance tracking
    """

    def __init__(self, config: DoctagsConfig):
        """Initialize the Doctags interface.

        Args:
        ----
            config: Configuration for the Doctags interface

        """
        self.config = config
        self.vlm_client_manager: DoclingVLMClientManager | None = None

        # Enhancement components
        self.medical_normalizer: MedicalNormalizer | None = None
        self.table_preserver: TableFidelityPreserver | None = None
        self.terminology_processor: MedicalTerminologyProcessor | None = None

        # Processing cache
        self._processing_cache: dict[str, DoctagsResult] = {}
        self._cache_timestamps: dict[str, float] = {}

    async def initialize(self) -> None:
        """Initialize the Doctags interface and all components."""
        try:
            # Initialize VLM client manager
            self.vlm_client_manager = DoclingVLMClientManager(self.config.vlm_service_endpoints)
            await self.vlm_client_manager.initialize()

            # Initialize enhancement components
            if self.config.enable_medical_normalization:
                self.medical_normalizer = MedicalNormalizer(NormalizationLevel.STANDARD)

            if self.config.enable_table_fidelity:
                self.table_preserver = TableFidelityPreserver(
                    preserve_structure=True, include_captions=True
                )

            if self.config.enable_terminology_support:
                self.terminology_processor = MedicalTerminologyProcessor(
                    enable_synonym_expansion=True, enable_validation=True
                )

            logger.info("Doctags interface initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Doctags interface: {e}")
            raise

    async def close(self) -> None:
        """Close the Doctags interface and all components."""
        if self.vlm_client_manager:
            await self.vlm_client_manager.close()

        # Clear cache
        self._processing_cache.clear()
        self._cache_timestamps.clear()

    async def process_pdf(self, pdf_path: str) -> DoctagsResult:
        """Process PDF using Docling VLM with Doctags output.

        Args:
        ----
            pdf_path: Path to the PDF file

        Returns:
        -------
            DoctagsResult with processing results

        """
        start_time = time.time()

        try:
            # Check cache first
            if self.config.enable_caching:
                cached_result = self._get_cached_result(pdf_path)
                if cached_result:
                    logger.debug(f"Using cached result for {pdf_path}")
                    return cached_result

            # Validate PDF path
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Step 1: VLM processing
            vlm_result = await self._process_with_vlm(pdf_path)

            # Step 2: Apply enhancements
            enhanced_result = await self._apply_enhancements(vlm_result, pdf_path)

            # Step 3: Quality validation
            if self.config.enable_quality_validation:
                enhanced_result = self._validate_quality(enhanced_result)

            # Step 4: Update processing metadata
            enhanced_result.processing_time = time.time() - start_time
            enhanced_result.status = DoctagsProcessingStatus.COMPLETED

            # Step 5: Cache result
            if self.config.enable_caching:
                self._cache_result(pdf_path, enhanced_result)

            return enhanced_result

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return self._create_error_result(pdf_path, str(e), start_time)

    async def process_pdf_batch(self, pdf_paths: list[str]) -> list[DoctagsResult]:
        """Process multiple PDFs in batch.

        Args:
        ----
            pdf_paths: List of PDF file paths

        Returns:
        -------
            List of DoctagsResult objects

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
                    logger.error(f"Batch processing failed for {batch[j]}: {result}")
                    error_result = self._create_error_result(batch[j], str(result))
                    results.append(error_result)
                else:
                    results.append(result)

        return results

    async def _process_with_vlm(self, pdf_path: str) -> dict[str, Any]:
        """Process PDF with VLM service."""
        if not self.vlm_client_manager:
            raise Exception("VLM client manager not initialized")

        # Prepare VLM configuration
        vlm_config = {
            "model_name": self.config.vlm_model_name,
            "max_model_len": self.config.vlm_max_model_len,
            "temperature": self.config.vlm_temperature,
            "enable_table_extraction": self.config.enable_table_extraction,
            "enable_figure_extraction": self.config.enable_figure_extraction,
            "enable_text_extraction": self.config.enable_text_extraction,
        }

        # Prepare processing options
        processing_options = {
            "enable_medical_normalization": self.config.enable_medical_normalization,
            "enable_table_fidelity": self.config.enable_table_fidelity,
            "enable_terminology_support": self.config.enable_terminology_support,
            "min_confidence_threshold": self.config.min_confidence_threshold,
            "enable_quality_validation": self.config.enable_quality_validation,
            "timeout_seconds": self.config.timeout_seconds,
        }

        # Process with VLM
        doctags_result = await self.vlm_client_manager.process_pdf(
            pdf_path, vlm_config, processing_options
        )

        # Convert to dictionary for processing
        return self._convert_doctags_to_dict(doctags_result)

    def _convert_doctags_to_dict(self, doctags_result) -> dict[str, Any]:
        """Convert Doctags result to dictionary."""
        # This would convert the gRPC DoctagsResult to a dictionary
        raise NotImplementedError(
            "Doctags interface mock structure removed. "
            "This interface requires a real doctags service implementation. "
            "Please implement or configure a proper doctags service."
        )

    async def _apply_enhancements(self, vlm_result: dict[str, Any], pdf_path: str) -> DoctagsResult:
        """Apply enhancements to VLM result."""
        # Create base DoctagsResult
        result = DoctagsResult(
            document_id=vlm_result.get("document_id", f"doc_{int(time.time())}"),
            pdf_path=pdf_path,
            title=vlm_result.get("title"),
            sections=vlm_result.get("sections", []),
            pages=vlm_result.get("pages", []),
            tables=vlm_result.get("tables", []),
            figures=vlm_result.get("figures", []),
            text_blocks=vlm_result.get("text_blocks", []),
            document_type=vlm_result.get("document_type"),
            language=vlm_result.get("language"),
            author=vlm_result.get("author"),
            keywords=vlm_result.get("keywords", []),
            abstract=vlm_result.get("abstract"),
            model_version=vlm_result.get("model_version"),
            gpu_memory_used=vlm_result.get("gpu_memory_used"),
            processing_config=asdict(self.config),
        )

        # Apply medical normalization
        if self.config.enable_medical_normalization and self.medical_normalizer:
            try:
                normalized_content = await self._apply_medical_normalization(result)
                result.normalized_content = normalized_content
            except Exception as e:
                logger.warning(f"Medical normalization failed: {e}")

        # Apply table fidelity preservation
        if self.config.enable_table_fidelity and self.table_preserver:
            try:
                table_preservation = await self._apply_table_fidelity(result)
                result.table_preservation = table_preservation
            except Exception as e:
                logger.warning(f"Table fidelity preservation failed: {e}")

        # Apply terminology support
        if self.config.enable_terminology_support and self.terminology_processor:
            try:
                terminology_processing = await self._apply_terminology_support(result)
                result.terminology_processing = terminology_processing
            except Exception as e:
                logger.warning(f"Terminology processing failed: {e}")

        return result

    async def _apply_medical_normalization(self, result: DoctagsResult) -> dict[str, Any]:
        """Apply medical normalization to result."""
        normalized_content = {}

        # Normalize text blocks
        for i, text_block in enumerate(result.text_blocks):
            if "content" in text_block:
                normalized = self.medical_normalizer.normalize_text(text_block["content"])
                normalized_content[f"text_block_{i}"] = {
                    "original": text_block["content"],
                    "normalized": normalized.normalized_text,
                    "machine_text": normalized.machine_text,
                    "confidence_score": normalized.confidence_score,
                }

        # Normalize table content
        for i, table in enumerate(result.tables):
            if "content" in table:
                normalized = self.medical_normalizer.normalize_text(table["content"])
                normalized_content[f"table_{i}"] = {
                    "original": table["content"],
                    "normalized": normalized.normalized_text,
                    "machine_text": normalized.machine_text,
                    "confidence_score": normalized.confidence_score,
                }

        return normalized_content

    async def _apply_table_fidelity(self, result: DoctagsResult) -> dict[str, Any]:
        """Apply table fidelity preservation to result."""
        table_preservation = {}

        for i, table in enumerate(result.tables):
            try:
                preserved = self.table_preserver.preserve_table_structure(table)
                table_preservation[f"table_{i}"] = {
                    "chunk_id": preserved.chunk_id,
                    "flattened_content": preserved.flattened_content,
                    "contextualized_content": preserved.contextualized_content,
                    "machine_content": preserved.machine_content,
                    "preservation_metadata": preserved.preservation_metadata,
                }
            except Exception as e:
                logger.warning(f"Table preservation failed for table {i}: {e}")
                table_preservation[f"table_{i}"] = {"error": str(e)}

        return table_preservation

    async def _apply_terminology_support(self, result: DoctagsResult) -> dict[str, Any]:
        """Apply terminology support to result."""
        terminology_processing = {}

        # Process text blocks
        for i, text_block in enumerate(result.text_blocks):
            if "content" in text_block:
                processed = self.terminology_processor.process_text(
                    text_block["content"], TerminologyType.SYNONYM_EXPANSION
                )
                terminology_processing[f"text_block_{i}"] = {
                    "original": text_block["content"],
                    "processed": processed.processed_text,
                    "expanded_terms": processed.expanded_terms,
                    "confidence_score": processed.confidence_score,
                }

        return terminology_processing

    def _validate_quality(self, result: DoctagsResult) -> DoctagsResult:
        """Validate result quality."""
        quality_issues = []

        # Check for minimum content
        if not result.text_blocks and not result.tables and not result.figures:
            quality_issues.append(
                {
                    "issue_type": "insufficient_content",
                    "description": "No content extracted from document",
                    "severity": "high",
                }
            )

        # Check processing time (should be reasonable)
        if result.processing_time > 600:  # 10 minutes
            quality_issues.append(
                {
                    "issue_type": "slow_processing",
                    "description": f"Processing took {result.processing_time:.2f} seconds",
                    "severity": "medium",
                }
            )

        # Check confidence scores
        if result.confidence_score < self.config.min_confidence_threshold:
            quality_issues.append(
                {
                    "issue_type": "low_confidence",
                    "description": f"Confidence score {result.confidence_score} below threshold",
                    "severity": "medium",
                }
            )

        # Update result
        result.quality_issues = quality_issues
        result.completeness_score = self._calculate_completeness_score(result)
        result.accuracy_score = self._calculate_accuracy_score(result)

        return result

    def _calculate_completeness_score(self, result: DoctagsResult) -> float:
        """Calculate completeness score."""
        score = 1.0

        # Reduce score for missing essential elements
        if not result.title:
            score -= 0.1
        if not result.text_blocks:
            score -= 0.3
        if not result.sections:
            score -= 0.1

        return max(0.0, score)

    def _calculate_accuracy_score(self, result: DoctagsResult) -> float:
        """Calculate accuracy score."""
        score = 1.0

        # Reduce score for quality issues
        for issue in result.quality_issues:
            if issue["severity"] == "high":
                score -= 0.3
            elif issue["severity"] == "medium":
                score -= 0.1
            elif issue["severity"] == "low":
                score -= 0.05

        return max(0.0, score)

    def _create_error_result(
        self, pdf_path: str, error_message: str, start_time: float
    ) -> DoctagsResult:
        """Create error result."""
        return DoctagsResult(
            document_id=f"error_{int(time.time())}",
            pdf_path=pdf_path,
            processing_time=time.time() - start_time,
            status=DoctagsProcessingStatus.FAILED,
            error_message=error_message,
            processing_config=asdict(self.config),
            confidence_score=0.0,
            completeness_score=0.0,
            accuracy_score=0.0,
        )

    def _get_cached_result(self, pdf_path: str) -> DoctagsResult | None:
        """Get cached result if available and not expired."""
        if pdf_path not in self._processing_cache:
            return None

        # Check if cache is expired
        cache_time = self._cache_timestamps.get(pdf_path, 0)
        if time.time() - cache_time > self.config.cache_ttl:
            del self._processing_cache[pdf_path]
            del self._cache_timestamps[pdf_path]
            return None

        return self._processing_cache[pdf_path]

    def _cache_result(self, pdf_path: str, result: DoctagsResult) -> None:
        """Cache processing result."""
        self._processing_cache[pdf_path] = result
        self._cache_timestamps[pdf_path] = time.time()

    async def health_check(self) -> dict[str, Any]:
        """Check health of Doctags interface and all components.

        Returns
        -------
            Health status information

        """
        health_info = {
            "interface_status": "healthy",
            "config": asdict(self.config),
        }

        # Check VLM client manager
        if self.vlm_client_manager:
            try:
                vlm_health = await self.vlm_client_manager.health_check()
                health_info["vlm_client_manager"] = vlm_health
            except Exception as e:
                health_info["vlm_client_manager"] = {"status": "unhealthy", "error": str(e)}
                health_info["interface_status"] = "unhealthy"
        else:
            health_info["vlm_client_manager"] = {"status": "not_initialized"}
            health_info["interface_status"] = "unhealthy"

        # Check enhancement components
        health_info["enhancement_components"] = {
            "medical_normalizer": {"status": "healthy" if self.medical_normalizer else "disabled"},
            "table_preserver": {"status": "healthy" if self.table_preserver else "disabled"},
            "terminology_processor": {
                "status": "healthy" if self.terminology_processor else "disabled"
            },
        }

        # Cache statistics
        health_info["cache_stats"] = {
            "cached_results": len(self._processing_cache),
            "cache_ttl": self.config.cache_ttl,
        }

        return health_info

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        stats = {
            "interface_stats": {
                "cached_results": len(self._processing_cache),
                "cache_ttl": self.config.cache_ttl,
            },
        }

        # Add VLM client manager stats
        if self.vlm_client_manager:
            stats["vlm_client_manager_stats"] = {
                "total_clients": len(self.vlm_client_manager.clients),
                "current_client_index": self.vlm_client_manager.current_client_index,
            }

        return stats

    def clear_cache(self) -> None:
        """Clear processing cache."""
        self._processing_cache.clear()
        self._cache_timestamps.clear()


def create_doctags_interface(config: DoctagsConfig) -> DoctagsInterface:
    """Create Doctags interface instance.

    Args:
    ----
        config: Doctags processing configuration

    Returns:
    -------
        DoctagsInterface instance

    """
    return DoctagsInterface(config)


def create_default_doctags_interface(vlm_service_endpoints: list[str]) -> DoctagsInterface:
    """Create Doctags interface with default configuration.

    Args:
    ----
        vlm_service_endpoints: List of VLM service endpoints

    Returns:
    -------
        DoctagsInterface instance

    """
    config = DoctagsConfig(
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

    return DoctagsInterface(config)
