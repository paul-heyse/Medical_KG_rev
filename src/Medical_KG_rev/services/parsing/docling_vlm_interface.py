"""Docling VLM processing interface for biomedical document processing.

This module provides a processing interface compatible with existing pipeline including:
- Ensure DoclingVLMService implements consistent processing interface
- Return DoclingVLMResult with consistent structure
- Include text, tables, figures, and metadata extraction
- Maintain backward compatibility with existing document formats
- Add provenance tracking for VLM processing (model_version, processing_time)
"""

import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .docling_vlm_service import (
    DoclingVLMConfig,
    DoclingVLMResult,
    DoclingVLMService,
)
from .medical_normalization import MedicalNormalizer, NormalizationLevel
from .medical_terminology import MedicalTerminologyProcessor, TerminologyType
from .table_fidelity import TableFidelityPreserver

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for VLM interface."""

    VLM_ONLY = "vlm_only"  # Use only VLM processing
    VLM_WITH_NORMALIZATION = "vlm_with_normalization"  # VLM + medical normalization
    VLM_WITH_TABLE_FIDELITY = "vlm_with_table_fidelity"  # VLM + table preservation
    VLM_FULL_PIPELINE = "vlm_full_pipeline"  # VLM + all enhancements


@dataclass
class ProcessingCompatibleResult:
    """Result compatible with processing interface."""

    # Core content
    text_content: str
    tables: list[dict[str, Any]]
    figures: list[dict[str, Any]]
    metadata: dict[str, Any]

    # Processing information
    processing_time: float
    status: str
    error_message: str | None = None

    # VLM-specific information
    model_version: str | None = None
    gpu_memory_used: float | None = None

    # Provenance tracking
    processing_method: str = "docling_vlm"
    processing_config: dict[str, Any] = None

    # Enhanced processing results
    normalized_text: str | None = None
    table_chunks: list[dict[str, Any]] = None
    terminology_applied: list[str] = None


@dataclass
class VLMProcessingConfig:
    """Configuration for VLM processing interface."""

    # VLM service configuration
    vlm_config: DoclingVLMConfig

    # Processing mode
    processing_mode: ProcessingMode = ProcessingMode.VLM_FULL_PIPELINE

    # Enhancement options
    enable_medical_normalization: bool = True
    enable_table_fidelity: bool = True
    enable_terminology_support: bool = True

    # Quality settings
    min_confidence_threshold: float = 0.7
    enable_quality_validation: bool = True

    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour


class DoclingVLMInterface:
    """Docling VLM processing interface compatible with existing pipeline.

    Handles:
    - Ensure DoclingVLMService implements consistent processing interface
    - Return DoclingVLMResult with consistent structure
    - Include text, tables, figures, and metadata extraction
    - Maintain backward compatibility with existing document formats
    - Add provenance tracking for VLM processing (model_version, processing_time)
    """

    def __init__(self, config: VLMProcessingConfig):
        """Initialize the Docling VLM interface.

        Args:
            config: Configuration for the VLM interface

        """
        self.config = config
        self.vlm_service: DoclingVLMService | None = None
        self.medical_normalizer: MedicalNormalizer | None = None
        self.table_preserver: TableFidelityPreserver | None = None
        self.terminology_processor: MedicalTerminologyProcessor | None = None

        # Processing cache
        self._processing_cache: dict[str, ProcessingCompatibleResult] = {}
        self._cache_timestamps: dict[str, float] = {}

    async def initialize(self) -> None:
        """Initialize the VLM interface and all components."""
        try:
            # Initialize VLM service
            self.vlm_service = DoclingVLMService(self.config.vlm_config)
            await self.vlm_service.initialize()

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

            logger.info("Docling VLM interface initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Docling VLM interface: {e}")
            raise

    async def close(self) -> None:
        """Close the VLM interface and all components."""
        if self.vlm_service:
            await self.vlm_service.close()

        # Clear cache
        self._processing_cache.clear()
        self._cache_timestamps.clear()

    async def process_pdf(self, pdf_path: str) -> ProcessingCompatibleResult:
        """Process PDF using Docling VLM (compatible with processing interface).

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ProcessingCompatibleResult with processing results

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

            # Step 2: Apply enhancements based on processing mode
            enhanced_result = await self._apply_enhancements(vlm_result, pdf_path)

            # Step 3: Create processing-compatible result
            processing_result = self._create_processing_compatible_result(
                enhanced_result, pdf_path, start_time
            )

            # Step 4: Quality validation
            if self.config.enable_quality_validation:
                if not self._validate_result_quality(processing_result):
                    logger.warning(f"Quality validation failed for {pdf_path}")
                    processing_result.status = "warning"

            # Step 5: Cache result
            if self.config.enable_caching:
                self._cache_result(pdf_path, processing_result)

            return processing_result

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return self._create_error_result(pdf_path, str(e), start_time)

    async def _process_with_vlm(self, pdf_path: str) -> DoclingVLMResult:
        """Process PDF with VLM service."""
        if not self.vlm_service:
            raise Exception("VLM service not initialized")

        return await self.vlm_service.process_pdf(pdf_path)

    async def _apply_enhancements(
        self, vlm_result: DoclingVLMResult, pdf_path: str
    ) -> dict[str, Any]:
        """Apply enhancements to VLM result."""
        enhanced_data = {
            "text_content": vlm_result.text_content,
            "tables": vlm_result.tables,
            "figures": vlm_result.figures,
            "metadata": vlm_result.metadata,
            "normalized_text": None,
            "table_chunks": [],
            "terminology_applied": [],
        }

        # Apply medical normalization
        if self.config.enable_medical_normalization and self.medical_normalizer:
            try:
                normalization_result = self.medical_normalizer.normalize_text(
                    vlm_result.text_content
                )
                enhanced_data["normalized_text"] = normalization_result.normalized_text
                enhanced_data["terminology_applied"].append("medical_normalization")
            except Exception as e:
                logger.warning(f"Medical normalization failed: {e}")

        # Apply table fidelity preservation
        if self.config.enable_table_fidelity and self.table_preserver:
            try:
                table_chunks = []
                for table in vlm_result.tables:
                    table_chunk = self.table_preserver.preserve_table_structure(table)
                    table_chunks.append(asdict(table_chunk))
                enhanced_data["table_chunks"] = table_chunks
                enhanced_data["terminology_applied"].append("table_fidelity")
            except Exception as e:
                logger.warning(f"Table fidelity preservation failed: {e}")

        # Apply terminology support
        if self.config.enable_terminology_support and self.terminology_processor:
            try:
                terminology_result = self.terminology_processor.process_text(
                    vlm_result.text_content, TerminologyType.SYNONYM_EXPANSION
                )
                enhanced_data["terminology_applied"].extend(terminology_result.terminology_applied)
            except Exception as e:
                logger.warning(f"Terminology processing failed: {e}")

        return enhanced_data

    def _create_processing_compatible_result(
        self, enhanced_data: dict[str, Any], pdf_path: str, start_time: float
    ) -> ProcessingCompatibleResult:
        """Create processing-compatible result from enhanced data."""
        return ProcessingCompatibleResult(
            text_content=enhanced_data["text_content"],
            tables=enhanced_data["tables"],
            figures=enhanced_data["figures"],
            metadata=enhanced_data["metadata"],
            processing_time=time.time() - start_time,
            status="completed",
            model_version=enhanced_data["metadata"].get("model_version"),
            gpu_memory_used=enhanced_data["metadata"].get("gpu_memory_used"),
            processing_method="docling_vlm",
            processing_config=asdict(self.config),
            normalized_text=enhanced_data["normalized_text"],
            table_chunks=enhanced_data["table_chunks"],
            terminology_applied=enhanced_data["terminology_applied"],
        )

    def _create_error_result(
        self, pdf_path: str, error_message: str, start_time: float
    ) -> ProcessingCompatibleResult:
        """Create error result."""
        return ProcessingCompatibleResult(
            text_content="",
            tables=[],
            figures=[],
            metadata={},
            processing_time=time.time() - start_time,
            status="failed",
            error_message=error_message,
            processing_method="docling_vlm",
            processing_config=asdict(self.config),
        )

    def _validate_result_quality(self, result: ProcessingCompatibleResult) -> bool:
        """Validate result quality."""
        # Check for minimum content
        if not result.text_content.strip():
            return False

        # Check processing time (should be reasonable)
        if result.processing_time > 600:  # 10 minutes
            return False

        # Check confidence if available
        if result.metadata.get("confidence_score", 1.0) < self.config.min_confidence_threshold:
            return False

        return True

    def _get_cached_result(self, pdf_path: str) -> ProcessingCompatibleResult | None:
        """Get cached result if available and not expired."""
        if pdf_path not in self._processing_cache:
            return None

        # Check if cache is expired
        cache_time = self._cache_timestamps.get(pdf_path, 0)
        if time.time() - cache_time > self.config.cache_ttl:
            # Remove expired cache entry
            del self._processing_cache[pdf_path]
            del self._cache_timestamps[pdf_path]
            return None

        return self._processing_cache[pdf_path]

    def _cache_result(self, pdf_path: str, result: ProcessingCompatibleResult) -> None:
        """Cache processing result."""
        self._processing_cache[pdf_path] = result
        self._cache_timestamps[pdf_path] = time.time()

    async def health_check(self) -> dict[str, Any]:
        """Check health of VLM interface and all components.

        Returns:
            Health status information

        """
        health_info = {
            "interface_status": "healthy",
            "components": {},
            "config": asdict(self.config),
        }

        # Check VLM service
        if self.vlm_service:
            try:
                vlm_health = await self.vlm_service.health_check()
                health_info["components"]["vlm_service"] = vlm_health
            except Exception as e:
                health_info["components"]["vlm_service"] = {"status": "unhealthy", "error": str(e)}
                health_info["interface_status"] = "unhealthy"
        else:
            health_info["components"]["vlm_service"] = {"status": "not_initialized"}
            health_info["interface_status"] = "unhealthy"

        # Check enhancement components
        health_info["components"]["medical_normalizer"] = {
            "status": "healthy" if self.medical_normalizer else "disabled"
        }
        health_info["components"]["table_preserver"] = {
            "status": "healthy" if self.table_preserver else "disabled"
        }
        health_info["components"]["terminology_processor"] = {
            "status": "healthy" if self.terminology_processor else "disabled"
        }

        # Cache statistics
        health_info["cache_stats"] = {
            "cached_items": len(self._processing_cache),
            "cache_enabled": self.config.enable_caching,
            "cache_ttl": self.config.cache_ttl,
        }

        return health_info

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        stats = {
            "config": asdict(self.config),
            "cache_stats": {
                "cached_items": len(self._processing_cache),
                "cache_enabled": self.config.enable_caching,
                "cache_ttl": self.config.cache_ttl,
            },
        }

        # Add VLM service stats
        if self.vlm_service:
            stats["vlm_service_stats"] = self.vlm_service.get_stats()

        # Add enhancement component stats
        if self.medical_normalizer:
            stats["medical_normalizer_stats"] = self.medical_normalizer.get_normalization_stats()

        if self.table_preserver:
            stats["table_preserver_stats"] = self.table_preserver.get_preservation_stats()

        if self.terminology_processor:
            stats["terminology_processor_stats"] = (
                self.terminology_processor.get_terminology_stats()
            )

        return stats

    def clear_cache(self) -> None:
        """Clear processing cache."""
        self._processing_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Processing cache cleared")


def create_docling_vlm_interface(config: VLMProcessingConfig) -> DoclingVLMInterface:
    """Create Docling VLM interface instance.

    Args:
        config: VLM processing configuration

    Returns:
        DoclingVLMInterface instance

    """
    return DoclingVLMInterface(config)


def create_processing_compatible_interface(
    vlm_config: DoclingVLMConfig, processing_mode: ProcessingMode = ProcessingMode.VLM_FULL_PIPELINE
) -> DoclingVLMInterface:
    """Create processing-compatible VLM interface.

    Args:
        vlm_config: VLM service configuration
        processing_mode: Processing mode

    Returns:
        DoclingVLMInterface instance

    """
    config = VLMProcessingConfig(
        vlm_config=vlm_config,
        processing_mode=processing_mode,
        enable_medical_normalization=True,
        enable_table_fidelity=True,
        enable_terminology_support=True,
        min_confidence_threshold=0.7,
        enable_quality_validation=True,
        enable_caching=True,
        cache_ttl=3600,
    )

    return DoclingVLMInterface(config)
