"""Convert stage for pipeline processing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests
import structlog
from pydantic import BaseModel, Field

from Medical_KG_rev.pipeline.stages import ConvertStage, StageResult, StageStatus
from Medical_KG_rev.services.parsing.docling_vlm_service import (
    DoclingVLMService,
)

logger = structlog.get_logger(__name__)


@dataclass
class ConvertStageConfig:
    """Configuration for convert stage."""

    docling_service_url: str = "http://localhost:8000"
    timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0
    max_file_size: int = 50 * 1024 * 1024  # 50MB


class ConvertStageImpl(ConvertStage):
    """Implementation of convert stage."""

    def __init__(self, config: ConvertStageConfig) -> None:
        """Initialize the convert stage."""
        super().__init__()
        self.config = config
        self.logger = logger
        self.docling_service = DoclingVLMService()

    def execute(self, context: dict[str, Any]) -> StageResult:
        """Execute the convert stage."""
        try:
            self.logger.info("Executing convert stage")

            # Get documents from context
            documents = context.get("documents", [])
            if not documents:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No documents found in context",
                    data=context,
                )

            # Process documents
            converted_documents = []
            for doc in documents:
                converted_doc = self._convert_document(doc)
                if converted_doc:
                    converted_documents.append(converted_doc)

            # Update context
            context["converted_documents"] = converted_documents
            context["conversion_count"] = len(converted_documents)

            return StageResult(
                status=StageStatus.COMPLETED,
                data=context,
                metadata={
                    "converted_count": len(converted_documents),
                    "total_count": len(documents),
                    "service_url": self.config.docling_service_url,
                },
            )

        except Exception as exc:
            self.logger.error(f"Convert stage failed: {exc}")
            return StageResult(
                status=StageStatus.FAILED,
                error=str(exc),
                data=context,
            )

    def _convert_document(self, document: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a document."""
        try:
            # Check file size
            content = document.get("content", "")
            if len(content) > self.config.max_file_size:
                self.logger.warning(f"Document too large: {len(content)} bytes")
                return None

            # Convert document using Docling VLM service
            converted_doc = self.docling_service.process_document(document)

            # Add conversion metadata
            converted_doc["conversion_metadata"] = {
                "converted_at": time.time(),
                "service_url": self.config.docling_service_url,
                "original_size": len(content),
                "converted_size": len(str(converted_doc)),
            }

            return converted_doc

        except Exception as exc:
            self.logger.error(f"Failed to convert document: {exc}")
            return None

    def health_check(self) -> dict[str, Any]:
        """Check stage health."""
        health = {
            "stage": "convert",
            "config": {
                "docling_service_url": self.config.docling_service_url,
                "timeout": self.config.timeout,
                "retry_count": self.config.retry_count,
                "retry_delay": self.config.retry_delay,
                "max_file_size": self.config.max_file_size,
            },
        }

        # Check Docling service health
        try:
            response = requests.get(f"{self.config.docling_service_url}/health", timeout=5)
            health["docling_service_status"] = (
                "healthy" if response.status_code == 200 else "unhealthy"
            )
        except Exception as e:
            health["docling_service_status"] = "unhealthy"
            health["docling_service_error"] = str(e)

        return health

    def get_config(self) -> ConvertStageConfig:
        """Get stage configuration."""
        return self.config

    def update_config(self, config: ConvertStageConfig) -> None:
        """Update stage configuration."""
        self.config = config
        self.logger.info("Convert stage configuration updated")


def create_convert_stage(config: ConvertStageConfig) -> ConvertStageImpl:
    """Create a convert stage instance."""
    return ConvertStageImpl(config)


def create_default_convert_stage_config() -> ConvertStageConfig:
    """Create default convert stage configuration."""
    return ConvertStageConfig(
        docling_service_url="http://localhost:8000",
        timeout=30,
        retry_count=3,
        retry_delay=1.0,
        max_file_size=50 * 1024 * 1024,
    )
