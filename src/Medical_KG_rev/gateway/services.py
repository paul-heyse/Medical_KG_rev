"""Gateway service layer."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from Medical_KG_rev.adapters.plugins.models import AdapterRequest
from Medical_KG_rev.config.settings import get_settings
from Medical_KG_rev.kg import ShaclValidator, ValidationError
from Medical_KG_rev.observability.metrics_migration import get_migration_helper
from Medical_KG_rev.orchestration import HaystackRetriever, JobLedger, JobLedgerEntry
from Medical_KG_rev.orchestration.dagster import (
    DagsterJobManager,
    DagsterResourceManager,
)
from Medical_KG_rev.orchestration.dagster.stages import create_default_pipeline_resource
from Medical_KG_rev.orchestration.stages.contracts import StageContext
from Medical_KG_rev.services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from Medical_KG_rev.services.embedding.persister import (
    EmbeddingPersister,
)
from Medical_KG_rev.services.embedding.policy import (
    EmbeddingPolicy,
)
from Medical_KG_rev.services.embedding.registry import EmbeddingModelRegistry
from Medical_KG_rev.services.embedding.telemetry import (
    EmbeddingTelemetry,
)
from Medical_KG_rev.services.extraction.templates import TemplateValidationError, validate_template
from Medical_KG_rev.services.retrieval.chunking import ChunkingService

from .coordinators.chunking import ChunkingCoordinator
from .coordinators.embedding import EmbeddingCoordinator
from .models import (
    EmbeddingRequest,
    EmbeddingResponse,
    ExtractionRequest,
    ExtractionResponse,
    IngestionRequest,
    IngestionResponse,
)

logger = structlog.get_logger(__name__)


@dataclass
class GatewayServiceConfig:
    """Configuration for gateway service."""

    chunking_service: ChunkingService
    embedding_namespace_registry: EmbeddingNamespaceRegistry
    embedding_model_registry: EmbeddingModelRegistry
    embedding_persister: EmbeddingPersister
    embedding_policy: EmbeddingPolicy
    embedding_telemetry: EmbeddingTelemetry
    dagster_job_manager: DagsterJobManager
    dagster_resource_manager: DagsterResourceManager
    haystack_retriever: HaystackRetriever
    job_ledger: JobLedger
    shacl_validator: ShaclValidator


class GatewayService:
    """Main gateway service for coordinating operations."""

    def __init__(self, config: GatewayServiceConfig) -> None:
        """Initialize the gateway service."""
        self.config = config
        self.settings = get_settings()

        # Initialize coordinators
        self.chunking_coordinator = ChunkingCoordinator(
            chunking_service=config.chunking_service,
            error_translator=None,  # Will be initialized later
        )

        self.embedding_coordinator = EmbeddingCoordinator(
            config=EmbeddingCoordinatorConfig(
                name="gateway_embedding",
                namespace_registry=config.embedding_namespace_registry,
                model_registry=config.embedding_model_registry,
                persister=config.embedding_persister,
                policy=config.embedding_policy,
                telemetry=config.embedding_telemetry,
                job_lifecycle=None,  # Will be initialized later
            )
        )

    async def embed_text(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed text using the specified model and namespace."""
        start_time = time.time()

        try:
            # Validate request
            self._validate_embedding_request(request)

            # Process embedding
            response = await self.embedding_coordinator.embed_text(request)

            # Record business event using domain-specific metrics
            settings = get_settings()
            helper = get_migration_helper(settings)
            helper.record_external_api_call(
                "POST", "/api/v1/embed", "200"
            )

            return response

        except Exception as exc:
            processing_time = time.time() - start_time

            logger.error(
                "embedding.failed",
                tenant_id=request.tenant_id,
                namespace=request.namespace,
                model=request.model,
                error=str(exc),
                processing_time=processing_time,
            )

            raise exc

    async def extract_entities(self, request: ExtractionRequest) -> ExtractionResponse:
        """Extract entities from text."""
        start_time = time.time()

        try:
            # Validate request
            self._validate_extraction_request(request)

            # Process extraction
            response = await self._process_extraction(request)

            # Record business event using domain-specific metrics
            settings = get_settings()
            helper = get_migration_helper(settings)
            helper.record_external_api_call(
                "POST", "/api/v1/extract", "200"
            )

            return response

        except Exception as exc:
            processing_time = time.time() - start_time

            logger.error(
                "extraction.failed",
                tenant_id=request.tenant_id,
                document_id=request.document_id,
                error=str(exc),
                processing_time=processing_time,
            )

            raise exc

    async def ingest_document(self, request: IngestionRequest) -> IngestionResponse:
        """Ingest a document."""
        start_time = time.time()

        try:
            # Validate request
            self._validate_ingestion_request(request)

            # Process ingestion
            response = await self._process_ingestion(request)

            # Record business event using domain-specific metrics
            settings = get_settings()
            helper = get_migration_helper(settings)
            helper.record_external_api_call(
                "POST", "/api/v1/ingest", "200"
            )

            return response

        except Exception as exc:
            processing_time = time.time() - start_time

            logger.error(
                "ingestion.failed",
                tenant_id=request.tenant_id,
                document_id=request.document_id,
                error=str(exc),
                processing_time=processing_time,
            )

            raise exc

    def _validate_embedding_request(self, request: EmbeddingRequest) -> None:
        """Validate an embedding request."""
        if not request.tenant_id:
            raise ValueError("tenant_id is required")

        if not request.namespace:
            raise ValueError("namespace is required")

        if not request.model:
            raise ValueError("model is required")

        if not request.texts:
            raise ValueError("texts are required")

    def _validate_extraction_request(self, request: ExtractionRequest) -> None:
        """Validate an extraction request."""
        if not request.tenant_id:
            raise ValueError("tenant_id is required")

        if not request.document_id:
            raise ValueError("document_id is required")

        if not request.content:
            raise ValueError("content is required")

        if not request.extraction_type:
            raise ValueError("extraction_type is required")

    def _validate_ingestion_request(self, request: IngestionRequest) -> None:
        """Validate an ingestion request."""
        if not request.tenant_id:
            raise ValueError("tenant_id is required")

        if not request.document_id:
            raise ValueError("document_id is required")

        if not request.content:
            raise ValueError("content is required")

        if not request.content_type:
            raise ValueError("content_type is required")

    async def _process_extraction(self, request: ExtractionRequest) -> ExtractionResponse:
        """Process extraction request."""
        # Mock implementation - would typically call extraction service
        from .models import Entity

        entities = [
            Entity(
                id=f"entity-{i}",
                type="PERSON",
                text=f"Entity {i}",
                confidence=0.9,
                metadata={"start": i * 10, "end": (i + 1) * 10},
            )
            for i in range(3)
        ]

        return ExtractionResponse(
            entities=entities,
            processing_time=0.1,
            extraction_type=request.extraction_type,
        )

    async def _process_ingestion(self, request: IngestionRequest) -> IngestionResponse:
        """Process ingestion request."""
        # Mock implementation - would typically call ingestion service
        return IngestionResponse(
            document_id=request.document_id,
            status="completed",
            processing_time=0.2,
            metadata={"content_type": request.content_type},
        )

    def get_available_models(self) -> list[str]:
        """Get available embedding models."""
        return self.embedding_coordinator.get_available_models()

    def get_available_namespaces(self, tenant_id: str) -> list[str]:
        """Get available namespaces for a tenant."""
        return self.embedding_coordinator.get_available_namespaces(tenant_id)

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get information about an embedding model."""
        return self.embedding_coordinator.get_model_info(model)

    def get_namespace_info(self, tenant_id: str, namespace: str) -> dict[str, Any]:
        """Get information about a namespace."""
        return self.embedding_coordinator.get_namespace_info(tenant_id, namespace)


# Global service instance
_gateway_service: GatewayService | None = None


def get_gateway_service() -> GatewayService:
    """Get the global gateway service instance."""
    global _gateway_service

    if _gateway_service is None:
        # Create mock configuration for now
        config = GatewayServiceConfig(
            chunking_service=None,  # Will be initialized
            embedding_namespace_registry=None,  # Will be initialized
            embedding_model_registry=None,  # Will be initialized
            embedding_persister=None,  # Will be initialized
            embedding_policy=None,  # Will be initialized
            embedding_telemetry=None,  # Will be initialized
            dagster_job_manager=None,  # Will be initialized
            dagster_resource_manager=None,  # Will be initialized
            haystack_retriever=None,  # Will be initialized
            job_ledger=None,  # Will be initialized
            shacl_validator=None,  # Will be initialized
        )

        _gateway_service = GatewayService(config)

    return _gateway_service
