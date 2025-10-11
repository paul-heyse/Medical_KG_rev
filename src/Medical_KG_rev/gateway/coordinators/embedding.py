"""Embedding coordinator for gateway operations."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from Medical_KG_rev.embeddings.ports import EmbeddingRecord
from Medical_KG_rev.embeddings.ports import EmbeddingRequest as AdapterEmbeddingRequest
from Medical_KG_rev.gateway.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
)
from Medical_KG_rev.services.embedding.namespace.registry import (
    EmbeddingNamespaceRegistry,
)
from Medical_KG_rev.services.embedding.persister import (
    EmbeddingPersister,
)
from Medical_KG_rev.services.embedding.policy import (
    EmbeddingPolicy,
)
from Medical_KG_rev.services.embedding.registry import EmbeddingModelRegistry
from Medical_KG_rev.services.embedding.telemetry import EmbeddingTelemetry

from .base import CoordinatorMetrics
from .job_lifecycle import JobLifecycleManager

logger = structlog.get_logger(__name__)


@dataclass
class EmbeddingCoordinatorConfig:
    """Configuration for embedding coordinator."""

    name: str
    namespace_registry: EmbeddingNamespaceRegistry
    model_registry: EmbeddingModelRegistry
    persister: EmbeddingPersister
    policy: EmbeddingPolicy
    telemetry: EmbeddingTelemetry
    job_lifecycle: JobLifecycleManager


class EmbeddingCoordinator:
    """Coordinates embedding operations for the gateway."""

    def __init__(self, config: EmbeddingCoordinatorConfig) -> None:
        """Initialize the embedding coordinator."""
        self.config = config
        self.metrics = CoordinatorMetrics.create(config.name)
        self.namespace_registry = config.namespace_registry
        self.model_registry = config.model_registry
        self.persister = config.persister
        self.policy = config.policy
        self.telemetry = config.telemetry
        self.job_lifecycle = config.job_lifecycle

    async def embed_text(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Embed text using the specified model and namespace."""
        start_time = time.time()

        try:
            # Validate request
            self._validate_request(request)

            # Get namespace configuration
            namespace_config = await self.namespace_registry.get_namespace(
                request.tenant_id, request.namespace
            )

            # Get model configuration
            model_config = self.model_registry.get_model(request.model)

            # Create adapter request
            adapter_request = AdapterEmbeddingRequest(
                texts=request.texts,
                model=request.model,
                namespace=request.namespace,
                metadata=request.metadata or {},
            )

            # Execute embedding
            embedding_record = await self._execute_embedding(
                adapter_request, namespace_config, model_config
            )

            # Persist embeddings
            await self.persister.persist_embeddings(embedding_record)

            # Record telemetry
            self.telemetry.record_embedding(
                tenant_id=request.tenant_id,
                namespace=request.namespace,
                model=request.model,
                vector_count=len(embedding_record.vectors),
                processing_time=time.time() - start_time,
            )

            # Convert to response
            response = EmbeddingResponse(
                vectors=[
                    EmbeddingVector(
                        id=vector.id,
                        model=vector.model,
                        kind=vector.kind,
                        values=vector.values,
                        metadata=vector.metadata,
                    )
                    for vector in embedding_record.vectors
                ],
                processing_time=time.time() - start_time,
                model_used=request.model,
                namespace_used=request.namespace,
            )

            logger.info(
                "embedding.completed",
                tenant_id=request.tenant_id,
                namespace=request.namespace,
                model=request.model,
                vector_count=len(response.vectors),
                processing_time=response.processing_time,
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

    async def embed_multiple_texts(
        self,
        requests: Sequence[EmbeddingRequest],
    ) -> list[EmbeddingResponse]:
        """Embed multiple texts."""
        responses = []

        for request in requests:
            try:
                response = await self.embed_text(request)
                responses.append(response)
            except Exception as exc:
                logger.error(
                    "embedding.batch_failed",
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model=request.model,
                    error=str(exc),
                )
                # Continue with other requests
                continue

        return responses

    def _validate_request(self, request: EmbeddingRequest) -> None:
        """Validate an embedding request."""
        if not request.tenant_id:
            raise ValueError("tenant_id is required")

        if not request.namespace:
            raise ValueError("namespace is required")

        if not request.model:
            raise ValueError("model is required")

        if not request.texts:
            raise ValueError("texts are required")

    async def _execute_embedding(
        self,
        request: AdapterEmbeddingRequest,
        namespace_config: Any,
        model_config: Any,
    ) -> EmbeddingRecord:
        """Execute the embedding operation."""
        # This would typically call the embedding service
        raise NotImplementedError(
            "Embedding coordinator mock response removed. "
            "This coordinator requires a real embedding service implementation. "
            "Please implement or configure a proper embedding service."
        )

    def get_available_models(self) -> list[str]:
        """Get available embedding models."""
        return self.model_registry.list_models()

    def get_available_namespaces(self, tenant_id: str) -> list[str]:
        """Get available namespaces for a tenant."""
        return self.namespace_registry.list_namespaces(tenant_id)

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get information about an embedding model."""
        return self.model_registry.get_model_info(model)

    def get_namespace_info(self, tenant_id: str, namespace: str) -> dict[str, Any]:
        """Get information about a namespace."""
        return self.namespace_registry.get_namespace_info(tenant_id, namespace)
