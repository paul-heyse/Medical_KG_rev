"""Embedding coordinator for synchronous text embedding operations.

This module provides the EmbeddingCoordinator class that coordinates synchronous
embedding operations by managing job lifecycle, namespace access policies,
model selection, persistence, and telemetry.

Key Responsibilities:
    - Namespace access policy evaluation and enforcement
    - Job lifecycle management (create, track, complete/fail jobs)
    - Model selection and configuration from namespace registry
    - Text validation and preprocessing
    - Embedding generation via model registry
    - Persistence of embeddings via persister
    - Telemetry emission for embedding operations
    - Error handling and translation

Collaborators:
    - Upstream: Gateway service layer (calls execute method)
    - Downstream: EmbeddingModelRegistry (model selection), EmbeddingNamespaceRegistry (namespace config),
      NamespaceAccessPolicy (access control), EmbeddingPersister (storage), EmbeddingTelemetry (metrics)

Side Effects:
    - Creates job entries in job lifecycle manager
    - Persists embeddings to storage backend
    - Emits telemetry metrics for embedding operations
    - Logs errors and operations
    - Updates job status (completed/failed)

Thread Safety:
    - Not thread-safe: Designed for single-threaded use per coordinator instance
    - Multiple coordinator instances can run concurrently

Performance Characteristics:
    - O(n) time complexity where n is number of texts to embed
    - Memory usage scales with embedding dimensions and batch size
    - Synchronous operation blocks until embedding completes
    - GPU utilization depends on model and batch configuration

Example:
    >>> from Medical_KG_rev.gateway.coordinators import EmbeddingCoordinator
    >>> coordinator = EmbeddingCoordinator(
    ...     lifecycle=JobLifecycleManager(),
    ...     registry=EmbeddingModelRegistry(),
    ...     namespace_registry=EmbeddingNamespaceRegistry(),
    ...     policy=NamespaceAccessPolicy(),
    ...     persister=EmbeddingPersister(),
    ...     telemetry=EmbeddingTelemetry(),
    ...     config=CoordinatorConfig(name="embedding")
    ... )
    >>> result = coordinator.execute(EmbeddingRequest(
    ...     namespace="medical",
    ...     texts=["Sample text to embed"],
    ...     options=EmbeddingOptions(model="sentence-transformers/all-MiniLM-L6-v2")
    ... ))
    >>> print(f"Generated {len(result.response.vectors)} embeddings")

"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from Medical_KG_rev.auth.scopes import Scopes
from Medical_KG_rev.embeddings.ports import EmbeddingRecord
from Medical_KG_rev.embeddings.ports import EmbeddingRequest as AdapterEmbeddingRequest
from Medical_KG_rev.gateway.models import (
    EmbeddingMetadata,
    EmbeddingOptions,
    EmbeddingResponse,
    EmbeddingVector,
    ProblemDetail,
)
from Medical_KG_rev.services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from Medical_KG_rev.services.embedding.persister import EmbeddingPersister, PersistenceContext
from Medical_KG_rev.services.embedding.policy import NamespaceAccessDecision, NamespaceAccessPolicy
from Medical_KG_rev.services.embedding.registry import EmbeddingModelRegistry
from Medical_KG_rev.services.embedding.telemetry import EmbeddingTelemetry

from .base import (
    BaseCoordinator,
    CoordinatorConfig,
    CoordinatorError,
    CoordinatorRequest,
    CoordinatorResult,
)
from .job_lifecycle import JobLifecycleManager

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class EmbeddingRequest(CoordinatorRequest):
    """Request for synchronous text embedding operations.

    Attributes:
        namespace: Namespace identifier for embedding configuration and access control.
        texts: Sequence of text strings to be embedded. All texts must be non-empty.
        options: Optional embedding configuration including model selection, batch size,
                and other embedding parameters.

    """

    def __init__(
        self,
        tenant_id: str,
        namespace: str,
        texts: Sequence[str],
        *,
        options: EmbeddingOptions | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize embedding request.

        Args:
            tenant_id: Tenant identifier for multi-tenancy.
            namespace: Namespace identifier for embedding configuration.
            texts: Sequence of text strings to be embedded.
            options: Optional embedding configuration.
            correlation_id: Optional correlation ID for request tracking.
            metadata: Optional metadata for request context.

        """
        super().__init__(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )
        self.namespace = namespace
        self.texts = texts
        self.options = options


@dataclass
class EmbeddingResult(CoordinatorResult):
    """Result of synchronous text embedding operations.

    Attributes:
        response: EmbeddingResponse containing generated vectors and metadata.
                 None if embedding operation failed.

    """

    response: EmbeddingResponse | None = None


# ============================================================================
# COORDINATOR IMPLEMENTATION
# ============================================================================


class EmbeddingCoordinator(BaseCoordinator[EmbeddingRequest, EmbeddingResult]):
    """Coordinates synchronous text embedding operations.

    This class implements the coordinator pattern for text embedding, managing
    the complete lifecycle of embedding jobs from namespace access control through
    model selection, embedding generation, persistence, and telemetry.

    The coordinator coordinates between the gateway service layer and the domain
    embedding services, providing a clean abstraction for synchronous embedding
    operations with comprehensive access control, persistence, and metrics.

    Attributes:
        _lifecycle: JobLifecycleManager for tracking job state and metadata.
        _registry: EmbeddingModelRegistry for model selection and configuration.
        _namespace_registry: EmbeddingNamespaceRegistry for namespace configuration.
        _policy: NamespaceAccessPolicy for access control evaluation.
        _persister: EmbeddingPersister for embedding storage.
        _telemetry: EmbeddingTelemetry for metrics emission (optional).

    Invariants:
        - self._lifecycle is never None after __init__
        - self._registry is never None after __init__
        - self._namespace_registry is never None after __init__
        - self._policy is never None after __init__
        - self._persister is never None after __init__
        - All public methods maintain job lifecycle consistency

    Thread Safety:
        - Not thread-safe: Designed for single-threaded use per coordinator instance
        - Multiple coordinator instances can run concurrently

    Lifecycle:
        - Created with injected dependencies (lifecycle, registry, policy, etc.)
        - Used for processing embedding requests via execute method
        - No explicit cleanup required (stateless operations)

    Example:
        >>> coordinator = EmbeddingCoordinator(
        ...     lifecycle=JobLifecycleManager(),
        ...     registry=EmbeddingModelRegistry(),
        ...     namespace_registry=EmbeddingNamespaceRegistry(),
        ...     policy=NamespaceAccessPolicy(),
        ...     persister=EmbeddingPersister(),
        ...     telemetry=EmbeddingTelemetry(),
        ...     config=CoordinatorConfig(name="embedding")
        ... )
        >>> result = coordinator.execute(EmbeddingRequest(
        ...     tenant_id="tenant1",
        ...     namespace="medical",
        ...     texts=["Sample text to embed"],
        ...     options=EmbeddingOptions(model="sentence-transformers/all-MiniLM-L6-v2")
        ... ))
        >>> print(f"Generated {len(result.response.vectors)} embeddings")

    """

    def __init__(
        self,
        lifecycle: JobLifecycleManager,
        registry: EmbeddingModelRegistry,
        namespace_registry: EmbeddingNamespaceRegistry,
        policy: NamespaceAccessPolicy,
        persister: EmbeddingPersister,
        telemetry: EmbeddingTelemetry | None,
        config: CoordinatorConfig,
    ) -> None:
        """Initialize the embedding coordinator with required dependencies.

        Args:
            lifecycle: Manager for tracking job lifecycle and metadata.
            registry: Registry for embedding model selection and configuration.
            namespace_registry: Registry for namespace-specific configuration.
            policy: Policy engine for namespace access control evaluation.
            persister: Service for persisting generated embeddings.
            telemetry: Optional telemetry service for metrics emission.
            config: Coordinator configuration settings.

        Raises:
            ValueError: If any required dependency is None.

        Example:
            >>> coordinator = EmbeddingCoordinator(
            ...     lifecycle=JobLifecycleManager(),
            ...     registry=EmbeddingModelRegistry(),
            ...     namespace_registry=EmbeddingNamespaceRegistry(),
            ...     policy=NamespaceAccessPolicy(),
            ...     persister=EmbeddingPersister(),
            ...     telemetry=EmbeddingTelemetry(),
            ...     config=CoordinatorConfig(name="embedding")
            ... )

        """
        from .base import CoordinatorMetrics

        super().__init__(config=config, metrics=CoordinatorMetrics.create(config.name))
        self._lifecycle = lifecycle
        self._registry = registry
        self._namespace_registry = namespace_registry
        self._policy = policy
        self._persister = persister
        self._telemetry = telemetry

    def _execute(self, request: EmbeddingRequest, /, **_: Any) -> EmbeddingResult:
        """Execute synchronous embedding generation for the given request.

        This method coordinates the complete embedding pipeline:
        1. Evaluates namespace access permissions
        2. Creates job lifecycle tracking
        3. Selects appropriate embedding model
        4. Generates embeddings for input texts
        5. Persists embeddings to storage
        6. Emits telemetry metrics

        Args:
            request: The embedding request containing texts, namespace, and options.
            **_: Additional keyword arguments (unused, for interface compatibility).

        Returns:
            EmbeddingResult containing generated embeddings and metadata.

        Raises:
            CoordinatorError: If namespace access is denied or embedding generation fails.

        Example:
            >>> request = EmbeddingRequest(
            ...     tenant_id="tenant1",
            ...     namespace="medical",
            ...     texts=["Sample text to embed"],
            ...     options=EmbeddingOptions(model="sentence-transformers/all-MiniLM-L6-v2")
            ... )
            >>> result = coordinator._execute(request)
            >>> print(f"Generated {len(result.response.vectors)} embeddings")

        """
        decision = self._evaluate_namespace(request)
        if not decision.allowed:
            detail = ProblemDetail(
                title="Namespace access denied",
                status=403,
                type="https://httpstatuses.com/403",
                detail=decision.reason or "Access to namespace not permitted",
            )
            raise CoordinatorError(detail.title, context={"problem": detail})

        config = decision.config or self._namespace_registry.get(request.namespace)
        job_id = self._lifecycle.create_job(request.tenant_id, "embed")
        correlation_id = uuid.uuid4().hex
        options = request.options or EmbeddingOptions()
        model_name = options.model or config.model_id

        if self._telemetry:
            self._telemetry.record_embedding_started(
                namespace=request.namespace,
                tenant_id=request.tenant_id,
                model=model_name,
            )

        texts: list[str] = []
        ids: list[str] = []
        metadata_payload: list[dict[str, Any]] = []

        for index, text in enumerate(request.texts):
            if not isinstance(text, str) or not text.strip():
                detail = ProblemDetail(
                    title="Invalid embedding input",
                    status=400,
                    type="https://httpstatuses.com/400",
                    detail="Embedding texts must be non-empty strings",
                )
                self._lifecycle.mark_failed(
                    job_id, reason=detail.detail or detail.title, stage="embed"
                )
                raise CoordinatorError(detail.title, context={"problem": detail, "job_id": job_id})
            body = text.strip()
            chunk_id = f"{job_id}:chunk:{index}"
            texts.append(body)
            ids.append(chunk_id)
            metadata_payload.append(
                {
                    "input_index": index,
                    "job_id": job_id,
                    "namespace": request.namespace,
                    "tenant_id": request.tenant_id,
                }
            )

        if not texts:
            payload = {"embeddings": 0, "model": model_name, "namespace": request.namespace}
            self._lifecycle.update_metadata(job_id, payload)
            self._lifecycle.mark_completed(job_id, payload=payload)
            metadata = EmbeddingMetadata(
                provider=config.provider,
                dimension=config.dim,
                duration_ms=0.0,
                model=model_name,
            )
            response = EmbeddingResponse(
                namespace=request.namespace, embeddings=(), metadata=metadata
            )
            return EmbeddingResult(
                job_id=job_id, duration_s=0.0, response=response, metadata=payload
            )

        embedder = self._registry.get(request.namespace)
        adapter_request = AdapterEmbeddingRequest(
            tenant_id=request.tenant_id,
            namespace=request.namespace,
            texts=texts,
            ids=ids,
            correlation_id=correlation_id,
            metadata=metadata_payload,
        )

        started = perf_counter()
        try:
            records = embedder.embed_documents(adapter_request)
        except Exception as exc:  # pragma: no cover - dependency failure
            detail = ProblemDetail(
                title="Embedding failed",
                status=502,
                type="https://httpstatuses.com/502",
                detail=str(exc),
            )
            self._lifecycle.mark_failed(job_id, reason=detail.detail or detail.title, stage="embed")
            if self._telemetry:
                self._telemetry.record_embedding_failure(
                    namespace=request.namespace,
                    tenant_id=request.tenant_id,
                    error=exc,
                )
            raise CoordinatorError(
                detail.title, context={"problem": detail, "job_id": job_id}
            ) from exc

        embeddings: list[EmbeddingVector] = []
        prepared_records: list[EmbeddingRecord] = []
        duration_ms = (perf_counter() - started) * 1000

        for record in records:
            meta = {**record.metadata}
            meta.setdefault("tenant_id", request.tenant_id)
            meta.setdefault("namespace", request.namespace)
            meta.setdefault("provider", config.provider)
            meta.setdefault("model", config.model_id)
            meta.setdefault("model_version", config.model_version)
            meta.setdefault("normalized", options.normalize or meta.get("normalized", False))
            meta.setdefault(
                "pipeline", f"{self._lifecycle.pipeline_name}:{self._lifecycle.pipeline_version}"
            )
            meta.setdefault("correlation_id", correlation_id)
            storage_meta = self._storage_metadata(record.kind, request.tenant_id, request.namespace)
            if storage_meta:
                meta.setdefault("storage", storage_meta)
            if record.kind == "multi_vector" and record.vectors:
                meta.setdefault("vectors", [list(vector) for vector in record.vectors])
            updated_record = replace(record, metadata=meta)
            prepared_records.append(updated_record)

            values: list[float] = []
            if updated_record.vectors:
                values = list(updated_record.vectors[0])
            if options.normalize and values and updated_record.kind != "sparse":
                magnitude = math.sqrt(sum(value * value for value in values))
                if magnitude > 0:
                    values = [value / magnitude for value in values]
            terms = dict(updated_record.terms or {}) if updated_record.kind == "sparse" else None
            dimension = updated_record.dim or (len(values) if values else 0)
            embeddings.append(
                EmbeddingVector(
                    id=updated_record.id,
                    model=model_name,
                    namespace=request.namespace,
                    kind=updated_record.kind,
                    dimension=dimension,
                    vector=values if updated_record.kind != "sparse" else None,
                    terms=terms,
                    metadata=meta,
                )
            )

        persistence_context = PersistenceContext(
            tenant_id=request.tenant_id,
            namespace=request.namespace,
            model=model_name,
            provider=config.provider,
            job_id=job_id,
            correlation_id=correlation_id,
            normalize=bool(options.normalize),
        )
        report = self._persister.persist_batch(prepared_records, persistence_context)

        payload = {
            "embeddings": len(embeddings),
            "model": model_name,
            "namespace": request.namespace,
            "provider": config.provider,
            "tenant_id": request.tenant_id,
            "persisted": report.persisted,
        }
        self._lifecycle.update_metadata(job_id, payload)
        self._lifecycle.mark_completed(job_id, payload=payload)

        if self._telemetry:
            self._telemetry.record_embedding_completed(
                namespace=request.namespace,
                tenant_id=request.tenant_id,
                model=model_name,
                provider=config.provider,
                duration_ms=duration_ms,
                embeddings=len(embeddings),
            )

        metadata = EmbeddingMetadata(
            provider=config.provider,
            dimension=config.dim,
            duration_ms=duration_ms,
            model=model_name,
        )
        response = EmbeddingResponse(
            namespace=request.namespace,
            embeddings=tuple(embeddings),
            metadata=metadata,
        )
        return EmbeddingResult(
            job_id=job_id,
            duration_s=duration_ms / 1000,
            response=response,
            metadata=payload,
        )

    def _evaluate_namespace(self, request: EmbeddingRequest) -> NamespaceAccessDecision:
        """Evaluate namespace access permissions for the embedding request.

        This method delegates to the namespace access policy to determine if the
        tenant has permission to perform embedding operations in the specified
        namespace with the required scope.

        Args:
            request: The embedding request containing tenant_id and namespace.

        Returns:
            NamespaceAccessDecision indicating whether access is allowed and any
            associated configuration or reason.

        Raises:
            CoordinatorError: If the policy evaluation fails unexpectedly.

        Example:
            >>> request = EmbeddingRequest(
            ...     tenant_id="tenant1",
            ...     namespace="medical",
            ...     texts=["Sample text"]
            ... )
            >>> decision = coordinator._evaluate_namespace(request)
            >>> if decision.allowed:
            ...     print("Access granted")

        """
        try:
            return self._policy.evaluate(
                namespace=request.namespace,
                tenant_id=request.tenant_id,
                required_scope=Scopes.EMBED_WRITE,
            )
        except Exception as exc:  # pragma: no cover - policy guard
            detail = ProblemDetail(
                title="Namespace policy failure",
                status=500,
                type="https://httpstatuses.com/500",
                detail=str(exc),
            )
            raise CoordinatorError(detail.title, context={"problem": detail}) from exc

    @staticmethod
    def _storage_metadata(kind: str, tenant_id: str, namespace: str) -> Mapping[str, Any] | None:
        """Generate storage metadata for embedding persistence.

        This static method creates metadata required for storing embeddings in
        different storage backends (FAISS index, Neo4j graph database) based on
        the embedding kind, tenant, and namespace.

        Args:
            kind: The type of embedding storage ("single_vector", "multi_vector", etc.).
            tenant_id: The tenant identifier for multi-tenancy.
            namespace: The namespace identifier (dots replaced with dashes).

        Returns:
            Mapping containing storage metadata (faiss_index, neo4j_label, filter)
            or None if the kind is not supported.

        Example:
            >>> metadata = EmbeddingCoordinator._storage_metadata(
            ...     kind="single_vector",
            ...     tenant_id="tenant1",
            ...     namespace="medical.docs"
            ... )
            >>> print(metadata["faiss_index"])
            /data/faiss/tenant1/medical-docs.index

        """
        sanitized_namespace = namespace.replace(".", "-")
        if kind in {"single_vector", "multi_vector"}:
            faiss_index = f"/data/faiss/{tenant_id}/{sanitized_namespace}.index"
            neo4j_label = f"Embedding::{sanitized_namespace}::{tenant_id}"
            return {
                "faiss_index": faiss_index,
                "neo4j_label": neo4j_label,
                "filter": {"tenant_id": tenant_id},
            }
        if kind == "sparse":
            index_name = f"embeddings-sparse-{sanitized_namespace}".replace("--", "-")
            return {"index": index_name, "tenant_id": tenant_id}
        return None


# ============================================================================
# ERROR TRANSLATION
# ============================================================================


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "EmbeddingCoordinator",
    "EmbeddingRequest",
    "EmbeddingResult",
]
