"""Embedding coordinator implementation."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Mapping, Sequence

from Medical_KG_rev.auth.scopes import Scopes
from Medical_KG_rev.embeddings.ports import EmbeddingRecord, EmbeddingRequest as AdapterEmbeddingRequest
from Medical_KG_rev.gateway.models import (
    EmbeddingMetadata,
    EmbeddingOptions,
    EmbeddingResponse,
    EmbeddingVector,
    ProblemDetail,
)
from Medical_KG_rev.services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from Medical_KG_rev.services.embedding.persister import EmbeddingPersister, PersistenceContext
from Medical_KG_rev.services.embedding.policy import NamespaceAccessPolicy
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


@dataclass
class EmbeddingRequest(CoordinatorRequest):
    namespace: str
    texts: Sequence[str]
    options: EmbeddingOptions | None = None


@dataclass
class EmbeddingResult(CoordinatorResult):
    response: EmbeddingResponse | None = None


class EmbeddingCoordinator(BaseCoordinator[EmbeddingRequest, EmbeddingResult]):
    """Coordinate embedding requests including persistence and telemetry."""

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
        from .base import CoordinatorMetrics

        super().__init__(config=config, metrics=CoordinatorMetrics.create(config.name))
        self._lifecycle = lifecycle
        self._registry = registry
        self._namespace_registry = namespace_registry
        self._policy = policy
        self._persister = persister
        self._telemetry = telemetry

    def _execute(self, request: EmbeddingRequest, /, **_: Any) -> EmbeddingResult:
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
                self._lifecycle.mark_failed(job_id, reason=detail.detail or detail.title, stage="embed")
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
            response = EmbeddingResponse(namespace=request.namespace, embeddings=(), metadata=metadata)
            return EmbeddingResult(job_id=job_id, duration_s=0.0, response=response, metadata=payload)

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
            raise CoordinatorError(detail.title, context={"problem": detail, "job_id": job_id}) from exc

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
            meta.setdefault("pipeline", f"{self._lifecycle.pipeline_name}:{self._lifecycle.pipeline_version}")
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

    def _evaluate_namespace(self, request: EmbeddingRequest):
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


__all__ = [
    "EmbeddingCoordinator",
    "EmbeddingRequest",
    "EmbeddingResult",
]
