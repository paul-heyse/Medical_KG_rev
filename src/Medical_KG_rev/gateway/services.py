"""Protocol-agnostic gateway service layer."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import structlog
from Medical_KG_rev.chunking.exceptions import (
    ChunkerConfigurationError,
    ChunkingFailedError,
    ChunkingUnavailableError,
    InvalidDocumentError,
    ProfileNotFoundError,
    TokenizerMismatchError,
)

from ..adapters import AdapterDomain, AdapterPluginManager, get_plugin_manager
from ..adapters.plugins.models import AdapterRequest

from ..kg import ShaclValidator, ValidationError
from ..auth.scopes import Scopes
from ..observability.metrics import (
    CROSS_TENANT_ACCESS_ATTEMPTS,
    observe_job_duration,
    record_business_event,
)
from ..orchestration import (
    HaystackRetriever,
    JobLedger,
    JobLedgerEntry,
)
from ..orchestration.dagster import (
    DagsterOrchestrator,
    PipelineConfigLoader,
    ResiliencePolicyLoader,
    StageFactory,
    build_default_stage_factory,
    submit_to_dagster,
)
from ..orchestration.stages.contracts import StageContext
from ..services.extraction.templates import TemplateValidationError, validate_template
from ..services.embedding.namespace.access import validate_namespace_access
from ..services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from ..services.embedding.registry import EmbeddingModelRegistry
from ..services.retrieval.chunking import ChunkingOptions, ChunkingService
from ..utils.errors import ProblemDetail as PipelineProblemDetail
from ..validation import UCUMValidator
from ..validation.fhir import FHIRValidationError, FHIRValidator
from Medical_KG_rev.embeddings.ports import EmbeddingRequest as AdapterEmbeddingRequest
from .models import (
    AdapterConfigSchemaView,
    AdapterHealthView,
    AdapterMetadataView,
    BatchError,
    BatchOperationResult,
    ChunkRequest,
    DocumentChunk,
    DocumentSummary,
    EmbeddingMetadata,
    EmbeddingResponse,
    EmbeddingVector,
    EmbedRequest,
    EmbeddingOptions,
    EntityLinkRequest,
    EntityLinkResult,
    ExtractionRequest,
    ExtractionResult,
    IngestionRequest,
    JobEvent,
    JobHistoryEntry,
    JobStatus,
    KnowledgeGraphWriteRequest,
    KnowledgeGraphWriteResult,
    NamespaceInfo,
    NamespaceValidationResponse,
    NamespaceValidationResult,
    OperationStatus,
    ProblemDetail,
    RetrievalResult,
    RetrieveRequest,
    SearchArguments,
    build_batch_result,
)
from .sse.manager import EventStreamManager

logger = structlog.get_logger(__name__)


class GatewayError(RuntimeError):
    """Domain specific exception carrying problem detail information."""

    def __init__(self, detail: ProblemDetail):
        super().__init__(detail.title)
        self.detail = detail


def _build_stage_factory(manager: AdapterPluginManager | None = None) -> StageFactory:
    registry = build_default_stage_factory(manager or get_plugin_manager())
    return StageFactory(registry)


@dataclass
class GatewayService:
    """Coordinates business logic shared across protocols."""

    _PIPELINE_NAME = "gateway-direct"
    _PIPELINE_VERSION = "v1"

    events: EventStreamManager
    orchestrator: DagsterOrchestrator
    ledger: JobLedger
    adapter_manager: AdapterPluginManager = field(default_factory=get_plugin_manager)
    stage_factory: StageFactory | None = None
    chunker: ChunkingService | None = None
    retriever: HaystackRetriever = field(default_factory=HaystackRetriever)
    shacl: ShaclValidator = field(default_factory=ShaclValidator.default)
    ucum: UCUMValidator = field(default_factory=UCUMValidator)
    fhir: FHIRValidator = field(default_factory=FHIRValidator)
    embedding_registry: EmbeddingModelRegistry = field(default_factory=EmbeddingModelRegistry)
    namespace_registry: EmbeddingNamespaceRegistry | None = None

    def __post_init__(self) -> None:
        if self.stage_factory is None:
            self.stage_factory = _build_stage_factory(self.adapter_manager)
        if self.chunker is None:
            self.chunker = ChunkingService(stage_factory=self.stage_factory)
        if self.namespace_registry is None:
            self.namespace_registry = self.embedding_registry.namespace_registry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _convert_problem(self, problem: PipelineProblemDetail) -> ProblemDetail:
        payload = problem.to_response()
        extensions = payload.pop("extra", {})
        payload.setdefault("extensions", extensions)
        return ProblemDetail.model_validate(payload)

    def _to_job_status(self, entry: JobLedgerEntry) -> JobStatus:
        history = [
            JobHistoryEntry(
                from_status=transition.from_status,
                to_status=transition.to_status,
                stage=transition.stage,
                reason=transition.reason,
                timestamp=transition.timestamp,
            )
            for transition in entry.history
        ]
        return JobStatus(
            job_id=entry.job_id,
            doc_key=entry.doc_key,
            tenant_id=entry.tenant_id,
            status=entry.status,
            stage=entry.stage,
            pipeline=entry.pipeline,
            metadata=dict(entry.metadata),
            attempts=entry.attempts,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            history=history,
        )

    def _new_job(self, tenant_id: str, operation: str, *, metadata: dict | None = None) -> str:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        doc_key = f"{operation}:{job_id}"
        self.ledger.create(
            job_id=job_id,
            doc_key=doc_key,
            tenant_id=tenant_id,
            pipeline=operation,
            metadata={"operation": operation, **(metadata or {})},
        )
        self.ledger.mark_processing(job_id, stage=operation)
        logger.info("gateway.job.created", tenant_id=tenant_id, job_id=job_id, operation=operation)
        self.events.publish(
            JobEvent(job_id=job_id, type="jobs.started", payload={"operation": operation})
        )
        return job_id

    def _complete_job(self, job_id: str, payload: dict | None = None) -> None:
        logger.info("gateway.job.completed", job_id=job_id)
        self.ledger.mark_completed(job_id, metadata=payload or {})
        self.events.publish(JobEvent(job_id=job_id, type="jobs.completed", payload=payload or {}))

    def _fail_job(self, job_id: str, reason: str) -> None:
        logger.warning("gateway.job.failed", job_id=job_id, reason=reason)
        self.ledger.mark_failed(job_id, stage="error", reason=reason)
        self.events.publish(JobEvent(job_id=job_id, type="jobs.failed", payload={"reason": reason}))

    def _submit_dagster_job(
        self,
        *,
        dataset: str,
        request: IngestionRequest,
        item: Mapping[str, Any],
        metadata: dict[str, Any],
    ) -> OperationStatus:
        pipeline_name = self._resolve_pipeline(dataset, item)
        topology = self.orchestrator.pipeline_loader.load(pipeline_name)
        document_id = str(item.get("id") or uuid.uuid4().hex)
        doc_key = f"{dataset}:{document_id}"
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        domain = self._ingest_domain(topology) or AdapterDomain.BIOMEDICAL
        correlation_id = uuid.uuid4().hex
        adapter_request = AdapterRequest(
            tenant_id=request.tenant_id,
            correlation_id=correlation_id,
            domain=domain,
            parameters={"dataset": dataset, "item": item},
        )
        payload = {"dataset": dataset, "item": item, "metadata": metadata}
        ledger_metadata = {
            "dataset": dataset,
            "item": item,
            **metadata,
            "pipeline_version": topology.version,
            "correlation_id": correlation_id,
            "adapter_request": adapter_request.model_dump(),
            "payload": payload,
        }

        entry = self.ledger.idempotent_create(
            job_id=job_id,
            doc_key=doc_key,
            tenant_id=request.tenant_id,
            pipeline=pipeline_name,
            metadata=ledger_metadata,
        )
        duplicate = entry.job_id != job_id
        if duplicate:
            error = BatchError(
                code="duplicate",
                message="Document already queued",
                details={"dataset": dataset},
            )
            return OperationStatus(
                job_id=entry.job_id,
                status="failed",
                message=f"Duplicate job for {dataset}",
                metadata={
                    "dataset": dataset,
                    "pipeline": entry.pipeline,
                    "doc_key": entry.doc_key,
                    "duplicate": True,
                },
                http_status=409,
                error=error,
            )

        context = StageContext(
            tenant_id=request.tenant_id,
            job_id=job_id,
            doc_id=document_id,
            correlation_id=correlation_id,
            metadata={"dataset": dataset, **metadata},
            pipeline_name=pipeline_name,
            pipeline_version=topology.version,
        )

        self.ledger.mark_processing(job_id, stage="bootstrap")
        self.events.publish(
            JobEvent(
                job_id=job_id,
                type="jobs.started",
                payload={"pipeline": pipeline_name, "dataset": dataset},
            )
        )

        error: BatchError | None = None
        status = "completed"
        message = f"Executed pipeline {pipeline_name}"
        result_metadata: dict[str, Any] = {}
        http_status = 202

        try:
            run_result = submit_to_dagster(
                self.orchestrator,
                pipeline=pipeline_name,
                context=context,
                adapter_request=adapter_request,
                payload=payload,
            )
            result_metadata = {"state": run_result.state.serialise()}
            if run_result.success:
                self.ledger.mark_completed(job_id, metadata=result_metadata)
                self.events.publish(
                    JobEvent(
                        job_id=job_id,
                        type="jobs.completed",
                        payload={"pipeline": pipeline_name},
                    )
                )
            else:
                status = "failed"
                message = f"Pipeline {pipeline_name} reported failure"
                error = BatchError(
                    code="pipeline-failed",
                    message="Dagster job reported failure",
                    details={"pipeline": pipeline_name},
                )
                self.ledger.mark_failed(
                    job_id,
                    stage=pipeline_name,
                    reason="dagster-failure",
                    metadata=result_metadata,
                )
                self.events.publish(
                    JobEvent(
                        job_id=job_id,
                        type="jobs.failed",
                        payload={"pipeline": pipeline_name},
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive guard
            status = "failed"
            message = f"Pipeline {pipeline_name} execution raised an exception"
            error = BatchError(
                code="dagster-error",
                message=str(exc),
                details={"pipeline": pipeline_name},
            )
            self.ledger.mark_failed(
                job_id,
                stage=pipeline_name,
                reason=str(exc),
                metadata={"exception": str(exc)},
            )
            self.events.publish(
                JobEvent(
                    job_id=job_id,
                    type="jobs.failed",
                    payload={"pipeline": pipeline_name, "error": str(exc)},
                )
            )

        return OperationStatus(
            job_id=job_id,
            status=status,
            message=message,
            metadata={
                "dataset": dataset,
                "pipeline": pipeline_name,
                "doc_key": doc_key,
                "duplicate": False,
                **result_metadata,
            },
            http_status=http_status,
            error=error,
        )

    def _resolve_pipeline(self, dataset: str, item: Mapping[str, Any]) -> str:
        dataset_key = dataset.lower()
        available = self.orchestrator.available_pipelines()
        for name in available:
            topology = self.orchestrator.pipeline_loader.load(name)
            if dataset_key in {source.lower() for source in topology.applicable_sources}:
                return topology.name
        if str(item.get("document_type", "")).lower() == "pdf":
            return "pdf-two-phase"
        return "auto"

    def _ingest_domain(self, topology) -> AdapterDomain | None:
        for stage in topology.stages:
            if stage.stage_type == "ingest":
                domain = stage.config.get("domain")
                if domain:
                    try:
                        return AdapterDomain(domain)
                    except Exception:  # pragma: no cover - validation guard
                        return None
        return None

    def ingest(self, dataset: str, request: IngestionRequest) -> BatchOperationResult:
        started = perf_counter()
        statuses: list[OperationStatus] = []
        for item in request.items:
            metadata = dict(request.metadata)
            if request.profile:
                metadata.setdefault("profile", request.profile)
            if request.chunking_options:
                metadata.setdefault("chunking_options", dict(request.chunking_options))
            status = self._submit_dagster_job(
                dataset=dataset,
                request=request,
                item=item,
                metadata=metadata,
            )
            statuses.append(status)
        result = build_batch_result(statuses)
        duration = perf_counter() - started
        observe_job_duration("ingest", duration)
        if request.items:
            record_business_event("documents_ingested", request.tenant_id)
        return result

    def chunk_document(self, request: ChunkRequest) -> Sequence[DocumentChunk]:
        job_id = self._new_job(request.tenant_id, "chunk")
        options_payload = request.options if isinstance(request.options, dict) else {}
        raw_text = options_payload.get("text") if isinstance(options_payload.get("text"), str) else None
        if not isinstance(raw_text, str) or not raw_text.strip():
            detail = ProblemDetail(
                title="Invalid document payload",
                status=400,
                type="https://httpstatuses.com/400",
                detail="Chunking requests must include a non-empty 'text' field in options",
                instance=f"/v1/chunk/{request.document_id}",
            )
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail)
        metadata = {
            key: value
            for key, value in options_payload.items()
            if key != "text"
        }
        profile_hint: str | None = None
        raw_profile = metadata.get("profile") if isinstance(metadata, dict) else None
        if isinstance(raw_profile, str) and raw_profile:
            profile_hint = raw_profile
        options = ChunkingOptions(
            strategy=request.strategy,
            max_tokens=request.chunk_size,
            overlap=request.overlap,
            metadata=metadata,
        )
        chunker = self.chunker or ChunkingService(stage_factory=self.stage_factory)
        try:
            raw_chunks = chunker.chunk(
                request.tenant_id,
                request.document_id,
                raw_text,
                options,
            )
        except ProfileNotFoundError as exc:
            detail = ProblemDetail(
                title="Chunking profile not found",
                status=400,
                type="https://medical-kg/errors/chunking-profile-not-found",
                detail=str(exc),
                extensions={"available_profiles": list(getattr(exc, "available", []))},
            )
            record_chunking_failure(profile_hint or "unknown", "ProfileNotFoundError")
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc
        except TokenizerMismatchError as exc:
            detail = ProblemDetail(
                title="Tokenizer mismatch",
                status=500,
                type="https://medical-kg/errors/tokenizer-mismatch",
                detail=str(exc),
            )
            record_chunking_failure(profile_hint or "unknown", "TokenizerMismatchError")
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc
        except ChunkingFailedError as exc:
            message = exc.detail or str(exc) or "Chunking process failed"
            detail = ProblemDetail(
                title="Chunking failed",
                status=500,
                type="https://medical-kg/errors/chunking-failed",
                detail=message,
            )
            record_chunking_failure(profile_hint or "unknown", "ChunkingFailedError")
            self._fail_job(job_id, message)
            raise GatewayError(detail) from exc
        except InvalidDocumentError as exc:
            detail = ProblemDetail(
                title="Invalid document payload",
                status=400,
                type="https://httpstatuses.com/400",
                detail=str(exc),
                instance=f"/v1/chunk/{request.document_id}",
            )
            record_chunking_failure(profile_hint or "unknown", "InvalidDocumentError")
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc
        except ChunkerConfigurationError as exc:
            detail = ProblemDetail(
                title="Chunker configuration invalid",
                status=422,
                type="https://httpstatuses.com/422",
                detail=str(exc),
                extensions={"valid_strategies": chunker.available_strategies()},
            )
            record_chunking_failure(profile_hint or "unknown", "ChunkerConfigurationError")
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc
        except ChunkingUnavailableError as exc:
            retry_after = max(1, int(round(exc.retry_after)))
            detail = ProblemDetail(
                title="Chunking temporarily unavailable",
                status=503,
                type="https://httpstatuses.com/503",
                detail=str(exc),
                extensions={"retry_after": retry_after},
            )
            record_chunking_failure(profile_hint or "unknown", "ChunkingUnavailableError")
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc
        except MineruOutOfMemoryError as exc:
            detail = ProblemDetail(
                title="MinerU out of memory",
                status=503,
                type="https://medical-kg/errors/mineru-oom",
                detail=str(exc),
                extensions={"reason": "gpu_out_of_memory"},
            )
            record_chunking_failure(profile_hint or "unknown", "MineruOutOfMemoryError")
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc
        except MineruGpuUnavailableError as exc:
            detail = ProblemDetail(
                title="MinerU GPU unavailable",
                status=503,
                type="https://medical-kg/errors/mineru-gpu-unavailable",
                detail=str(exc),
                extensions={"reason": "gpu_unavailable"},
            )
            record_chunking_failure(profile_hint or "unknown", "MineruGpuUnavailableError")
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc
        except MemoryError as exc:
            message = str(exc) or "Chunking operation exhausted available memory"
            detail = ProblemDetail(
                title="Chunking resources exhausted",
                status=503,
                type="https://httpstatuses.com/503",
                detail=message,
                extensions={"retry_after": 60},
            )
            self._fail_job(job_id, message)
            raise GatewayError(detail) from exc
        except TimeoutError as exc:
            message = str(exc) or "Chunking operation timed out"
            detail = ProblemDetail(
                title="Chunking resources exhausted",
                status=503,
                type="https://httpstatuses.com/503",
                detail=message,
                extensions={"retry_after": 30},
            )
            self._fail_job(job_id, message)
            raise GatewayError(detail) from exc
        except RuntimeError as exc:
            message = str(exc)
            if "GPU semantic checks" in message:
                detail = ProblemDetail(
                    title="GPU unavailable for semantic chunking",
                    status=503,
                    type="https://httpstatuses.com/503",
                    detail=message,
                    extensions={"reason": "gpu_unavailable"},
                )
                self._fail_job(job_id, message)
                raise GatewayError(detail) from exc
            self._fail_job(job_id, message or "Runtime error during chunking")
            raise
        chunks: list[DocumentChunk] = []
        for index, chunk in enumerate(raw_chunks):
            metadata = dict(chunk.meta)
            metadata.setdefault("granularity", chunk.granularity)
            metadata.setdefault("chunker", chunk.chunker)
            chunks.append(
                DocumentChunk(
                    document_id=request.document_id,
                    chunk_index=index,
                    content=chunk.body,
                    metadata=metadata,
                    token_count=metadata.get("token_count", 0),
                )
            )
        self.ledger.update_metadata(job_id, {"chunks": len(chunks)})
        self._complete_job(job_id, payload={"chunks": len(chunks)})
        return chunks

    def embed(self, request: EmbedRequest) -> EmbeddingResponse:
        if self.namespace_registry is None:
            raise RuntimeError("Namespace registry not initialised")

        started = perf_counter()
        options = request.options or EmbeddingOptions()
        namespace = request.namespace
        config = self.namespace_registry.get(namespace)

        access = validate_namespace_access(
            self.namespace_registry,
            namespace=namespace,
            tenant_id=request.tenant_id,
            required_scope=Scopes.EMBED_WRITE,
        )
        if not access.allowed:
            if access.reason and "Tenant" in access.reason:
                allowed = ",".join(sorted(self.namespace_registry.get_allowed_tenants(namespace)))
                CROSS_TENANT_ACCESS_ATTEMPTS.labels(
                    source_tenant=request.tenant_id,
                    target_tenant=allowed or "restricted",
                ).inc()
            detail = ProblemDetail(
                title="Namespace access denied",
                status=403,
                type="https://httpstatuses.com/403",
                detail=access.reason or "Access to namespace not permitted",
            )
            raise GatewayError(detail)

        job_id = self._new_job(request.tenant_id, "embed")
        correlation_id = uuid.uuid4().hex
        model_name = options.model or config.model_id

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
                self._fail_job(job_id, detail.detail or detail.title)
                raise GatewayError(detail)
            body = text.strip()
            chunk_id = f"{job_id}:chunk:{index}"
            texts.append(body)
            ids.append(chunk_id)
            metadata_payload.append(
                {
                    "input_index": index,
                    "job_id": job_id,
                    "namespace": namespace,
                    "tenant_id": request.tenant_id,
                }
            )

        if not texts:
            payload = {"embeddings": 0, "model": model_name, "namespace": namespace}
            self.ledger.update_metadata(job_id, payload)
            self._complete_job(job_id, payload=payload)
            metadata = EmbeddingMetadata(
                provider=config.provider,
                dimension=config.dim,
                duration_ms=0.0,
                model=model_name,
            )
            return EmbeddingResponse(namespace=namespace, embeddings=(), metadata=metadata)

        embedder = self.embedding_registry.get(namespace)
        adapter_request = AdapterEmbeddingRequest(
            tenant_id=request.tenant_id,
            namespace=namespace,
            texts=texts,
            ids=ids,
            correlation_id=correlation_id,
            metadata=metadata_payload,
        )

        try:
            records = embedder.embed_documents(adapter_request)
        except Exception as exc:  # pragma: no cover - network/library error
            detail = ProblemDetail(
                title="Embedding failed",
                status=502,
                type="https://httpstatuses.com/502",
                detail=str(exc),
            )
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc

        embeddings: list[EmbeddingVector] = []
        storage_router = getattr(self.embedding_registry, "storage_router", None)
        duration_ms = (perf_counter() - started) * 1000

        for record in records:
            meta = {**record.metadata}
            meta.setdefault("tenant_id", request.tenant_id)
            meta.setdefault("namespace", namespace)
            meta.setdefault("provider", config.provider)
            meta.setdefault("model", config.model_id)
            meta.setdefault("model_version", config.model_version)
            meta.setdefault("normalized", options.normalize or meta.get("normalized", False))
            meta.setdefault("pipeline", f"{self._PIPELINE_NAME}:{self._PIPELINE_VERSION}")
            meta.setdefault("correlation_id", correlation_id)
            storage_meta = self._storage_metadata(record.kind, request.tenant_id, namespace)
            if storage_meta:
                meta.setdefault("storage", storage_meta)
            if record.kind == "multi_vector" and record.vectors:
                meta.setdefault("vectors", [list(vector) for vector in record.vectors])
            updated_record = replace(record, metadata=meta)
            if storage_router is not None:
                storage_router.persist(updated_record)

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
                    namespace=namespace,
                    kind=updated_record.kind,
                    dimension=dimension,
                    vector=values if updated_record.kind != "sparse" else None,
                    terms=terms,
                    metadata=meta,
                )
            )

        payload = {
            "embeddings": len(embeddings),
            "model": model_name,
            "namespace": namespace,
            "provider": config.provider,
            "tenant_id": request.tenant_id,
        }
        self.ledger.update_metadata(job_id, payload)
        self._complete_job(job_id, payload=payload)
        if embeddings:
            record_business_event("embeddings_generated", request.tenant_id)

        metadata = EmbeddingMetadata(
            provider=config.provider,
            dimension=config.dim,
            duration_ms=duration_ms,
            model=model_name,
        )
        return EmbeddingResponse(
            namespace=namespace,
            embeddings=embeddings,
            metadata=metadata,
        )

    def _storage_metadata(self, kind: str, tenant_id: str, namespace: str) -> dict[str, Any]:
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
            return {
                "opensearch_index": f"{index_name}-{tenant_id}",
                "rank_features_field": "splade_terms",
                "filter": {"term": {"tenant_id": tenant_id}},
            }
        if kind == "neural_sparse":
            index_name = f"embeddings-neural-{sanitized_namespace}".replace("--", "-")
            return {
                "opensearch_index": f"{index_name}-{tenant_id}",
                "neural_field": "neural_embedding",
                "filter": {"term": {"tenant_id": tenant_id}},
            }
        return {}

    def list_namespaces(
        self,
        *,
        tenant_id: str,
        scope: str = Scopes.EMBED_READ,
    ) -> list[NamespaceInfo]:
        if self.namespace_registry is None:
            raise RuntimeError("Namespace registry not initialised")
        entries = self.namespace_registry.list_enabled(tenant_id=tenant_id, scope=scope)
        return [
            NamespaceInfo(
                id=namespace,
                provider=config.provider,
                kind=config.kind.value,
                dimension=config.dim,
                max_tokens=config.max_tokens,
                enabled=config.enabled,
                allowed_tenants=list(config.allowed_tenants),
                allowed_scopes=list(config.allowed_scopes),
            )
            for namespace, config in entries
        ]

    def validate_namespace_texts(
        self,
        *,
        tenant_id: str,
        namespace: str,
        texts: Sequence[str],
    ) -> NamespaceValidationResponse:
        if self.namespace_registry is None:
            raise RuntimeError("Namespace registry not initialised")
        access = validate_namespace_access(
            self.namespace_registry,
            namespace=namespace,
            tenant_id=tenant_id,
            required_scope=Scopes.EMBED_READ,
        )
        if not access.allowed:
            detail = ProblemDetail(
                title="Namespace access denied",
                status=403,
                type="https://httpstatuses.com/403",
                detail=access.reason or "Access to namespace not permitted",
            )
            raise GatewayError(detail)

        max_tokens = self.namespace_registry.get_max_tokens(namespace)
        if max_tokens is None:
            detail = ProblemDetail(
                title="Namespace lacks token budget",
                status=400,
                type="https://httpstatuses.com/400",
                detail=f"Namespace '{namespace}' does not declare max_tokens",
            )
            raise GatewayError(detail)

        try:
            tokenizer = self.namespace_registry.get_tokenizer(namespace)
        except (ValueError, RuntimeError) as exc:
            detail = ProblemDetail(
                title="Tokenizer unavailable",
                status=502,
                type="https://httpstatuses.com/502",
                detail=str(exc),
            )
            raise GatewayError(detail) from exc

        results: list[NamespaceValidationResult] = []
        for index, text in enumerate(texts):
            encoded = tokenizer.encode(text or "", add_special_tokens=False)
            token_count = len(encoded)
            exceeds = token_count > max_tokens
            warning = f"Exceeds {max_tokens} tokens" if exceeds else None
            results.append(
                NamespaceValidationResult(
                    text_index=index,
                    token_count=token_count,
                    exceeds_budget=exceeds,
                    warning=warning,
                )
            )

        return NamespaceValidationResponse(
            namespace=namespace,
            valid=all(not item.exceeds_budget for item in results),
            results=results,
        )

    def retrieve(self, request: RetrieveRequest) -> RetrievalResult:
        started = perf_counter()
        job_id = self._new_job(request.tenant_id, "retrieve")
        filters = dict(request.filters or {})
        metadata = dict(request.metadata)
        if request.profile:
            metadata['profile'] = request.profile
        try:
            results = self.retriever.retrieve(request.query, filters=filters or None)
        except Exception as exc:
            detail = ProblemDetail(
                title="Retrieval failed",
                status=502,
                type="https://httpstatuses.com/502",
                detail=str(exc),
            )
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc
        documents: list[DocumentSummary] = []
        for index, item in enumerate(results[: request.top_k]):
            meta = dict(item.get('meta') or {})
            doc_id = str(item.get('id') or meta.pop('id', None) or f"retrieved-{index}")
            title = meta.pop('title', None) or doc_id
            summary = meta.pop('summary', None)
            source = meta.pop('source', None) or 'hybrid'
            score = float(item.get('score') or meta.pop('score', 0.0) or 0.0)
            content = item.get('content')
            if content and 'content' not in meta:
                meta['content'] = content
            documents.append(
                DocumentSummary(
                    id=doc_id,
                    title=title,
                    score=score,
                    summary=summary,
                    source=source,
                    metadata=meta,
                    explain=item.get('explain') if request.explain else None,
                )
            )
        retrieval_duration = perf_counter() - started
        observe_job_duration('retrieve', retrieval_duration)
        record_business_event('retrieval_requests', request.tenant_id)
        if documents:
            record_business_event('documents_retrieved', request.tenant_id)
        stage_timings = {'retrieve': round(retrieval_duration, 6)}
        result = RetrievalResult(
            query=request.query,
            documents=documents,
            total=len(documents),
            rerank_metrics={'stage_timings_ms': {name: round(value * 1000, 3) for name, value in stage_timings.items()}},
            pipeline_version='haystack-hybrid/v1',
            partial=False,
            degraded=False,
            errors=[],
            stage_timings=stage_timings,
        )
        ledger_metadata = {
            'documents': result.total,
            'pipeline_version': result.pipeline_version,
            'filters': filters,
            'metadata': metadata,
        }
        self.ledger.update_metadata(job_id, ledger_metadata)
        if result.total == 0 and request.rerank:
            ledger_metadata['status'] = 'no-results'
        self._complete_job(job_id, payload=ledger_metadata)
        return result

    def entity_link(self, request: EntityLinkRequest) -> Sequence[EntityLinkResult]:
        job_id = self._new_job(request.tenant_id, "entity-link")
        results = [
            EntityLinkResult(
                mention=mention,
                entity_id=f"ENT-{abs(hash(mention)) % 9999:04d}",
                confidence=0.9,
                metadata={"context": request.context},
            )
            for mention in request.mentions
        ]
        self.ledger.update_metadata(job_id, {"links": len(results)})
        self._complete_job(job_id, payload={"links": len(results)})
        return results

    def extract(self, kind: str, request: ExtractionRequest) -> ExtractionResult:
        job_id = self._new_job(request.tenant_id, f"extract:{kind}")
        text = request.options.get("text") if request.options else None
        if not isinstance(text, str) or not text.strip():
            text = self._default_extraction_text(kind)
        try:
            payload = self._build_template(kind, text)
            validated = validate_template(kind, payload, text)
        except TemplateValidationError as exc:
            detail = ProblemDetail(
                title="Extraction validation failed",
                status=422,
                type="https://httpstatuses.com/422",
                detail=str(exc),
            )
            raise GatewayError(detail) from exc
        self.ledger.update_metadata(job_id, {"kind": kind, "spans": len(validated)})
        self._complete_job(job_id, payload={"kind": kind})
        return ExtractionResult(kind=kind, document_id=request.document_id, results=[validated])

    def write_kg(self, request: KnowledgeGraphWriteRequest) -> KnowledgeGraphWriteResult:
        job_id = self._new_job(request.tenant_id, "kg-write")
        nodes_payload = [node.model_dump(mode="python") for node in request.nodes]
        edges_payload = [edge.model_dump(mode="python") for edge in request.edges]
        try:
            self.shacl.validate_payload(nodes_payload, edges_payload)
            self._validate_fhir(nodes_payload)
        except (ValidationError, FHIRValidationError) as exc:
            detail = ProblemDetail(
                title="Knowledge graph validation failed",
                status=422,
                type="https://httpstatuses.com/422",
                detail=str(exc),
            )
            raise GatewayError(detail) from exc
        self.ledger.update_metadata(
            job_id,
            {
                "nodes": len(request.nodes),
                "edges": len(request.edges),
                "transactional": request.transactional,
            },
        )
        self._complete_job(
            job_id,
            payload={
                "nodes": len(request.nodes),
                "edges": len(request.edges),
                "transactional": request.transactional,
            },
        )
        return KnowledgeGraphWriteResult(
            nodes_written=len(request.nodes),
            edges_written=len(request.edges),
            metadata={"transactional": request.transactional},
        )

    # ------------------------------------------------------------------
    # Template helpers
    # ------------------------------------------------------------------
    def _default_extraction_text(self, kind: str) -> str:
        base = {
            "pico": (
                "Population: Adults with hypertension. Intervention: ACE inhibitor administered daily. "
                "Comparison: Placebo tablets. Outcome: Reduced systolic blood pressure at 12 weeks."
            ),
            "effects": (
                "Effect size 0.45 (95% CI 0.30-0.60) for reduction in systolic blood pressure after 12 weeks."
            ),
            "ae": ("Adverse event: dry cough (moderate, probable) occurred in 12% of patients."),
            "dose": ("Lisinopril 20 mg orally once daily for 12 weeks improved outcomes."),
            "eligibility": (
                "Inclusion: Age 18-75 with primary hypertension. Exclusion: Severe renal impairment or pregnancy."
            ),
        }
        return base.get(kind, base["pico"])

    def _span(self, text: str, phrase: str) -> dict[str, object]:
        if not phrase:
            return {"text": "", "start": 0, "end": 0}
        lower_text = text.lower()
        lower_phrase = phrase.lower()
        start = lower_text.find(lower_phrase)
        if start == -1:
            start = 0
            snippet = text[: len(phrase)]
        else:
            snippet = text[start : start + len(phrase)]
        end = start + len(snippet)
        return {"text": snippet, "start": start, "end": end}

    def _build_template(self, kind: str, text: str) -> dict[str, object]:
        lowered = kind.lower()
        if lowered == "pico":
            return {
                "population": {
                    "description": "Adults with hypertension",
                    "age_range": "18-75",
                    "gender": None,
                    "condition": "Hypertension",
                    "sample_size": 120,
                    "span": self._span(text, "Adults with hypertension"),
                },
                "interventions": [
                    {
                        "name": "ACE inhibitor",
                        "type": "medication",
                        "route": "oral",
                        "dose": "20 mg",
                        "span": self._span(text, "ACE inhibitor"),
                    }
                ],
                "comparison": {
                    "description": "Placebo",
                    "span": self._span(text, "Placebo"),
                },
                "outcomes": [
                    {
                        "name": "Reduced systolic blood pressure",
                        "measurement": "mmHg",
                        "timepoint": "12 weeks",
                        "effect_size": 0.45,
                        "span": self._span(text, "Reduced systolic blood pressure"),
                    }
                ],
                "confidence": 0.82,
            }
        if lowered == "effects":
            return {
                "measures": [
                    {
                        "outcome": "Systolic blood pressure",
                        "effect_size": 0.45,
                        "unit": "standardised mean difference",
                        "ci_low": 0.3,
                        "ci_high": 0.6,
                        "span": self._span(text, "Effect size 0.45"),
                    }
                ]
            }
        if lowered == "ae":
            return {
                "events": [
                    {
                        "event_type": "Dry cough",
                        "severity": "moderate",
                        "frequency": "12%",
                        "causality": "probable",
                        "span": self._span(text, "dry cough"),
                    }
                ]
            }
        if lowered == "dose":
            ucum = self.ucum.validate_value(20, "mg", context="dose")
            return {
                "regimens": [
                    {
                        "drug": "Lisinopril",
                        "dose_value": ucum.normalized_value,
                        "dose_unit": ucum.normalized_unit,
                        "route": "oral",
                        "frequency": "once daily",
                        "duration": "12 weeks",
                        "span": self._span(text, "20 mg"),
                    }
                ]
            }
        if lowered == "eligibility":
            return {
                "inclusion": [
                    {
                        "text": "Age 18-75",
                        "span": self._span(text, "Age 18-75"),
                        "metadata": {"type": "age"},
                    },
                    {
                        "text": "Primary hypertension",
                        "span": self._span(text, "primary hypertension"),
                        "metadata": {"type": "condition"},
                    },
                ],
                "exclusion": [
                    {
                        "text": "Severe renal impairment",
                        "span": self._span(text, "Severe renal impairment"),
                        "metadata": {"type": "condition"},
                    },
                ],
            }
        raise TemplateValidationError(f"Unknown extraction kind '{kind}'")

    def _validate_fhir(self, nodes: Sequence[Mapping[str, object]]) -> None:
        for node in nodes:
            properties = node.get("properties", {}) if isinstance(node, Mapping) else {}
            resource = properties.get("fhir") if isinstance(properties, Mapping) else None
            if isinstance(resource, Mapping):
                self.fhir.validate(resource)  # may raise FHIRValidationError

    def search(self, args: SearchArguments) -> RetrievalResult:
        job_id = self._new_job("system", "search")
        request = RetrievalResult(
            query=args.query,
            documents=[
                DocumentSummary(
                    id="doc-search-1",
                    title="GraphQL Search Result",
                    score=0.95,
                    summary=f"Result for {args.query}",
                    source="search",
                    metadata=args.filters,
                )
            ],
            total=1,
        )
        self.ledger.update_metadata(job_id, {"query": args.query})
        self._complete_job(job_id, payload={"documents": 1})
        return request

    # ------------------------------------------------------------------
    # Job APIs
    # ------------------------------------------------------------------
    def get_job(self, job_id: str, *, tenant_id: str) -> JobStatus | None:
        entry = self.ledger.get(job_id)
        if not entry or entry.tenant_id != tenant_id:
            return None
        return self._to_job_status(entry)

    def list_jobs(self, *, tenant_id: str, status: str | None = None) -> list[JobStatus]:
        entries = self.ledger.list(status=status)
        filtered = [entry for entry in entries if entry.tenant_id == tenant_id]
        return [self._to_job_status(entry) for entry in filtered]

    def cancel_job(
        self, job_id: str, *, tenant_id: str, reason: str | None = None
    ) -> JobStatus | None:
        entry = self.ledger.get(job_id)
        if not entry or entry.tenant_id != tenant_id:
            return None
        updated = self.ledger.mark_cancelled(job_id, reason=reason)
        return self._to_job_status(updated)

    # ------------------------------------------------------------------
    # Adapter plugin helpers
    # ------------------------------------------------------------------
    def list_adapters(self, domain: str | None = None) -> list[AdapterMetadataView]:
        domain_enum = AdapterDomain(domain) if domain else None
        metadata = self.adapter_manager.list_metadata(domain=domain_enum)
        return [self._to_adapter_view(item) for item in metadata]

    def get_adapter_metadata(self, name: str) -> AdapterMetadataView | None:
        try:
            metadata = self.adapter_manager.get_metadata(name)
        except Exception:
            return None
        return self._to_adapter_view(metadata)

    def get_adapter_health(self, name: str) -> AdapterHealthView | None:
        if name not in {meta.name for meta in self.adapter_manager.list_metadata()}:
            return None
        healthy = self.adapter_manager.check_health(name)
        return AdapterHealthView(name=name, healthy=healthy)

    def get_adapter_config_schema(self, name: str) -> AdapterConfigSchemaView | None:
        metadata = self.get_adapter_metadata(name)
        if metadata is None:
            return None
        schema = metadata.config_schema or {}
        return AdapterConfigSchemaView(name=name, schema=schema)

    def _to_adapter_view(self, metadata) -> AdapterMetadataView:
        dataset = getattr(metadata, "dataset", None)
        extra = metadata.extra if hasattr(metadata, "extra") else {}
        return AdapterMetadataView(
            name=metadata.name,
            version=metadata.version,
            domain=metadata.domain,
            summary=metadata.summary,
            capabilities=list(metadata.capabilities),
            maintainer=metadata.maintainer,
            dataset=dataset,
            config_schema=dict(metadata.config_schema or {}),
            extra=dict(extra),
        )


_service: GatewayService | None = None
_ledger: JobLedger | None = None
_orchestrator: DagsterOrchestrator | None = None
_pipeline_loader: PipelineConfigLoader | None = None
_resilience_loader: ResiliencePolicyLoader | None = None
_stage_factory: StageFactory | None = None


def get_gateway_service() -> GatewayService:
    global _service, _ledger, _orchestrator, _pipeline_loader, _resilience_loader, _stage_factory
    if _service is None:
        events = EventStreamManager()
        _ledger = JobLedger()
        adapter_manager = get_plugin_manager()
        _pipeline_loader = PipelineConfigLoader()
        _resilience_loader = ResiliencePolicyLoader()
        stage_builders = build_default_stage_factory(adapter_manager)
        _stage_factory = StageFactory(stage_builders)
        _orchestrator = DagsterOrchestrator(
            _pipeline_loader,
            _resilience_loader,
            _stage_factory,
        )
        _service = GatewayService(
            events=events,
            orchestrator=_orchestrator,
            ledger=_ledger,
            adapter_manager=adapter_manager,
        )
    return _service
from ..services.mineru import MineruGpuUnavailableError, MineruOutOfMemoryError
