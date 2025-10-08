"""Protocol-agnostic gateway service layer."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from time import perf_counter
from typing import Any

import structlog
from Medical_KG_rev.chunking.exceptions import InvalidDocumentError
from aiolimiter import AsyncLimiter
from pybreaker import CircuitBreaker

from ..adapters import AdapterDomain, AdapterPluginManager, get_plugin_manager
from ..adapters.plugins.models import AdapterRequest

from ..kg import ShaclValidator, ValidationError
from ..auth.scopes import Scopes
from ..observability.events import record_business_event
from ..observability.metrics import observe_job_duration
from ..config.settings import get_settings
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
    build_stage_factory,
    create_stage_plugin_manager,
    submit_to_dagster,
)
from ..orchestration.dagster.stages import create_default_pipeline_resource
from ..orchestration.stages.contracts import StageContext
from ..services.extraction.templates import TemplateValidationError, validate_template
from ..services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from ..services.embedding.policy import (
    NamespaceAccessPolicy,
    NamespacePolicySettings,
    build_policy_chain,
)
from ..services.embedding.persister import (
    EmbeddingPersister,
    PersistenceContext,
    PersisterRuntimeSettings,
    build_persister,
)
from ..services.embedding.registry import EmbeddingModelRegistry
from ..services.embedding.telemetry import EmbeddingTelemetry, StandardEmbeddingTelemetry
from ..services.retrieval.chunking import ChunkingService
from ..services.retrieval.chunking_command import ChunkCommand
from .chunking import ChunkingErrorTranslator
from ..utils.errors import ProblemDetail as PipelineProblemDetail
from ..validation import UCUMValidator
from ..validation.fhir import FHIRValidationError, FHIRValidator
from Medical_KG_rev.embeddings.ports import (
    EmbeddingRecord,
    EmbeddingRequest as AdapterEmbeddingRequest,
from ..utils.errors import ProblemDetail as PipelineProblemDetail
from ..validation import UCUMValidator
from ..validation.fhir import FHIRValidationError, FHIRValidator
from .coordinators import (
    ChunkingCoordinator,
    ChunkingRequest as CoordinatorChunkingRequest,
    ChunkingResult,
    CoordinatorConfig,
    CoordinatorError,
    EmbeddingCoordinator,
    EmbeddingRequest as CoordinatorEmbeddingRequest,
    EmbeddingResult,
    JobLifecycleManager,
)
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
    NamespacePolicyDiagnosticsView,
    NamespacePolicyHealthView,
    NamespacePolicyMetricsView,
    NamespacePolicySettingsView,
    NamespacePolicyStatus,
    NamespacePolicyUpdateRequest,
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


def _build_stage_factory(
    manager: AdapterPluginManager | None = None,
    ledger: JobLedger | None = None,
) -> StageFactory:
    adapter_manager = manager or get_plugin_manager()
    job_ledger = ledger or JobLedger()
    pipeline_resource = create_default_pipeline_resource()
    return build_stage_factory(adapter_manager, pipeline_resource, job_ledger)
    *,
    job_ledger: JobLedger | None = None,
) -> StageFactory:
    plugin_manager = create_stage_plugin_manager(
        manager or get_plugin_manager(),
        job_ledger=job_ledger,
    )
    return StageFactory(plugin_manager)


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
    chunking_error_translator: ChunkingErrorTranslator | None = None
    retriever: HaystackRetriever = field(default_factory=HaystackRetriever)
    shacl: ShaclValidator = field(default_factory=ShaclValidator.default)
    ucum: UCUMValidator = field(default_factory=UCUMValidator)
    fhir: FHIRValidator = field(default_factory=FHIRValidator)
    embedding_registry: EmbeddingModelRegistry = field(default_factory=EmbeddingModelRegistry)
    namespace_registry: EmbeddingNamespaceRegistry | None = None
    namespace_policy: NamespaceAccessPolicy | None = None
    namespace_policy_settings: NamespacePolicySettings | None = None
    embedding_persister: EmbeddingPersister | None = None
    embedding_persister_settings: PersisterRuntimeSettings | None = None
    embedding_telemetry: EmbeddingTelemetry | None = None
    job_lifecycle: JobLifecycleManager | None = None
    chunking_coordinator: ChunkingCoordinator | None = None
    embedding_coordinator: EmbeddingCoordinator | None = None

    def __post_init__(self) -> None:
        self.job_lifecycle = JobLifecycleManager(self.ledger, self.events)
        if self.stage_factory is None:
            self.stage_factory = _build_stage_factory(
                self.adapter_manager,
                job_ledger=self.ledger,
            )
            self.stage_factory = _build_stage_factory(self.adapter_manager, self.ledger)
            
        if self.chunker is None:
            self.chunker = ChunkingService(stage_factory=self.stage_factory)
        if self.chunking_error_translator is None:
            self.chunking_error_translator = ChunkingErrorTranslator(
                available_strategies=self.chunker.available_strategies,
            )
        if self.namespace_registry is None:
            self.namespace_registry = self.embedding_registry.namespace_registry
        if self.embedding_telemetry is None:
            self.embedding_telemetry = StandardEmbeddingTelemetry()
        if self.namespace_policy is None:
            if self.namespace_policy_settings is None:
                self.namespace_policy_settings = NamespacePolicySettings()
            self.namespace_policy = self._build_namespace_policy()
        self.namespace_policy_settings = self.namespace_policy.settings
        if self.embedding_persister is None:
            router = getattr(self.embedding_registry, "storage_router", None)
            if router is None:
                raise RuntimeError("Embedding registry missing storage router")
            settings = self.embedding_persister_settings or PersisterRuntimeSettings()
            self.embedding_persister = build_persister(
                router,
                telemetry=self.embedding_telemetry,
                settings=settings,
            )
            self.embedding_persister_settings = settings
        if self.chunking_coordinator is None:
            self.chunking_coordinator = ChunkingCoordinator(
                lifecycle=self.job_lifecycle,
                chunker=self.chunker,
                config=self._build_coordinator_config("chunking"),
            )
        if self.embedding_coordinator is None:
            if self.namespace_registry is None:
                raise RuntimeError("Namespace registry not initialised")
            if self.embedding_persister is None:
                raise RuntimeError("Embedding persister not initialised")
            if self.namespace_policy is None:
                raise RuntimeError("Namespace policy not initialised")
            self.embedding_coordinator = EmbeddingCoordinator(
                lifecycle=self.job_lifecycle,
                registry=self.embedding_registry,
                namespace_registry=self.namespace_registry,
                policy=self.namespace_policy,
                persister=self.embedding_persister,
                telemetry=self.embedding_telemetry,
                config=self._build_coordinator_config("embedding", retry_attempts=4),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_namespace_policy(self) -> NamespaceAccessPolicy:
        if self.namespace_registry is None or self.namespace_policy is None:
            raise RuntimeError("Namespace policy not initialised")
        return self.namespace_policy

    def _build_namespace_policy(self) -> NamespaceAccessPolicy:
        if self.namespace_registry is None:
            if self.embedding_registry is None:
                raise RuntimeError("Namespace registry not initialised")
            self.namespace_registry = self.embedding_registry.namespace_registry
        settings = self.namespace_policy_settings or NamespacePolicySettings()
        policy = build_policy_chain(
            self.namespace_registry,
            telemetry=self.embedding_telemetry,
            settings=settings,
            dry_run=settings.dry_run,
        )
        return policy

    def _build_coordinator_config(self, name: str, *, retry_attempts: int = 3) -> CoordinatorConfig:
        return CoordinatorConfig(
            name=name,
            retry_attempts=retry_attempts,
            retry_wait_base=0.2,
            retry_wait_max=2.0,
            breaker=CircuitBreaker(f"{name}-coordinator"),
            limiter=AsyncLimiter(10, 1),
        )

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
        metadata = {
            key: value
            for key, value in options_payload.items()
            if key != "text"
        }
        chunker = self.chunker or ChunkingService(stage_factory=self.stage_factory)
        translator = self.chunking_error_translator or ChunkingErrorTranslator(
            available_strategies=chunker.available_strategies,
        )
        command: ChunkCommand | None = None
        try:
            command = ChunkCommand.from_request(
                request,
                text=raw_text,
                metadata=metadata,
            ).with_context(job_id=job_id, endpoint="gateway")
            raw_chunks = chunker.chunk(command)
        except Exception as exc:
            try:
                translated = translator.translate(exc, command=command)
            except Exception:
                message = str(exc) or "Runtime error during chunking"
                self._fail_job(job_id, message)
                raise
            self._fail_job(job_id, translated.detail.detail or translated.detail.title)
            raise GatewayError(translated.detail) from exc
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
        if self.chunking_coordinator is None:
            raise RuntimeError("Chunking coordinator not initialised")

        coordinator_request = CoordinatorChunkingRequest(
            tenant_id=request.tenant_id,
            correlation_id=None,
            metadata={"document_id": request.document_id},
            document_id=request.document_id,
            strategy=request.strategy,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            options=request.options,
        )
        try:
            result: ChunkingResult = self.chunking_coordinator(coordinator_request)
        except CoordinatorError as exc:
            detail = exc.context.get("problem") if isinstance(exc.context, dict) else None
            if isinstance(detail, ProblemDetail):
                raise GatewayError(detail) from exc
            raise
        observe_job_duration("chunk", result.duration_s)
        return list(result.chunks)

    def embed(self, request: EmbedRequest) -> EmbeddingResponse:
        if self.embedding_coordinator is None:
            raise RuntimeError("Embedding coordinator not initialised")
        if self.namespace_registry is None or self.namespace_policy is None or self.embedding_persister is None:
            raise RuntimeError("Embedding components not initialised")

        started = perf_counter()
        options = request.options or EmbeddingOptions()
        namespace = request.namespace

        try:
            decision = self.namespace_policy.evaluate(
                namespace=namespace,
                tenant_id=request.tenant_id,
                required_scope=Scopes.EMBED_WRITE,
            )
        except Exception as exc:  # pragma: no cover - defensive
            detail = ProblemDetail(
                title="Namespace policy failure",
                status=500,
                type="https://httpstatuses.com/500",
                detail=str(exc),
            )
            raise GatewayError(detail) from exc

        if not decision.allowed:
            detail = ProblemDetail(
                title="Namespace access denied",
                status=403,
                type="https://httpstatuses.com/403",
                detail=decision.reason or "Access to namespace not permitted",
            )
            raise GatewayError(detail)

        config = decision.config or self.namespace_registry.get(namespace)

        job_id = self._new_job(request.tenant_id, "embed")
        correlation_id = uuid.uuid4().hex
        model_name = options.model or config.model_id

        if self.embedding_telemetry:
            self.embedding_telemetry.record_embedding_started(
                namespace=namespace,
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

        coordinator_request = CoordinatorEmbeddingRequest(
            tenant_id=request.tenant_id,
            correlation_id=None,
            metadata={"namespace": request.namespace},
            namespace=request.namespace,
            texts=request.texts,
            options=request.options,
        )
        try:
            result: EmbeddingResult = self.embedding_coordinator(coordinator_request)
        except CoordinatorError as exc:
            detail = exc.context.get("problem") if isinstance(exc.context, dict) else None
            if isinstance(detail, ProblemDetail):
                raise GatewayError(detail) from exc
            raise
        observe_job_duration("embed", result.duration_s)
        if result.response is None:
            raise RuntimeError("Embedding coordinator returned no response")
        return result.response
            records = embedder.embed_documents(adapter_request)
        except Exception as exc:  # pragma: no cover - network/library error
            detail = ProblemDetail(
                title="Embedding failed",
                status=502,
                type="https://httpstatuses.com/502",
                detail=str(exc),
            )
            self._fail_job(job_id, detail.detail or detail.title)
            if self.embedding_telemetry:
                self.embedding_telemetry.record_embedding_failure(
                    namespace=namespace,
                    tenant_id=request.tenant_id,
                    error=exc,
                )
            raise GatewayError(detail) from exc

        embeddings: list[EmbeddingVector] = []
        prepared_records: list[EmbeddingRecord] = []
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
                    namespace=namespace,
                    kind=updated_record.kind,
                    dimension=dimension,
                    vector=values if updated_record.kind != "sparse" else None,
                    terms=terms,
                    metadata=meta,
                )
            )

        persistence_context = PersistenceContext(
            tenant_id=request.tenant_id,
            namespace=namespace,
            model=model_name,
            provider=config.provider,
            job_id=job_id,
            correlation_id=correlation_id,
            normalize=bool(options.normalize),
        )
        persistence_report = self.embedding_persister.persist_batch(prepared_records, persistence_context)

        payload = {
            "embeddings": len(embeddings),
            "model": model_name,
            "namespace": namespace,
            "provider": config.provider,
            "tenant_id": request.tenant_id,
            "persisted": persistence_report.persisted,
        }
        self.ledger.update_metadata(job_id, payload)
        self._complete_job(job_id, payload=payload)

        if self.embedding_telemetry:
            self.embedding_telemetry.record_embedding_completed(
                namespace=namespace,
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

    def namespace_policy_status(self) -> NamespacePolicyStatus:
        policy = self._require_namespace_policy()
        settings_view = NamespacePolicySettingsView(**asdict(policy.settings))
        return NamespacePolicyStatus(
            settings=settings_view,
            stats=dict(policy.stats()),
            operational=dict(policy.operational_metrics()),
        )

    def update_namespace_policy(
        self, updates: NamespacePolicyUpdateRequest
    ) -> NamespacePolicyStatus:
        policy = self._require_namespace_policy()
        current = asdict(policy.settings)
        changes: dict[str, object] = {}
        if updates.cache_ttl_seconds is not None:
            value = float(updates.cache_ttl_seconds)
            changes["cache_ttl_seconds"] = value
            current["cache_ttl_seconds"] = value
        if updates.max_cache_entries is not None:
            value = int(updates.max_cache_entries)
            changes["max_cache_entries"] = value
            current["max_cache_entries"] = value
        dry_run_changed = False
        if updates.dry_run is not None and updates.dry_run != policy.settings.dry_run:
            current["dry_run"] = bool(updates.dry_run)
            dry_run_changed = True

        if dry_run_changed:
            self.namespace_policy_settings = NamespacePolicySettings(**current)
            self.namespace_policy = self._build_namespace_policy()
            self.namespace_policy_settings = self.namespace_policy.settings
        elif changes:
            policy.update_settings(**changes)
            self.namespace_policy_settings = policy.settings

        return self.namespace_policy_status()

    def namespace_policy_diagnostics(self) -> NamespacePolicyDiagnosticsView:
        policy = self._require_namespace_policy()
        snapshot = policy.debug_snapshot()
        settings_view = NamespacePolicySettingsView(**asdict(policy.settings))
        cache_entries = [
            f"{namespace}:{tenant}:{scope}"
            for namespace, tenant, scope in snapshot.get("cache_keys", [])
        ]
        return NamespacePolicyDiagnosticsView(
            settings=settings_view,
            cache_keys=cache_entries,
            stats=dict(snapshot.get("stats", {})),
        )

    def namespace_policy_health(self) -> NamespacePolicyHealthView:
        policy = self._require_namespace_policy()
        return NamespacePolicyHealthView(**policy.health_status())

    def namespace_policy_metrics(self) -> NamespacePolicyMetricsView:
        policy = self._require_namespace_policy()
        return NamespacePolicyMetricsView(metrics=dict(policy.operational_metrics()))

    def invalidate_namespace_policy_cache(self, namespace: str | None = None) -> None:
        policy = self._require_namespace_policy()
        policy.invalidate(namespace)

    def validate_namespace_texts(
        self,
        *,
        tenant_id: str,
        namespace: str,
        texts: Sequence[str],
    ) -> NamespaceValidationResponse:
        if self.namespace_registry is None or self.namespace_policy is None:
            raise RuntimeError("Namespace policy not initialised")
        decision = self.namespace_policy.evaluate(
            namespace=namespace,
            tenant_id=tenant_id,
            required_scope=Scopes.EMBED_READ,
        )
        if not decision.allowed:
            detail = ProblemDetail(
                title="Namespace access denied",
                status=403,
                type="https://httpstatuses.com/403",
                detail=decision.reason or "Access to namespace not permitted",
            )
            raise GatewayError(detail)

        config = decision.config or self.namespace_registry.get(namespace)
        max_tokens = config.max_tokens
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
        pipeline_resource = create_default_pipeline_resource()
        _stage_factory = build_stage_factory(adapter_manager, pipeline_resource, _ledger)
        stage_plugin_manager = create_stage_plugin_manager(
            adapter_manager,
            job_ledger=_ledger,
        )
        _stage_factory = StageFactory(stage_plugin_manager)
        _orchestrator = DagsterOrchestrator(
            _pipeline_loader,
            _resilience_loader,
            _stage_factory,
            plugin_manager=adapter_manager,
            job_ledger=_ledger,
            pipeline_resource=pipeline_resource,
        )
        settings = get_settings()
        embedding_cfg = settings.embedding
        policy_settings = NamespacePolicySettings(
            cache_ttl_seconds=embedding_cfg.policy.cache_ttl_seconds,
            max_cache_entries=embedding_cfg.policy.max_cache_entries,
            dry_run=embedding_cfg.policy.dry_run,
        )
        persister_settings = PersisterRuntimeSettings(
            backend=embedding_cfg.persister.backend,
            cache_limit=embedding_cfg.persister.cache_limit,
            hybrid_backends=dict(embedding_cfg.persister.hybrid_backends),
        )
        settings = get_settings()
        embedding_cfg = settings.embedding
        policy_settings = NamespacePolicySettings(
            cache_ttl_seconds=embedding_cfg.policy.cache_ttl_seconds,
            max_cache_entries=embedding_cfg.policy.max_cache_entries,
            dry_run=embedding_cfg.policy.dry_run,
        )
        persister_settings = PersisterRuntimeSettings(
            backend=embedding_cfg.persister.backend,
            cache_limit=embedding_cfg.persister.cache_limit,
            hybrid_backends=dict(embedding_cfg.persister.hybrid_backends),
        )
        settings = get_settings()
        embedding_cfg = settings.embedding
        policy_settings = NamespacePolicySettings(
            cache_ttl_seconds=embedding_cfg.policy.cache_ttl_seconds,
            max_cache_entries=embedding_cfg.policy.max_cache_entries,
            dry_run=embedding_cfg.policy.dry_run,
        )
        persister_settings = PersisterRuntimeSettings(
            backend=embedding_cfg.persister.backend,
            cache_limit=embedding_cfg.persister.cache_limit,
            hybrid_backends=dict(embedding_cfg.persister.hybrid_backends),
        )
        settings = get_settings()
        embedding_cfg = settings.embedding
        policy_settings = NamespacePolicySettings(
            cache_ttl_seconds=embedding_cfg.policy.cache_ttl_seconds,
            max_cache_entries=embedding_cfg.policy.max_cache_entries,
            dry_run=embedding_cfg.policy.dry_run,
        )
        persister_settings = PersisterRuntimeSettings(
            backend=embedding_cfg.persister.backend,
            cache_limit=embedding_cfg.persister.cache_limit,
            hybrid_backends=dict(embedding_cfg.persister.hybrid_backends),
        )
        settings = get_settings()
        embedding_cfg = settings.embedding
        policy_settings = NamespacePolicySettings(
            cache_ttl_seconds=embedding_cfg.policy.cache_ttl_seconds,
            max_cache_entries=embedding_cfg.policy.max_cache_entries,
            dry_run=embedding_cfg.policy.dry_run,
        )
        persister_settings = PersisterRuntimeSettings(
            backend=embedding_cfg.persister.backend,
            cache_limit=embedding_cfg.persister.cache_limit,
            hybrid_backends=dict(embedding_cfg.persister.hybrid_backends),
        )
        _service = GatewayService(
            events=events,
            orchestrator=_orchestrator,
            ledger=_ledger,
            adapter_manager=adapter_manager,
            namespace_policy_settings=policy_settings,
            embedding_persister_settings=persister_settings,
        )
    return _service
from ..services.mineru import MineruGpuUnavailableError, MineruOutOfMemoryError
