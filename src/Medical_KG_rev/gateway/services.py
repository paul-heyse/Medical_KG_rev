"""Protocol-agnostic gateway service layer."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import structlog
from Medical_KG_rev.chunking.exceptions import (
    ChunkerConfigurationError,
    ChunkingUnavailableError,
    InvalidDocumentError,
)

from ..adapters import AdapterDomain, AdapterPluginManager, get_plugin_manager
from ..adapters.plugins.models import AdapterRequest

from ..kg import ShaclValidator, ValidationError
from ..observability.metrics import observe_job_duration, record_business_event
from ..orchestration import (
    JobLedger,
    JobLedgerEntry,
    ParallelExecutor,
    PipelineConfigManager,
    PipelineContext,
    PipelineProfile,
    ProfileDetector,
    ProfileManager,
    QueryPipelineBuilder,
    StrategySpec,
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
from ..services.retrieval.chunking import ChunkingOptions, ChunkingService
from ..services.retrieval.reranker import CrossEncoderReranker
from ..services.retrieval.router import (
    RetrievalRouter,
    RetrievalStrategy,
    RouterMatch,
    RoutingRequest,
)
from ..utils.errors import ProblemDetail as PipelineProblemDetail
from ..validation import UCUMValidator
from ..validation.fhir import FHIRValidationError, FHIRValidator
from .models import (
    AdapterConfigSchemaView,
    AdapterHealthView,
    AdapterMetadataView,
    BatchError,
    BatchOperationResult,
    ChunkRequest,
    DocumentChunk,
    DocumentSummary,
    EmbeddingVector,
    EmbedRequest,
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


@dataclass
class GatewayService:
    """Coordinates business logic shared across protocols."""

    events: EventStreamManager
    orchestrator: DagsterOrchestrator
    ledger: JobLedger
    adapter_manager: AdapterPluginManager = field(default_factory=get_plugin_manager)
    chunker: ChunkingService = field(default_factory=ChunkingService)
    reranker: CrossEncoderReranker = field(default_factory=CrossEncoderReranker)
    shacl: ShaclValidator = field(default_factory=ShaclValidator.default)
    ucum: UCUMValidator = field(default_factory=UCUMValidator)
    fhir: FHIRValidator = field(default_factory=FHIRValidator)
    config_manager: PipelineConfigManager | None = None
    profile_manager: ProfileManager | None = None
    profile_detector: ProfileDetector | None = None
    query_pipeline_builder: QueryPipelineBuilder | None = None
    retrieval_router: RetrievalRouter | None = None
    _parallel_executor: ParallelExecutor | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._ensure_pipeline_components()

    def _ensure_pipeline_components(self) -> None:
        if self.config_manager is None:
            self.config_manager = PipelineConfigManager(Path("config/orchestration/pipelines.yaml"))
        if self._parallel_executor is None:
            self._parallel_executor = ParallelExecutor(max_workers=4)
        if self.profile_manager is None:
            config = self.config_manager.config
            self.profile_manager = ProfileManager(config, config.profiles)
        if self.profile_detector is None:
            profiles = self.profile_manager.list_profiles()
            default_profile = profiles[0] if profiles else "default"
            self.profile_detector = ProfileDetector(
                self.profile_manager,
                default_profile=default_profile,
            )
        if self.retrieval_router is None:
            max_workers = getattr(self._parallel_executor, "max_workers", 4)
            self.retrieval_router = RetrievalRouter(max_workers=max(1, max_workers))
        if self.query_pipeline_builder is None:
            self.query_pipeline_builder = self._build_query_builder()

    def _refresh_pipeline_components(self) -> None:
        if not self.config_manager:
            return
        updated = self.config_manager.reload()
        if not updated:
            return
        self.profile_manager = ProfileManager(updated, updated.profiles)
        profiles = self.profile_manager.list_profiles()
        default_profile = (
            self.profile_detector.default_profile
            if self.profile_detector and self.profile_detector.default_profile in profiles
            else (profiles[0] if profiles else "default")
        )
        max_workers = getattr(self._parallel_executor, "max_workers", 4)
        self.retrieval_router = RetrievalRouter(max_workers=max(1, max_workers))
        self.profile_detector = ProfileDetector(self.profile_manager, default_profile=default_profile)
        self.query_pipeline_builder = self._build_query_builder()

    def _build_query_builder(self) -> QueryPipelineBuilder:
        assert self.config_manager is not None
        assert self.profile_manager is not None
        assert self._parallel_executor is not None
        return QueryPipelineBuilder(
            config_manager=self.config_manager,
            profile_manager=self.profile_manager,
            parallel_executor=self._parallel_executor,
            strategies=self._strategy_registry(),
            rerank_runner=self._rerank_candidates,
        )

    def _strategy_registry(self) -> dict[str, StrategySpec]:
        assert self.retrieval_router is not None
        return {
            "bm25": StrategySpec.from_router(
                self.retrieval_router,
                self._synthetic_strategy("bm25", 0.92),
                timeout_ms=50,
            ),
            "dense": StrategySpec.from_router(
                self.retrieval_router,
                self._synthetic_strategy("dense", 0.88),
                timeout_ms=60,
            ),
            "splade": StrategySpec.from_router(
                self.retrieval_router,
                self._synthetic_strategy("splade", 0.86),
                timeout_ms=60,
            ),
        }

    def _executor_for_profile(self, profile: PipelineProfile):
        if self.query_pipeline_builder is None:
            self.query_pipeline_builder = self._build_query_builder()
        return self.query_pipeline_builder.executor_for_profile(profile)

    def _resolve_profile(
        self, explicit: str | None, metadata: Mapping[str, Any]
    ) -> PipelineProfile:
        if self.profile_detector is not None:
            return self.profile_detector.detect(explicit=explicit, metadata=metadata)
        if self.profile_manager is None:
            raise KeyError(explicit or "default")
        if explicit:
            return self.profile_manager.get(explicit)
        profiles = self.profile_manager.list_profiles()
        default_name = profiles[0] if profiles else "default"
        return self.profile_manager.get(default_name)

    def _synthetic_strategy(self, name: str, base_score: float) -> RetrievalStrategy:
        def handler(request: RoutingRequest) -> list[RouterMatch]:
            query = request.query or ""
            metadata = request.context if isinstance(request.context, Mapping) else {}
            profile = str(metadata.get("profile") or metadata.get("dataset") or "default")
            limit = max(1, int(request.top_k))
            matches: list[RouterMatch] = []
            for index in range(1, limit + 1):
                score = max(base_score - (index - 1) * 0.05, 0.0)
                doc_id = f"{name}-{profile}-{index}"
                matches.append(
                    RouterMatch(
                        id=doc_id,
                        score=score,
                        metadata={
                            "title": f"{name.upper()} result {index}",
                            "summary": f"Synthetic {name} result for '{query}'",
                            "source": name,
                            "profile": profile,
                        },
                    )
                )
            return matches

        return RetrievalStrategy(name=name, handler=handler)

    def _rerank_candidates(
        self,
        context: PipelineContext,
        candidates: Sequence[dict[str, Any]],
        options: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        reranked: list[dict[str, Any]] = []
        query = str(context.data.get("query", ""))
        top_n = int(options.get("rerank_candidates", min(len(candidates), 100)))
        for rank, candidate in enumerate(candidates[:top_n], start=1):
            updated = dict(candidate)
            document = dict(candidate.get("document", {}))
            document.setdefault("summary", f"Candidate for '{query}'")
            document.setdefault("source", document.get("source", "hybrid"))
            updated["document"] = document
            updated["score"] = float(candidate.get("score", 0.0)) + (top_n - rank) * 0.01
            reranked.append(updated)
        return reranked

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
        entry = self.ledger.idempotent_create(
            job_id=job_id,
            doc_key=doc_key,
            tenant_id=request.tenant_id,
            pipeline=pipeline_name,
            metadata={"dataset": dataset, "item": item, **metadata},
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

        correlation_id = uuid.uuid4().hex
        context = StageContext(
            tenant_id=request.tenant_id,
            doc_id=document_id,
            correlation_id=correlation_id,
            metadata={"dataset": dataset, **metadata},
            pipeline_name=pipeline_name,
            pipeline_version=topology.version,
        )

        domain = self._ingest_domain(topology) or AdapterDomain.BIOMEDICAL
        adapter_request = AdapterRequest(
            tenant_id=request.tenant_id,
            correlation_id=correlation_id,
            domain=domain,
            parameters={"dataset": dataset, "item": item},
        )
        payload = {"dataset": dataset, "item": item, "metadata": metadata}

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
            result_metadata = {"state": run_result.state}
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
        if raw_text is not None and not raw_text.strip():
            detail = ProblemDetail(
                title="Invalid document payload",
                status=400,
                type="https://httpstatuses.com/400",
                detail="Text payload must be a non-empty string",
                instance=f"/v1/chunk/{request.document_id}",
            )
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail)
        sample_text = raw_text or (
            "Population: Adults with hypertension. Intervention: ACE inhibitor administered daily. "
            "Comparison: Placebo. Outcome: Reduced systolic blood pressure at 12 weeks."
        )
        options = ChunkingOptions(
            strategy=request.strategy,
            max_tokens=request.chunk_size,
            overlap=request.overlap,
        )
        try:
            raw_chunks = self.chunker.chunk(
                request.tenant_id, request.document_id, sample_text, options
            )
        except InvalidDocumentError as exc:
            detail = ProblemDetail(
                title="Invalid document payload",
                status=400,
                type="https://httpstatuses.com/400",
                detail=str(exc),
                instance=f"/v1/chunk/{request.document_id}",
            )
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc
        except ChunkerConfigurationError as exc:
            detail = ProblemDetail(
                title="Chunker configuration invalid",
                status=422,
                type="https://httpstatuses.com/422",
                detail=str(exc),
                extensions={"valid_strategies": self.chunker.available_strategies()},
            )
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

    def embed(self, request: EmbedRequest) -> Sequence[EmbeddingVector]:
        job_id = self._new_job(request.tenant_id, "embed")
        embeddings: list[EmbeddingVector] = []
        for index, text in enumerate(request.inputs):
            vector = [round(math.sin(i + len(text)) % 1, 4) for i in range(8)]
            embeddings.append(
                EmbeddingVector(
                    id=f"emb-{index}",
                    vector=vector,
                    model=request.model,
                    metadata={"length": len(text), "normalized": request.normalize},
                )
            )
        self.ledger.update_metadata(job_id, {"embeddings": len(embeddings)})
        self._complete_job(job_id, payload={"embeddings": len(embeddings)})
        return embeddings

    def retrieve(self, request: RetrieveRequest) -> RetrievalResult:
        self._ensure_pipeline_components()
        self._refresh_pipeline_components()
        started = perf_counter()
        job_id = self._new_job(request.tenant_id, "retrieve")
        metadata: dict[str, Any] = {"filters": request.filters, **request.metadata}
        if request.profile:
            metadata["profile"] = request.profile
        try:
            profile = self._resolve_profile(request.profile, metadata)
        except KeyError as exc:
            detail = ProblemDetail(
                title="Unknown profile",
                status=400,
                type="https://httpstatuses.com/400",
                detail=str(exc),
            )
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc
        overrides: dict[str, dict[str, Any]] = {
            "final": {"top_k": request.top_k, "explain": request.explain},
            "rerank": {
                "enabled": request.rerank,
                "rerank_candidates": request.rerank_top_k,
                "allow_overflow": request.rerank_overflow,
            },
        }
        context = PipelineContext(
            tenant_id=request.tenant_id,
            operation="retrieve",
            data={
                "query": request.query,
                "filters": request.filters,
                "metadata": metadata,
                "profile": profile.name,
                "config": overrides,
                "explain": request.explain,
                "top_k": request.top_k,
            },
        )
        executor = self._executor_for_profile(profile)
        pipeline_name = getattr(executor.executor, "pipeline", profile.query)
        result_context = executor.run(context)

        documents: list[DocumentSummary] = []
        for item in result_context.data.get("results", [])[: request.top_k]:
            document_payload = dict(item.get("document", {}))
            doc_id = str(document_payload.get("id") or item.get("id"))
            metadata_payload = {
                key: value
                for key, value in document_payload.items()
                if key not in {"id", "title", "summary", "source"}
            }
            documents.append(
                DocumentSummary(
                    id=doc_id,
                    title=str(document_payload.get("title", doc_id)),
                    score=float(item.get("score", 0.0)),
                    summary=document_payload.get("summary"),
                    source=document_payload.get("source", pipeline_name),
                    metadata=metadata_payload,
                    explain=item.get("strategies") if request.explain else None,
                )
            )

        errors = [self._convert_problem(problem) for problem in result_context.errors]
        rerank_metrics = {
            "stage_timings_ms": {
                name: round(duration * 1000, 3)
                for name, duration in result_context.stage_timings.items()
            }
        }
        result = RetrievalResult(
            query=request.query,
            documents=documents,
            total=len(documents),
            rerank_metrics=rerank_metrics,
            pipeline_version=result_context.data.get("pipeline_version") or context.pipeline_version,
            partial=result_context.partial,
            degraded=result_context.data.get("degraded", False),
            errors=errors,
            stage_timings={
                name: round(duration, 6)
                for name, duration in result_context.stage_timings.items()
            },
        )

        duration = perf_counter() - started
        observe_job_duration("retrieve", duration)
        record_business_event("retrieval_requests", request.tenant_id)
        if result.total:
            record_business_event("documents_retrieved", request.tenant_id)

        ledger_metadata = {
            "documents": result.total,
            "pipeline_version": result.pipeline_version,
            "partial": result.partial,
            "degraded": result.degraded,
        }
        if result_context.degradation_events:
            ledger_metadata["degradation_events"] = list(result_context.degradation_events)

        if not result.documents and errors:
            problem = errors[0]
            self._fail_job(job_id, problem.detail or problem.title)
            raise GatewayError(problem)

        self.ledger.update_metadata(job_id, ledger_metadata)
        if result.partial:
            partial_payload = dict(ledger_metadata)
            partial_payload["status"] = "partial"
            self._complete_job(job_id, payload=partial_payload)
        else:
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
