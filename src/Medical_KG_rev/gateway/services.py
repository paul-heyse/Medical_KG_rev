"""Protocol-agnostic gateway service layer."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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

from ..chunking.models import Chunk
from ..kg import ShaclValidator, ValidationError
from ..observability.metrics import (
    observe_job_duration,
    record_business_event,
    record_chunking_failure,
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
from ..orchestration.dagster.configuration import StageDefinition
from ..orchestration.stages.contracts import EmbeddingBatch, StageContext
from ..services.extraction.templates import TemplateValidationError, validate_template
from ..services.retrieval.chunking import ChunkingOptions, ChunkingService
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

    def __post_init__(self) -> None:
        if self.stage_factory is None:
            self.stage_factory = _build_stage_factory(self.adapter_manager)
        if self.chunker is None:
            self.chunker = ChunkingService(stage_factory=self.stage_factory)

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

    def _resolve_stage(self, stage_type: str) -> object:
        if self.stage_factory is None:
            self.stage_factory = _build_stage_factory(self.adapter_manager)
        definition = StageDefinition(name=f"gateway-{stage_type}", type=stage_type)
        return self.stage_factory.resolve(self._PIPELINE_NAME, definition)

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

    def embed(self, request: EmbedRequest) -> Sequence[EmbeddingVector]:
        job_id = self._new_job(request.tenant_id, "embed")
        stage = self._resolve_stage("embed")
        context = StageContext(
            tenant_id=request.tenant_id,
            correlation_id=uuid.uuid4().hex,
            metadata={"model": request.model, "normalize": request.normalize},
            pipeline_name=self._PIPELINE_NAME,
            pipeline_version=self._PIPELINE_VERSION,
        )
        chunks: list[Chunk] = []
        for index, text in enumerate(request.inputs):
            if not isinstance(text, str) or not text.strip():
                detail = ProblemDetail(
                    title="Invalid embedding input",
                    status=400,
                    type="https://httpstatuses.com/400",
                    detail="Embedding inputs must be non-empty strings",
                )
                self._fail_job(job_id, detail.detail or detail.title)
                raise GatewayError(detail)
            body = text.strip()
            chunk_id = f"{job_id}:chunk:{index}"
            doc_id = f"{job_id}:doc"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    tenant_id=request.tenant_id,
                    body=body,
                    title_path=(),
                    section=None,
                    start_char=0,
                    end_char=len(body),
                    granularity="document",
                    chunker="gateway.manual",
                    chunker_version="1.0.0",
                    meta={"input_index": index, "job_id": job_id},
                )
            )
        try:
            batch: EmbeddingBatch = stage.execute(context, chunks)
        except Exception as exc:
            detail = ProblemDetail(
                title="Embedding failed",
                status=502,
                type="https://httpstatuses.com/502",
                detail=str(exc),
            )
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc
        embeddings: list[EmbeddingVector] = []
        for vector in batch.vectors:
            values = list(vector.values)
            if request.normalize and values:
                magnitude = math.sqrt(sum(value * value for value in values))
                if magnitude > 0:
                    values = [value / magnitude for value in values]
            metadata = dict(vector.metadata)
            metadata.setdefault("model", batch.model)
            metadata.setdefault("pipeline", f"{self._PIPELINE_NAME}:{self._PIPELINE_VERSION}")
            embeddings.append(
                EmbeddingVector(
                    id=vector.id,
                    vector=values,
                    model=batch.model,
                    metadata=metadata,
                )
            )
        payload = {"embeddings": len(embeddings), "model": batch.model}
        self.ledger.update_metadata(job_id, payload)
        self._complete_job(job_id, payload=payload)
        if embeddings:
            record_business_event("embeddings_generated", request.tenant_id)
        return embeddings

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
