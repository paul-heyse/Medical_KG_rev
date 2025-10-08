"""Protocol-agnostic gateway service layer."""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import json

import structlog
from Medical_KG_rev.chunking.exceptions import (
    ChunkerConfigurationError,
    ChunkingUnavailableError,
    InvalidDocumentError,
)

from ..adapters import AdapterDomain, AdapterPluginManager, get_plugin_manager
from ..auth.context import SecurityContext
from ..config import AppSettings, RerankingSettings
from ..kg import ShaclValidator, ValidationError
from ..observability.metrics import observe_job_duration, record_business_event
from ..orchestration import JobLedger, JobLedgerEntry, Orchestrator
from ..orchestration.kafka import KafkaClient
from ..orchestration.worker import IngestWorker, MappingWorker, WorkerBase
from ..services.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationRunner,
    TestSetManager,
    build_test_set,
)
from ..services.extraction.templates import TemplateValidationError, validate_template
from ..services.retrieval import FAISSIndex, OpenSearchClient, RetrievalService
from ..services.retrieval.chunking import ChunkingOptions, ChunkingService
from ..services.retrieval.routing import IntentClassifier, QueryIntent
from ..services.reranking import PipelineSettings
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
    EvaluationRequest,
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

RETRIEVAL_PIPELINE_VERSION = "retrieval-service:v1"


class GatewayError(RuntimeError):
    """Domain specific exception carrying problem detail information."""

    def __init__(self, detail: ProblemDetail):
        super().__init__(detail.title)
        self.detail = detail


def _is_table_document(metadata: Mapping[str, Any] | None) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    if bool(metadata.get("is_table")):
        return True
    segment = str(metadata.get("segment_type", ""))
    if segment.lower() == "table":
        return True
    intent_hint = str(metadata.get("intent_hint", "")).lower()
    if intent_hint in {"ae", "adverse_events", "table"}:
        return True
    if "table_html" in metadata:
        return True
    return False


@dataclass
class GatewayService:
    """Coordinates business logic shared across protocols."""

    events: EventStreamManager
    orchestrator: Orchestrator
    ledger: JobLedger
    adapter_manager: AdapterPluginManager = field(default_factory=get_plugin_manager)
    workers: list[WorkerBase] = field(default_factory=list)
    chunker: ChunkingService = field(default_factory=ChunkingService)
    intent_classifier: IntentClassifier = field(default_factory=IntentClassifier)
    shacl: ShaclValidator = field(default_factory=ShaclValidator.default)
    ucum: UCUMValidator = field(default_factory=UCUMValidator)
    fhir: FHIRValidator = field(default_factory=FHIRValidator)
    app_settings: AppSettings | None = None
    reranking_settings: RerankingSettings | None = None
    opensearch_client: OpenSearchClient | None = None
    faiss_index: FAISSIndex | None = None
    retrieval_service: RetrievalService | None = None
    test_set_manager: TestSetManager = field(default_factory=TestSetManager)
    _evaluation_runner: EvaluationRunner | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._ensure_retrieval_components()

    def _ensure_retrieval_components(self) -> None:
        if self.app_settings is None:
            self.app_settings = AppSettings()
        if self.reranking_settings is None:
            self.reranking_settings = self.app_settings.reranking
        if self.opensearch_client is None:
            self.opensearch_client = OpenSearchClient()
        if self.retrieval_service is None:
            pipeline_cfg = getattr(self.reranking_settings, "pipeline", None)
            pipeline_settings = PipelineSettings(
                retrieve_candidates=(
                    int(pipeline_cfg.retrieve_candidates)
                    if pipeline_cfg is not None
                    else 1000
                ),
                rerank_candidates=max(
                    200,
                    int(pipeline_cfg.rerank_candidates) if pipeline_cfg is not None else 100,
                ),
                return_top_k=max(
                    50,
                    int(pipeline_cfg.return_top_k) if pipeline_cfg is not None else 10,
                ),
            )
            self.retrieval_service = RetrievalService(
                self.opensearch_client,
                self.faiss_index,
                reranking_settings=self.reranking_settings,
                pipeline_settings=pipeline_settings,
            )

    def _index_for_request(
        self, request: RetrieveRequest, metadata: Mapping[str, Any]
    ) -> str:
        candidate = metadata.get("index")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if request.profile:
            return f"{request.profile}-chunks"
        dataset = metadata.get("dataset") or metadata.get("source")
        if isinstance(dataset, str) and dataset.strip():
            return f"{dataset.strip()}-chunks"
        return "chunks"

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

    def ingest(self, dataset: str, request: IngestionRequest) -> BatchOperationResult:
        started = perf_counter()
        statuses: list[OperationStatus] = []
        for item in request.items:
            metadata = dict(request.metadata)
            if request.profile:
                metadata.setdefault("profile", request.profile)
            entry = self.orchestrator.submit_job(
                tenant_id=request.tenant_id,
                dataset=dataset,
                item=item,
                priority=request.priority,
                metadata=metadata,
            )
            duplicate = bool(entry.metadata.get("duplicate"))
            message = (
                f"Duplicate job for {dataset}" if duplicate else f"Queued pipeline {entry.pipeline}"
            )
            error = None
            http_status = 202
            status_value = entry.status
            if duplicate:
                http_status = 409
                status_value = "failed"
                error = BatchError(
                    code="duplicate",
                    message="Document already queued",
                    details={"dataset": dataset},
                )
            statuses.append(
                OperationStatus(
                    job_id=entry.job_id,
                    status=status_value,
                    message=message,
                    metadata={
                        "dataset": dataset,
                        "pipeline": entry.pipeline,
                        "doc_key": entry.doc_key,
                        "duplicate": duplicate,
                    },
                    http_status=http_status,
                    error=error,
                )
            )
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
                    model=request.model,
                    namespace=request.namespace,
                    kind="single_vector",
                    dimension=len(vector),
                    vector=vector,
                    metadata={
                        "length": len(text),
                        "normalized": request.normalize,
                        "namespace": request.namespace,
                    },
                )
            )
        self.ledger.update_metadata(job_id, {"embeddings": len(embeddings)})
        self._complete_job(
            job_id,
            payload={
                "embeddings": len(embeddings),
                "namespace": request.namespace,
                "model": request.model,
            },
        )
        return embeddings

    def retrieve(self, request: RetrieveRequest) -> RetrievalResult:
        self._ensure_retrieval_components()
        started = perf_counter()
        job_id = self._new_job(request.tenant_id, "retrieve")
        metadata: dict[str, Any] = {"filters": request.filters, **request.metadata}
        if request.profile:
            metadata.setdefault("profile", request.profile)

        classification = self.intent_classifier.classify(
            request.query, override=request.query_intent
        )
        intent_payload = {
            "detected": classification.intent.value,
            "confidence": classification.confidence,
            "override": classification.override.value if classification.override else None,
            "matched_patterns": list(classification.matched_patterns),
            "table_only": request.table_only,
        }
        metadata.setdefault("intent", intent_payload.copy())

        index_name = self._index_for_request(request, metadata)
        rerank_flag = request.rerank if request.rerank is not None else True
        retrieval_service = self.retrieval_service
        assert retrieval_service is not None
        security_context = SecurityContext(
            subject="gateway",
            tenant_id=request.tenant_id,
            scopes={"*", "retrieve:read"},
        )

        try:
            service_results = retrieval_service.search(
                index=index_name,
                query=request.query,
                filters=request.filters,
                k=request.top_k,
                rerank=rerank_flag,
                rerank_model=request.rerank_model,
                context=security_context,
                explain=request.explain,
            )
        except Exception as exc:  # pragma: no cover - defensive guardrail
            detail = ProblemDetail(
                title="Retrieval failed",
                status=500,
                type="https://medical-kg/errors/retrieval",
                detail=str(exc),
            )
            self._fail_job(job_id, detail.detail or detail.title)
            raise GatewayError(detail) from exc

        documents: list[DocumentSummary] = []
        table_documents: list[DocumentSummary] = []
        rerank_metadata: dict[str, Any] = {}
        stage_timings: dict[str, float] = {}
        component_errors: list[str] = []

        for result in service_results:
            metadata_payload = dict(result.metadata)
            if metadata_payload:
                payload_rerank = metadata_payload.get("reranking")
                if isinstance(payload_rerank, Mapping):
                    rerank_metadata = dict(payload_rerank)
                components = metadata_payload.get("components")
                if isinstance(components, Mapping):
                    errors = components.get("errors", [])
                    if isinstance(errors, Sequence):
                        component_errors.extend(str(error) for error in errors if error)
                metrics_payload = metadata_payload.get("pipeline_metrics")
                if isinstance(metrics_payload, Mapping):
                    timings = metrics_payload.get("timing")
                    if isinstance(timings, Mapping):
                        for name, duration in timings.items():
                            try:
                                stage_timings[name] = float(duration)
                            except (TypeError, ValueError):
                                continue
            title = str(
                metadata_payload.get("title")
                or metadata_payload.get("document_id")
                or metadata_payload.get("id")
                or result.id
            )
            summary_text = metadata_payload.get("summary")
            source = str(
                metadata_payload.get("source")
                or metadata_payload.get("strategy")
                or metadata_payload.get("profile")
                or "hybrid"
            )
            explain_payload = (
                metadata_payload.get("component_scores") if request.explain else None
            )
            score = (
                float(result.rerank_score)
                if result.rerank_score is not None
                else float(result.retrieval_score)
            )
            document = DocumentSummary(
                id=result.id,
                title=title,
                score=score,
                summary=summary_text if isinstance(summary_text, str) else None,
                source=source,
                metadata=metadata_payload,
                explain=explain_payload,
            )
            documents.append(document)
            if _is_table_document(metadata_payload):
                table_documents.append(document)

        if request.table_only:
            if table_documents:
                selected_documents = table_documents[: request.top_k]
            else:
                intent_payload.setdefault("warnings", []).append("table_only_fallback")
                selected_documents = documents[: request.top_k]
        else:
            selected_documents = documents[: request.top_k]

        unique_errors = sorted({error for error in component_errors if error})
        problem_details = [
            ProblemDetail(
                type="https://medical-kg/errors/retrieval-component",
                title="Retrieval component error",
                status=502,
                detail=error,
            )
            for error in unique_errors
        ]

        stage_timings_ms = {
            name: round(value * 1000.0, 3) for name, value in stage_timings.items()
        }
        rerank_metrics = dict(rerank_metadata)
        if stage_timings_ms:
            rerank_metrics.setdefault("stage_timings_ms", stage_timings_ms)
        if request.rerank_model:
            rerank_metrics.setdefault("requested_model", request.rerank_model)
        if unique_errors:
            rerank_metrics.setdefault("component_errors", unique_errors)

        result = RetrievalResult(
            query=request.query,
            documents=selected_documents,
            total=len(selected_documents),
            rerank_metrics=rerank_metrics,
            pipeline_version=RETRIEVAL_PIPELINE_VERSION,
            partial=bool(unique_errors),
            degraded=bool(unique_errors),
            errors=problem_details,
            stage_timings={name: round(value, 6) for name, value in stage_timings.items()},
            intent=intent_payload,
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
            "index": index_name,
        }
        if unique_errors:
            ledger_metadata["component_errors"] = unique_errors

        if not result.documents and problem_details:
            problem = problem_details[0]
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

    def evaluate_retrieval(self, request: EvaluationRequest) -> EvaluationResult:
        self._ensure_retrieval_components()
        if self._evaluation_runner is None:
            self._evaluation_runner = EvaluationRunner()
        if request.test_set_name:
            test_set = self.test_set_manager.load(
                request.test_set_name,
                expected_version=request.test_set_version,
            )
        else:
            inline_queries = [
                {
                    "query_id": query.query_id,
                    "query_text": query.query_text,
                    "query_type": query.query_type,
                    "relevant_docs": [
                        {"doc_id": doc.doc_id, "grade": doc.grade} for doc in query.relevant_docs
                    ],
                    "metadata": dict(query.metadata),
                }
                for query in request.queries or []
            ]
            version = request.test_set_version or "inline"
            serialised = json.dumps(inline_queries, sort_keys=True).encode("utf-8")
            inline_id = hashlib.sha256(serialised).hexdigest()[:8]
            name = request.test_set_name or f"inline-{inline_id}"
            test_set = build_test_set(name=name, queries=inline_queries, version=version)
        config = EvaluationConfig(
            top_k=request.top_k,
            components=tuple(request.components) if request.components else None,
            rerank=request.rerank,
        )

        def _run(record) -> Sequence[str]:
            metadata = dict(request.metadata)
            evaluation_meta = dict(metadata.get("evaluation", {}))
            evaluation_meta.update(
                {
                    "query_id": record.query_id,
                    "query_type": record.query_type.value,
                    "test_set_version": test_set.version,
                }
            )
            if record.metadata:
                evaluation_meta.setdefault("query_metadata", dict(record.metadata))
            if config.components:
                evaluation_meta["components"] = list(config.components)
            metadata["evaluation"] = evaluation_meta
            retrieval_request = RetrieveRequest(
                tenant_id=request.tenant_id,
                query=record.query_text,
                top_k=request.top_k,
                filters=dict(request.filters),
                rerank=request.rerank if request.rerank is not None else True,
                rerank_top_k=request.rerank_top_k,
                rerank_overflow=request.rerank_overflow,
                profile=request.profile,
                metadata=metadata,
                explain=False,
            )
            response = self.retrieve(retrieval_request)
            return [document.id for document in response.documents]

        return self._evaluation_runner.evaluate(
            test_set,
            _run,
            config=config,
            use_cache=request.use_cache,
        )

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
        intent_payload = {
            "detected": args.query_intent.value if args.query_intent else None,
            "table_only": args.table_only,
        }
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
            intent=intent_payload,
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
        updated = self.orchestrator.cancel_job(job_id, reason=reason)
        if not updated:
            return None
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
_kafka: KafkaClient | None = None
_ledger: JobLedger | None = None
_orchestrator: Orchestrator | None = None
_workers: list[WorkerBase] = []


def get_gateway_service() -> GatewayService:
    global _service, _kafka, _ledger, _orchestrator, _workers
    if _service is None:
        events = EventStreamManager()
        _kafka = KafkaClient()
        _kafka.create_topics(
            ["ingest.requests.v1", "ingest.results.v1", "mapping.events.v1", "ingest.deadletter.v1"]
        )
        _ledger = JobLedger()
        _orchestrator = Orchestrator(_kafka, _ledger, events)
        _service = GatewayService(events=events, orchestrator=_orchestrator, ledger=_ledger)
        _workers = [
            IngestWorker(_orchestrator, _kafka, _ledger, events),
            MappingWorker(_kafka, _ledger, events),
        ]
        _service.workers.extend(_workers)
    return _service
