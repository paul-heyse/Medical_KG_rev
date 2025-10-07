"""Protocol-agnostic gateway service layer."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter

import structlog

from ..kg import ShaclValidator, ValidationError
from ..observability.metrics import observe_job_duration, record_business_event
from ..orchestration import JobLedger, JobLedgerEntry, Orchestrator
from ..orchestration.kafka import KafkaClient
from ..orchestration.worker import IngestWorker, MappingWorker, WorkerBase
from ..services.extraction.templates import TemplateValidationError, validate_template
from ..services.retrieval.chunking import ChunkingOptions, ChunkingService
from ..services.retrieval.reranker import CrossEncoderReranker
from ..validation import UCUMValidator
from ..validation.fhir import FHIRValidationError, FHIRValidator
from .models import (
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
    orchestrator: Orchestrator
    ledger: JobLedger
    workers: list[WorkerBase] = field(default_factory=list)
    chunker: ChunkingService = field(default_factory=ChunkingService)
    reranker: CrossEncoderReranker = field(default_factory=CrossEncoderReranker)
    shacl: ShaclValidator = field(default_factory=ShaclValidator.default)
    ucum: UCUMValidator = field(default_factory=UCUMValidator)
    fhir: FHIRValidator = field(default_factory=FHIRValidator)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
            entry = self.orchestrator.submit_job(
                tenant_id=request.tenant_id,
                dataset=dataset,
                item=item,
                priority=request.priority,
                metadata=request.metadata,
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
            record_business_event("documents_ingested", len(request.items))
        return result

    def chunk_document(self, request: ChunkRequest) -> Sequence[DocumentChunk]:
        job_id = self._new_job(request.tenant_id, "chunk")
        sample_text = request.options.get("text") if isinstance(request.options, dict) else None  # type: ignore[attr-defined]
        if not sample_text:
            sample_text = (
                "Population: Adults with hypertension. Intervention: ACE inhibitor administered daily. "
                "Comparison: Placebo. Outcome: Reduced systolic blood pressure at 12 weeks."
            )
        options = ChunkingOptions(
            strategy=request.strategy,
            max_tokens=request.chunk_size,
            overlap=request.overlap,
        )
        raw_chunks = self.chunker.chunk(
            request.tenant_id, request.document_id, sample_text, options
        )
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
        started = perf_counter()
        job_id = self._new_job(request.tenant_id, "retrieve")
        documents = [
            DocumentSummary(
                id=f"doc-{i}",
                title=f"Synthetic Document {i}",
                score=1.0 - (i * 0.1),
                source="synthetic",
                summary=f"Summary for {request.query} #{i}",
                metadata=request.filters,
            )
            for i in range(min(request.top_k, 3))
        ]
        rerank_metrics = {}
        if request.rerank and documents:
            candidates = []
            for doc in documents:
                payload = doc.model_dump(mode="python")
                payload["text"] = doc.summary or doc.title
                candidates.append(payload)
            ranked, metrics = self.reranker.rerank(
                request.query,
                candidates,
                text_field="text",
                top_k=request.rerank_top_k,
            )
            rerank_metrics = dict(metrics)
            documents = [
                DocumentSummary(
                    id=item["id"],
                    title=item["title"],
                    score=float(item.get("score", 0.0)),
                    summary=item.get("summary"),
                    source=item.get("source", "synthetic"),
                    metadata=item.get("metadata", {}),
                )
                for item in ranked
            ]
        result = RetrievalResult(
            query=request.query,
            documents=documents,
            total=len(documents),
            rerank_metrics=rerank_metrics,
        )
        self.ledger.update_metadata(job_id, {"documents": result.total})
        self._complete_job(job_id, payload={"documents": result.total})
        observe_job_duration("retrieve", perf_counter() - started)
        record_business_event("retrieval_requests")
        if result.total:
            record_business_event("documents_retrieved", result.total)
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
        updated = self.orchestrator.cancel_job(job_id, reason=reason)
        if not updated:
            return None
        return self._to_job_status(updated)


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
