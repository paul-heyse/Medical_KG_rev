"""Protocol-agnostic gateway service layer."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Optional, Sequence

import structlog

from .models import (
    BatchOperationResult,
    ChunkRequest,
    DocumentChunk,
    DocumentSummary,
    EmbedRequest,
    EmbeddingVector,
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
from ..orchestration import JobLedger, JobLedgerEntry, Orchestrator
from ..orchestration.kafka import KafkaClient
from ..orchestration.worker import IngestWorker, MappingWorker, WorkerBase
from ..observability.metrics import observe_job_duration, record_business_event

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
    workers: List[WorkerBase] = field(default_factory=list)

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

    def _new_job(
        self, tenant_id: str, operation: str, *, metadata: Optional[dict] = None
    ) -> str:
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

    def _complete_job(self, job_id: str, payload: Optional[dict] = None) -> None:
        logger.info("gateway.job.completed", job_id=job_id)
        self.ledger.mark_completed(job_id, metadata=payload or {})
        self.events.publish(JobEvent(job_id=job_id, type="jobs.completed", payload=payload or {}))

    def _fail_job(self, job_id: str, reason: str) -> None:
        logger.warning("gateway.job.failed", job_id=job_id, reason=reason)
        self.ledger.mark_failed(job_id, stage="error", reason=reason)
        self.events.publish(JobEvent(job_id=job_id, type="jobs.failed", payload={"reason": reason}))

    def ingest(self, dataset: str, request: IngestionRequest) -> BatchOperationResult:
        started = perf_counter()
        statuses: List[OperationStatus] = []
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
            statuses.append(
                OperationStatus(
                    job_id=entry.job_id,
                    status=entry.status,
                    message=message,
                    metadata={
                        "dataset": dataset,
                        "pipeline": entry.pipeline,
                        "doc_key": entry.doc_key,
                        "duplicate": duplicate,
                    },
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
        chunks: List[DocumentChunk] = []
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        step = max(1, len(text) // request.chunk_size)
        for index, start in enumerate(range(0, len(text), step)):
            chunks.append(
                DocumentChunk(
                    document_id=request.document_id,
                    chunk_index=index,
                    content=text[start : start + step],
                    metadata={"strategy": request.strategy},
                )
            )
        self.ledger.update_metadata(job_id, {"chunks": len(chunks)})
        self._complete_job(job_id, payload={"chunks": len(chunks)})
        return chunks

    def embed(self, request: EmbedRequest) -> Sequence[EmbeddingVector]:
        job_id = self._new_job(request.tenant_id, "embed")
        embeddings: List[EmbeddingVector] = []
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
        result = RetrievalResult(query=request.query, documents=documents, total=len(documents))
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
        results = [
            {"kind": kind, "document_id": request.document_id, "value": "synthetic"},
        ]
        self.ledger.update_metadata(job_id, {"kind": kind})
        self._complete_job(job_id, payload={"kind": kind})
        return ExtractionResult(kind=kind, document_id=request.document_id, results=results)

    def write_kg(self, request: KnowledgeGraphWriteRequest) -> KnowledgeGraphWriteResult:
        job_id = self._new_job(request.tenant_id, "kg-write")
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
    def get_job(self, job_id: str, *, tenant_id: str) -> Optional[JobStatus]:
        entry = self.ledger.get(job_id)
        if not entry or entry.tenant_id != tenant_id:
            return None
        return self._to_job_status(entry)

    def list_jobs(self, *, tenant_id: str, status: Optional[str] = None) -> List[JobStatus]:
        entries = self.ledger.list(status=status)
        filtered = [entry for entry in entries if entry.tenant_id == tenant_id]
        return [self._to_job_status(entry) for entry in filtered]

    def cancel_job(
        self, job_id: str, *, tenant_id: str, reason: Optional[str] = None
    ) -> Optional[JobStatus]:
        entry = self.ledger.get(job_id)
        if not entry or entry.tenant_id != tenant_id:
            return None
        updated = self.orchestrator.cancel_job(job_id, reason=reason)
        if not updated:
            return None
        return self._to_job_status(updated)


_service: Optional[GatewayService] = None
_kafka: Optional[KafkaClient] = None
_ledger: Optional[JobLedger] = None
_orchestrator: Optional[Orchestrator] = None
_workers: List[WorkerBase] = []


def get_gateway_service() -> GatewayService:
    global _service, _kafka, _ledger, _orchestrator, _workers
    if _service is None:
        events = EventStreamManager()
        _kafka = KafkaClient()
        _kafka.create_topics(["ingest.requests.v1", "ingest.results.v1", "mapping.events.v1", "ingest.deadletter.v1"])
        _ledger = JobLedger()
        _orchestrator = Orchestrator(_kafka, _ledger, events)
        _service = GatewayService(events=events, orchestrator=_orchestrator, ledger=_ledger)
        _workers = [
            IngestWorker(_orchestrator, _kafka, _ledger, events),
            MappingWorker(_kafka, _ledger, events),
        ]
        _service.workers.extend(_workers)
    return _service
