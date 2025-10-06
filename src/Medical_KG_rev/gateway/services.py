"""Protocol-agnostic gateway service layer."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

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

    def _new_job(self, tenant_id: str, operation: str) -> str:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        logger.info("gateway.job.created", tenant_id=tenant_id, job_id=job_id, operation=operation)
        self.events.publish(
            JobEvent(job_id=job_id, type="jobs.started", payload={"operation": operation})
        )
        return job_id

    def _complete_job(self, job_id: str, payload: Optional[dict] = None) -> None:
        logger.info("gateway.job.completed", job_id=job_id)
        self.events.publish(JobEvent(job_id=job_id, type="jobs.completed", payload=payload or {}))

    def _fail_job(self, job_id: str, reason: str) -> None:
        logger.warning("gateway.job.failed", job_id=job_id, reason=reason)
        self.events.publish(JobEvent(job_id=job_id, type="jobs.failed", payload={"reason": reason}))

    def ingest(self, dataset: str, request: IngestionRequest) -> BatchOperationResult:
        statuses: List[OperationStatus] = []
        for item in request.items:
            job_id = self._new_job(request.tenant_id, f"ingest:{dataset}")
            status = OperationStatus(
                job_id=job_id,
                status="completed",
                message=f"Ingested item for {dataset}",
                metadata={"dataset": dataset, "item": item.get("id", uuid.uuid4().hex)},
            )
            statuses.append(status)
            self._complete_job(job_id, payload={"dataset": dataset})
        return build_batch_result(statuses)

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
        self._complete_job(job_id, payload={"embeddings": len(embeddings)})
        return embeddings

    def retrieve(self, request: RetrieveRequest) -> RetrievalResult:
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
        self._complete_job(job_id, payload={"documents": result.total})
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
        self._complete_job(job_id, payload={"links": len(results)})
        return results

    def extract(self, kind: str, request: ExtractionRequest) -> ExtractionResult:
        job_id = self._new_job(request.tenant_id, f"extract:{kind}")
        results = [
            {"kind": kind, "document_id": request.document_id, "value": "synthetic"},
        ]
        self._complete_job(job_id, payload={"kind": kind})
        return ExtractionResult(kind=kind, document_id=request.document_id, results=results)

    def write_kg(self, request: KnowledgeGraphWriteRequest) -> KnowledgeGraphWriteResult:
        job_id = self._new_job(request.tenant_id, "kg-write")
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
        return request


_service: Optional[GatewayService] = None


def get_gateway_service() -> GatewayService:
    global _service
    if _service is None:
        _service = GatewayService(events=EventStreamManager())
    return _service
