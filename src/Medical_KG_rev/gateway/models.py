"""Shared models for the multi-protocol gateway."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel, Field


class ProblemDetail(BaseModel):
    """RFC 7807 compliant problem details payload."""

    type: str = Field(default="about:blank")
    title: str
    status: int
    detail: Optional[str] = None
    instance: Optional[str] = None
    extensions: Dict[str, Any] = Field(default_factory=dict)


class BatchError(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class OperationStatus(BaseModel):
    """Represents the state of a submitted operation across protocols."""

    job_id: str = Field(default_factory=lambda: "job-unknown")
    status: Literal["queued", "processing", "completed", "failed", "cancelled"] = "queued"
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    http_status: int = Field(default=202, ge=100, le=599)
    error: Optional[BatchError] = None


class BatchOperationResult(BaseModel):
    """Collection of operation statuses returned for batch requests."""

    operations: Sequence[OperationStatus]
    total: int
    succeeded: int
    failed: int


class DocumentChunk(BaseModel):
    document_id: str
    chunk_index: int
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int = Field(default=0, ge=0)


class EmbeddingVector(BaseModel):
    id: str
    vector: Sequence[float]
    model: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    id: str
    title: str
    score: float
    summary: Optional[str] = None
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    documents: Sequence[DocumentSummary]
    total: int
    rerank_metrics: Dict[str, Any] = Field(default_factory=dict)


class EntityLinkResult(BaseModel):
    mention: str
    entity_id: str
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    kind: str
    document_id: str
    results: Sequence[Dict[str, Any]]


class KnowledgeGraphWriteResult(BaseModel):
    nodes_written: int
    edges_written: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    type: str
    start: str
    end: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class IngestionRequest(BaseModel):
    tenant_id: str
    items: Sequence[Dict[str, Any]]
    priority: Literal["low", "normal", "high"] = "normal"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkRequest(BaseModel):
    tenant_id: str
    document_id: str
    strategy: Literal["semantic", "section", "paragraph", "table", "sliding-window"] = "section"
    chunk_size: int = Field(ge=64, le=4096, default=512)
    overlap: float = Field(default=0.1, ge=0.0, lt=1.0)
    options: Dict[str, Any] = Field(default_factory=dict)


class EmbedRequest(BaseModel):
    tenant_id: str
    inputs: Sequence[str]
    model: str
    normalize: bool = True


class RetrieveRequest(BaseModel):
    tenant_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Dict[str, Any] = Field(default_factory=dict)
    rerank: bool = True
    rerank_top_k: int = Field(default=10, ge=1, le=200)
    rerank_overflow: bool = False


class EntityLinkRequest(BaseModel):
    tenant_id: str
    mentions: Sequence[str]
    context: Optional[str] = None


class ExtractionRequest(BaseModel):
    tenant_id: str
    document_id: str
    options: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphWriteRequest(BaseModel):
    tenant_id: str
    nodes: Sequence[GraphNode] = Field(default_factory=list)
    edges: Sequence[GraphEdge] = Field(default_factory=list)
    transactional: bool = True


class JobEvent(BaseModel):
    job_id: str
    type: Literal[
        "jobs.started",
        "jobs.progress",
        "jobs.completed",
        "jobs.failed",
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)
    emitted_at: datetime = Field(default_factory=datetime.utcnow)


class JobHistoryEntry(BaseModel):
    from_status: str
    to_status: str
    stage: str
    reason: Optional[str] = None
    timestamp: datetime


class JobStatus(BaseModel):
    job_id: str
    doc_key: str
    tenant_id: str
    status: str
    stage: str
    pipeline: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    attempts: int = 0
    created_at: datetime
    updated_at: datetime
    history: Sequence[JobHistoryEntry] = Field(default_factory=list)


class Pagination(BaseModel):
    """GraphQL/REST shared pagination arguments."""

    after: Optional[str] = None
    first: int = Field(default=10, ge=1, le=100)


class SearchArguments(BaseModel):
    query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    pagination: Pagination = Field(default_factory=Pagination)


def build_batch_result(statuses: Iterable[OperationStatus]) -> BatchOperationResult:
    items = list(statuses)
    succeeded = sum(1 for status in items if status.error is None and 200 <= status.http_status < 300)
    failed = len(items) - succeeded
    return BatchOperationResult(operations=items, total=len(items), succeeded=succeeded, failed=failed)
