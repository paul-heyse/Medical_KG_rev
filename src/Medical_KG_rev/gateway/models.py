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


class OperationStatus(BaseModel):
    """Represents the state of a submitted operation across protocols."""

    job_id: str = Field(default_factory=lambda: "job-unknown")
    status: Literal["queued", "processing", "completed", "failed", "cancelled"] = "queued"
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchOperationResult(BaseModel):
    """Collection of operation statuses returned for batch requests."""

    operations: Sequence[OperationStatus]
    total: int


class DocumentChunk(BaseModel):
    document_id: str
    chunk_index: int
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


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


class IngestionRequest(BaseModel):
    tenant_id: str
    items: Sequence[Dict[str, Any]]
    priority: Literal["low", "normal", "high"] = "normal"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkRequest(BaseModel):
    tenant_id: str
    document_id: str
    strategy: Literal["semantic", "fixed"] = "semantic"
    chunk_size: int = Field(ge=128, le=4096, default=1024)


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
    nodes: Sequence[Dict[str, Any]] = Field(default_factory=list)
    edges: Sequence[Dict[str, Any]] = Field(default_factory=list)
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
    return BatchOperationResult(operations=items, total=len(items))
