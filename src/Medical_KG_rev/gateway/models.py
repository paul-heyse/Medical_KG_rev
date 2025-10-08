"""Shared models for the multi-protocol gateway."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from Medical_KG_rev.adapters import AdapterDomain


class ProblemDetail(BaseModel):
    """RFC 7807 compliant problem details payload."""

    type: str = Field(default="about:blank")
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)


class BatchError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class OperationStatus(BaseModel):
    """Represents the state of a submitted operation across protocols."""

    job_id: str = Field(default_factory=lambda: "job-unknown")
    status: Literal["queued", "processing", "completed", "failed", "cancelled"] = "queued"
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    http_status: int = Field(default=202, ge=100, le=599)
    error: BatchError | None = None


class BatchOperationResult(BaseModel):
    """Collection of operation statuses returned for batch requests."""

    operations: Sequence[OperationStatus]
    total: int
    succeeded: int
    failed: int


class AdapterMetadataView(BaseModel):
    """REST response model representing adapter metadata."""

    name: str
    version: str
    domain: AdapterDomain
    summary: str
    capabilities: list[str] = Field(default_factory=list)
    maintainer: str | None = None
    dataset: str | None = None
    config_schema: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class AdapterHealthView(BaseModel):
    name: str
    healthy: bool


class AdapterConfigSchemaView(BaseModel):
    name: str
    schema: dict[str, Any]


class DocumentChunk(BaseModel):
    document_id: str
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int = Field(default=0, ge=0)


class EmbeddingVector(BaseModel):
    id: str
    model: str
    namespace: str
    kind: str = "single_vector"
    dimension: int = Field(ge=0, default=0)
    vector: Sequence[float] | None = None
    terms: dict[str, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    id: str
    title: str
    score: float
    summary: str | None = None
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    explain: dict[str, Any] | None = None


class RetrievalResult(BaseModel):
    query: str
    documents: Sequence[DocumentSummary]
    total: int
    rerank_metrics: dict[str, Any] = Field(default_factory=dict)
    pipeline_version: str | None = None
    partial: bool = False
    degraded: bool = False
    errors: Sequence[ProblemDetail] = Field(default_factory=list)
    stage_timings: dict[str, float] = Field(default_factory=dict)


class EntityLinkResult(BaseModel):
    mention: str
    entity_id: str
    confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    kind: str
    document_id: str
    results: Sequence[dict[str, Any]]


class KnowledgeGraphWriteResult(BaseModel):
    nodes_written: int
    edges_written: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    id: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    type: str
    start: str
    end: str
    properties: dict[str, Any] = Field(default_factory=dict)


class IngestionRequest(BaseModel):
    tenant_id: str
    items: Sequence[dict[str, Any]]
    priority: Literal["low", "normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)
    profile: str | None = None


class PipelineIngestionRequest(IngestionRequest):
    dataset: str


class ChunkRequest(BaseModel):
    tenant_id: str
    document_id: str
    strategy: Literal["semantic", "section", "paragraph", "table", "sliding-window"] = "section"
    chunk_size: int = Field(ge=64, le=4096, default=512)
    overlap: float = Field(default=0.1, ge=0.0, lt=1.0)
    options: dict[str, Any] = Field(default_factory=dict)


class EmbedRequest(BaseModel):
    tenant_id: str
    inputs: Sequence[str]
    model: str
    namespace: str
    normalize: bool = True


class RetrieveRequest(BaseModel):
    tenant_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, Any] = Field(default_factory=dict)
    rerank: bool = True
    rerank_top_k: int = Field(default=10, ge=1, le=200)
    rerank_overflow: bool = False
    profile: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    explain: bool = False


class PipelineQueryRequest(RetrieveRequest):
    profile: str | None = None


class EntityLinkRequest(BaseModel):
    tenant_id: str
    mentions: Sequence[str]
    context: str | None = None


class ExtractionRequest(BaseModel):
    tenant_id: str
    document_id: str
    options: dict[str, Any] = Field(default_factory=dict)


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
    payload: dict[str, Any] = Field(default_factory=dict)
    emitted_at: datetime = Field(default_factory=datetime.utcnow)


class JobHistoryEntry(BaseModel):
    from_status: str
    to_status: str
    stage: str
    reason: str | None = None
    timestamp: datetime


class JobStatus(BaseModel):
    job_id: str
    doc_key: str
    tenant_id: str
    status: str
    stage: str
    pipeline: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    attempts: int = 0
    created_at: datetime
    updated_at: datetime
    history: Sequence[JobHistoryEntry] = Field(default_factory=list)


class Pagination(BaseModel):
    """GraphQL/REST shared pagination arguments."""

    after: str | None = None
    first: int = Field(default=10, ge=1, le=100)


class SearchArguments(BaseModel):
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    pagination: Pagination = Field(default_factory=Pagination)


def build_batch_result(statuses: Iterable[OperationStatus]) -> BatchOperationResult:
    items = list(statuses)
    succeeded = sum(
        1 for status in items if status.error is None and 200 <= status.http_status < 300
    )
    failed = len(items) - succeeded
    return BatchOperationResult(
        operations=items, total=len(items), succeeded=succeeded, failed=failed
    )
