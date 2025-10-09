"""Shared models for the multi-protocol gateway.

This module provides Pydantic models used across all protocol handlers in the
gateway. These models define the structure of requests, responses, and data
transfer objects used by REST, GraphQL, gRPC, SOAP, and SSE endpoints.

Key Responsibilities:
    - Define request/response models for all protocol handlers
    - Provide validation and serialization for API data
    - Standardize error response formats across protocols
    - Define batch operation models for bulk requests
    - Provide models for evaluation and retrieval operations

Collaborators:
    - Upstream: All protocol handlers (REST, GraphQL, gRPC, SOAP, SSE)
    - Downstream: Gateway services, coordinators, domain services

Side Effects:
    - None: Pure data models with no side effects

Thread Safety:
    - Thread-safe: Pydantic models are immutable and thread-safe

Performance Characteristics:
    - O(1) model instantiation and validation
    - O(n) serialization where n is model size
    - Efficient JSON serialization with Pydantic

Example:
    >>> from Medical_KG_rev.gateway.models import ProblemDetail
    >>> error = ProblemDetail(
    ...     title="Validation Error",
    ...     status=400,
    ...     detail="Invalid input provided"
    ... )
    >>> print(error.model_dump_json())
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from Medical_KG_rev.adapters import AdapterDomain
from Medical_KG_rev.services.evaluation import EvaluationResult, MetricSummary
from Medical_KG_rev.services.retrieval.routing import QueryIntent

# ==============================================================================
# ERROR MODELS
# ==============================================================================


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


# ==============================================================================
# OPERATION MODELS
# ==============================================================================

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


# ==============================================================================
# ADAPTER MODELS
# ==============================================================================


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


# ==============================================================================
# RETRIEVAL MODELS
# ==============================================================================

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
    intent: dict[str, Any] = Field(default_factory=dict)


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


# ==============================================================================
# REQUEST MODELS
# ==============================================================================

class IngestionRequest(BaseModel):
    tenant_id: str
    items: Sequence[dict[str, Any]]
    priority: Literal["low", "normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)
    profile: str | None = None
    chunking_options: dict[str, Any] | None = None


class PipelineIngestionRequest(IngestionRequest):
    dataset: str


class ChunkRequest(BaseModel):
    tenant_id: str
    document_id: str
    text: str | None = None
    strategy: Literal["semantic", "section", "paragraph", "table", "sliding-window"] = "section"
    chunk_size: int = Field(ge=64, le=4096, default=512)
    overlap: float = Field(default=0.1, ge=0.0, lt=1.0)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_text(cls, values: "ChunkRequest") -> "ChunkRequest":
        text = values.text
        if text is None:
            option_text = values.options.get("text") if isinstance(values.options, dict) else None
            if isinstance(option_text, str):
                text = option_text
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Chunk requests must include a non-empty 'text' field")
        values.text = text
        if "text" in values.options:
            values.options = {k: v for k, v in values.options.items() if k != "text"}
        return values


class EmbeddingOptions(BaseModel):
    normalize: bool = True
    model: str | None = None


class EmbedRequest(BaseModel):
    tenant_id: str
    texts: Sequence[str]
    namespace: str
    options: EmbeddingOptions | None = None


class EmbeddingMetadata(BaseModel):
    provider: str
    dimension: int | None = None
    duration_ms: float | None = None
    model: str | None = None


class EmbeddingResponse(BaseModel):
    namespace: str
    embeddings: Sequence[EmbeddingVector]
    metadata: EmbeddingMetadata


class NamespaceInfo(BaseModel):
    id: str
    provider: str
    kind: str
    dimension: int | None = None
    max_tokens: int | None = None
    enabled: bool = True
    allowed_tenants: list[str] = Field(default_factory=list)
    allowed_scopes: list[str] = Field(default_factory=list)


class NamespaceValidationResult(BaseModel):
    text_index: int
    token_count: int
    exceeds_budget: bool
    warning: str | None = None


class NamespaceValidationResponse(BaseModel):
    namespace: str
    valid: bool
    results: Sequence[NamespaceValidationResult]


# ==============================================================================
# NAMESPACE MODELS
# ==============================================================================

class NamespaceValidationRequest(BaseModel):
    tenant_id: str
    texts: Sequence[str] = Field(default_factory=list)


class NamespacePolicySettingsView(BaseModel):
    cache_ttl_seconds: float = Field(default=60.0, ge=0.0)
    max_cache_entries: int = Field(default=512, ge=1)
    dry_run: bool = False


class NamespacePolicyStatus(BaseModel):
    settings: NamespacePolicySettingsView
    stats: dict[str, Any] = Field(default_factory=dict)
    operational: dict[str, Any] = Field(default_factory=dict)


class NamespacePolicyUpdateRequest(BaseModel):
    cache_ttl_seconds: float | None = Field(default=None, ge=0.0)
    max_cache_entries: int | None = Field(default=None, ge=1)
    dry_run: bool | None = None


class NamespacePolicyDiagnosticsView(BaseModel):
    settings: NamespacePolicySettingsView
    cache_keys: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class NamespacePolicyHealthView(BaseModel):
    policy: str
    evaluations: int
    denials: int
    cache_entries: int


class NamespacePolicyMetricsView(BaseModel):
    metrics: dict[str, Any] = Field(default_factory=dict)


class NamespacePolicyInvalidateRequest(BaseModel):
    namespace: str | None = None


class RetrieveRequest(BaseModel):
    tenant_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, Any] = Field(default_factory=dict)
    rerank: bool | None = None
    rerank_model: str | None = Field(default=None, min_length=1, max_length=128)
    rerank_top_k: int = Field(default=10, ge=1, le=200)
    rerank_overflow: bool = False
    profile: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    explain: bool = False
    query_intent: QueryIntent | None = Field(default=None)
    table_only: bool = False


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


class EvaluationRelevantDoc(BaseModel):
    doc_id: str
    grade: float = Field(ge=0.0, le=3.0)


class EvaluationQuery(BaseModel):
    query_id: str
    query_text: str
    query_type: Literal["exact_term", "paraphrase", "complex_clinical"]
    relevant_docs: Sequence[EvaluationRelevantDoc]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# EVALUATION MODELS
# ==============================================================================

class EvaluationRequest(BaseModel):
    tenant_id: str
    test_set_name: str | None = None
    test_set_version: str | None = None
    queries: Sequence[EvaluationQuery] | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    components: Sequence[str] | None = None
    rerank: bool | None = None
    rerank_top_k: int = Field(default=50, ge=1, le=500)
    rerank_overflow: bool = False
    profile: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    use_cache: bool = True

    @model_validator(mode="after")
    def _validate_source(self) -> "EvaluationRequest":
        if not self.test_set_name and not self.queries:
            raise ValueError("Either 'test_set_name' or 'queries' must be provided")
        return self


class MetricSummaryView(BaseModel):
    mean: float
    median: float
    std: float
    ci_low: float | None = None
    ci_high: float | None = None

    @classmethod
    def from_metric(cls, summary: MetricSummary) -> "MetricSummaryView":
        return cls(
            mean=summary.mean,
            median=summary.median,
            std=summary.std,
            ci_low=summary.ci_low,
            ci_high=summary.ci_high,
        )


class EvaluationResponse(BaseModel):
    dataset: str
    test_set_version: str
    metrics: dict[str, MetricSummaryView]
    latency_ms: MetricSummaryView
    per_query_type: dict[str, dict[str, float]]
    per_query: dict[str, dict[str, float]]
    cache: dict[str, Any]
    config: dict[str, Any]

    @classmethod
    def from_result(cls, result: EvaluationResult) -> "EvaluationResponse":
        metrics = {name: MetricSummaryView.from_metric(summary) for name, summary in result.metrics.items()}
        latency = MetricSummaryView.from_metric(result.latency)
        config = json.loads(result.config.to_json())
        return cls(
            dataset=result.dataset,
            test_set_version=result.test_set_version,
            metrics=metrics,
            latency_ms=latency,
            per_query_type={key: dict(values) for key, values in result.per_query_type.items()},
            per_query={key: dict(values) for key, values in result.per_query.items()},
            cache={"key": result.cache_key, "hit": result.cache_hit},
            config=config,
        )


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
    query_intent: QueryIntent | None = Field(default=None)
    table_only: bool = False


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def build_batch_result(statuses: Iterable[OperationStatus]) -> BatchOperationResult:
    """Build a batch operation result from a collection of operation statuses.

    Args:
        statuses: Collection of operation status objects.

    Returns:
        Batch operation result with summary statistics.
    """
    items = list(statuses)
    succeeded = sum(
        1 for status in items if status.error is None and 200 <= status.http_status < 300
    )
    failed = len(items) - succeeded
    return BatchOperationResult(
        operations=items, total=len(items), succeeded=succeeded, failed=failed
    )


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Error models
    "ProblemDetail",
    "BatchError",

    # Operation models
    "OperationStatus",
    "BatchOperationResult",

    # Adapter models
    "AdapterMetadataView",

    # Retrieval models
    "DocumentSummary",
    "RetrievalResult",
    "EntityLinkResult",
    "GraphNode",
    "GraphEdge",

    # Request models
    "IngestionRequest",
    "PipelineIngestionRequest",
    "ChunkRequest",
    "EmbedRequest",
    "RetrieveRequest",

    # Namespace models
    "NamespaceValidationRequest",
    "NamespaceValidationResult",
    "NamespaceValidationResponse",
    "NamespacePolicySettingsView",
    "NamespacePolicyStatus",
    "NamespacePolicyUpdateRequest",
    "NamespacePolicyDiagnosticsView",
    "NamespacePolicyHealthView",

    # Evaluation models
    "EvaluationRequest",
    "EvaluationQuery",
    "EvaluationRelevantDoc",
    "EvaluationResult",
    "MetricSummaryView",
    "EvaluationResponse",

    # Job models
    "JobHistoryEntry",
    "JobStatus",

    # Utility models
    "Pagination",
    "SearchArguments",

    # Helper functions
    "build_batch_result",
]
