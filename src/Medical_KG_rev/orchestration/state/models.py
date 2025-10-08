"""Pydantic representations for pipeline state payloads."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field


class StageContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    job_id: str | None = None
    doc_id: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    pipeline_name: str | None = None
    pipeline_version: str | None = None


class StageResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    stage_type: str
    attempts: int | None = None
    duration_ms: int | None = None
    output_count: int | None = None
    error: str | None = None


class PdfGateModel(BaseModel):
    """Structured representation of PDF gate state."""

    model_config = ConfigDict(extra="forbid")

    downloaded: bool = False
    ir_ready: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineStateModel(BaseModel):
    """Validated representation of the PipelineState serialisation payload."""

    model_config = ConfigDict(extra="forbid")

    version: str
    job_id: str | None
    context: StageContextModel
    adapter_request: Mapping[str, Any]
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_count: int
    document_id: str | None = None
    chunk_count: int = 0
    embedding_count: int = 0
    entity_count: int = 0
    claim_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    stage_results: dict[str, StageResultModel] = Field(default_factory=dict)
    pdf_gate: PdfGateModel | None = None
    index_receipt: dict[str, Any] | None = None
    graph_receipt: dict[str, Any] | None = None


__all__ = [
    "PipelineStateModel",
    "PdfGateModel",
    "StageContextModel",
    "StageResultModel",
]
