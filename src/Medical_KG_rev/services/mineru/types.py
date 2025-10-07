from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from Medical_KG_rev.models.equation import Equation
from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.models.ir import Block as IrBlock
from Medical_KG_rev.models.ir import Document as IrDocument
from Medical_KG_rev.models.table import Table


@dataclass(slots=True)
class Block:
    """Representation of a document block produced by MinerU."""

    id: str
    page: int
    kind: str
    text: str | None
    bbox: tuple[float, float, float, float] | None
    confidence: float | None
    reading_order: int | None
    metadata: dict[str, Any]
    table: Table | None = None
    figure: Figure | None = None
    equation: Equation | None = None
    ir_block: IrBlock | None = None


@dataclass(slots=True)
class Document:
    """Structured intermediate representation for a PDF document."""

    document_id: str
    tenant_id: str
    blocks: list[Block] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    ir_document: IrDocument | None = None


@dataclass(slots=True)
class MineruRequest:
    tenant_id: str
    document_id: str
    content: bytes


@dataclass(slots=True)
class MineruResponse:
    document: Document
    processed_at: datetime
    duration_seconds: float
    metadata: "ProcessingMetadata"


@dataclass(slots=True)
class MineruBatchResponse:
    documents: list[Document]
    processed_at: datetime
    duration_seconds: float
    metadata: list["ProcessingMetadata"]


@dataclass(slots=True)
class MineruBatchRequest:
    tenant_id: str
    requests: Sequence[MineruRequest]


@dataclass(slots=True)
class ProcessingMetadata:
    document_id: str
    mineru_version: str | None
    model_names: dict[str, str]
    gpu_id: str | None
    worker_id: str | None
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    cli_stdout: str
    cli_stderr: str
    cli_descriptor: str
    planned_memory_mb: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "mineru_version": self.mineru_version,
            "model_names": dict(self.model_names),
            "gpu_id": self.gpu_id,
            "worker_id": self.worker_id,
            "started_at": self.started_at.astimezone(timezone.utc).isoformat(),
            "completed_at": self.completed_at.astimezone(timezone.utc).isoformat(),
            "duration_seconds": self.duration_seconds,
            "cli_stdout": self.cli_stdout,
            "cli_stderr": self.cli_stderr,
            "cli": self.cli_descriptor,
            "planned_memory_mb": self.planned_memory_mb,
        }


__all__ = [
    "Block",
    "Document",
    "MineruRequest",
    "MineruResponse",
    "MineruBatchRequest",
    "MineruBatchResponse",
    "ProcessingMetadata",
]
