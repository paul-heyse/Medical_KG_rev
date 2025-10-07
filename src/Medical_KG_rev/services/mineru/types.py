from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from Medical_KG_rev.models.equation import Equation
from Medical_KG_rev.models.figure import Figure
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


__all__ = ["Block", "Document", "MineruRequest", "MineruResponse"]
