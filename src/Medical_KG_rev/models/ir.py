"""Core intermediate representation (IR) models.

These models provide the shared vocabulary for documents that flow through the
system. The structure intentionally mirrors the design decisions captured in
``openspec/changes/add-foundation-infrastructure/design.md`` by providing a
hierarchy of documents, sections, blocks and spans that other modules can
consume without having to know the underlying source specific formats.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from .equation import Equation
from .figure import Figure
from .table import Table, TableCell

class IRBaseModel(BaseModel):
    """Base model that enforces strict validation across the IR."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Span(IRBaseModel):
    """Represents a span of text in a document."""

    start: int = Field(ge=0, description="Inclusive character offset")
    end: int = Field(gt=0, description="Exclusive character offset")
    text: str | None = Field(default=None, description="Optional resolved text")

    @model_validator(mode="after")
    def _validate_range(self) -> Span:
        if self.end <= self.start:
            raise ValueError("Span end must be greater than start")
        if self.text is not None and len(self.text) != self.end - self.start:
            raise ValueError("Span text length must match span range")
        return self


class BlockType(str, Enum):
    """Block type enumeration to keep downstream logic consistent."""

    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    HEADER = "header"
    FOOTNOTE = "footnote"
    EQUATION = "equation"


class Block(IRBaseModel):
    """A block is the smallest logical unit we track in the IR."""

    id: str
    type: BlockType = BlockType.PARAGRAPH
    text: str | None = Field(default=None, description="Plain-text content")
    spans: Sequence[Span] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    layout_bbox: tuple[float, float, float, float] | None = Field(default=None)
    reading_order: int | None = Field(default=None, ge=0)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    table: Table | None = Field(default=None)
    figure: Figure | None = Field(default=None)
    equation: Equation | None = Field(default=None)

    @field_validator("spans")
    @classmethod
    def _ensure_spans_sorted(cls, value: Sequence[Span]) -> Sequence[Span]:
        if any(value[i].start > value[i + 1].start for i in range(len(value) - 1)):
            raise ValueError("Block spans must be ordered by start offset")
        return value


class Section(IRBaseModel):
    """High-level grouping of blocks."""

    id: str
    title: str | None = None
    blocks: Sequence[Block] = Field(default_factory=list)

    @field_validator("blocks")
    @classmethod
    def _validate_blocks(cls, value: Sequence[Block]) -> Sequence[Block]:
        ids = [block.id for block in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Section blocks must have unique identifiers")
        return value


class Document(IRBaseModel):
    """Top level document representation."""

    id: str
    source: str = Field(description="Logical source identifier (e.g. clinicaltrials)")
    title: str | None = None
    sections: Sequence[Section] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = Field(default="v1")
    metadata: dict[str, Any] = Field(default_factory=dict)
    pdf_url: HttpUrl | None = Field(
        default=None,
        description="Canonical URL for downloading the document PDF",
    )
    pdf_size: int | None = Field(
        default=None,
        ge=0,
        description="Size of the referenced PDF in bytes when known",
    )
    pdf_content_type: str | None = Field(
        default=None,
        description="MIME content type reported for the PDF resource",
    )
    pdf_checksum: str | None = Field(
        default=None,
        description="Checksum of the PDF payload when validated downstream",
    )
    pdf_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional PDF specific metadata captured during ingestion",
    )

    @field_validator("sections")
    @classmethod
    def _validate_sections(cls, value: Sequence[Section]) -> Sequence[Section]:
        ids = [section.id for section in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Document sections must have unique identifiers")
        return value

    @model_validator(mode="after")
    def _validate_spans(self) -> Document:
        text_lengths: dict[str, int] = {}
        for section in self.sections:
            for block in section.blocks:
                if block.text is not None:
                    text_lengths[block.id] = len(block.text)
                for span in block.spans:
                    if block.text is None:
                        continue
                    if span.end > text_lengths.get(block.id, 0):
                        raise ValueError(
                            f"Span {span.start}-{span.end} exceeds bounds for block '{block.id}'"
                        )
        return self

    @model_validator(mode="after")
    def _validate_pdf_fields(self) -> Document:
        if self.pdf_content_type and "pdf" not in self.pdf_content_type.lower():
            raise ValueError(
                "pdf_content_type must describe a PDF media type when provided"
            )
        if self.pdf_checksum and not self.pdf_checksum.strip():
            raise ValueError("pdf_checksum must be a non-empty string when provided")
        if self.pdf_metadata and not isinstance(self.pdf_metadata, dict):
            raise ValueError("pdf_metadata must be a mapping of metadata values")
        if self.pdf_url is None:
            if any(value is not None for value in (self.pdf_size, self.pdf_content_type, self.pdf_checksum)):
                raise ValueError(
                    "pdf_size, pdf_content_type, and pdf_checksum require a pdf_url to be set"
                )
        return self

    def iter_blocks(self) -> Iterable[Block]:
        """Iterate over all blocks contained in the document."""

        for section in self.sections:
            for block in section.blocks:
                yield block

    def find_spans(self, predicate: Any | None = None) -> list[Span]:
        """Return spans that match the predicate."""

        spans: list[Span] = []
        for block in self.iter_blocks():
            for span in block.spans:
                if predicate is None or predicate(span):
                    spans.append(span)
        return spans
