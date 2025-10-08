"""Validation helpers for chunk outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field, ValidationError, model_validator

from .port import Chunk


class ChunkValidationError(ValueError):
    """Raised when chunk validation fails."""


@dataclass(slots=True)
class ChunkValidationResult:
    chunk_id: str
    valid: bool
    reason: str | None = None


_REQUIRED_METADATA = {"chunking_profile", "source_system", "chunker_version", "created_at"}


class ChunkModel(BaseModel):
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    char_offsets: tuple[int, int]
    section_label: str
    intent_hint: str
    page_bbox: dict[str, Any] | None = None
    metadata: dict[str, Any]

    @model_validator(mode="after")
    def _validate_offsets(cls, values: "ChunkModel") -> "ChunkModel":
        start, end = values.char_offsets
        if start < 0 or end < start:
            raise ValueError("char_offsets must be ordered and non-negative")
        missing = _REQUIRED_METADATA.difference(values.metadata)
        if missing:
            raise ValueError(f"missing metadata: {', '.join(sorted(missing))}")
        return values


def validate_chunk(chunk: Chunk) -> ChunkValidationResult:
    try:
        ChunkModel(**asdict(chunk))
    except ValidationError as exc:
        reason = ", ".join(err.get("msg", "invalid") for err in exc.errors())
        return ChunkValidationResult(chunk_id=getattr(chunk, "chunk_id", ""), valid=False, reason=reason)
    return ChunkValidationResult(chunk_id=chunk.chunk_id, valid=True)


def ensure_valid_chunks(chunks: Sequence[Chunk]) -> None:
    for result in (validate_chunk(chunk) for chunk in chunks):
        if not result.valid:
            raise ChunkValidationError(result.reason or "invalid chunk")


def validate_chunks(chunks: Iterable[Chunk]) -> list[ChunkValidationResult]:
    return [validate_chunk(chunk) for chunk in chunks]


__all__ = ["ChunkValidationError", "ChunkValidationResult", "ensure_valid_chunks", "validate_chunks"]
