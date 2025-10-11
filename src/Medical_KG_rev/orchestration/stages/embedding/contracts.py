"""Typed contracts for embedding operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddingRequest(BaseModel):
    """Typed request for embedding generation."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    texts: tuple[str, ...] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text chunks to embed"
    )

    namespace: str = Field(
        ...,
        pattern=r"^[a-z0-9_]+$",
        min_length=1,
        max_length=100,
        description="Embedding namespace"
    )

    model_id: str = Field(
        ...,
        min_length=1,
        description="Embedding model identifier"
    )

    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for tracing"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for processing"
    )

    @field_validator("texts")
    @classmethod
    def validate_text_length(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Validate individual text length."""
        for i, text in enumerate(v):
            if len(text) > 10000:
                raise ValueError(
                    f"Text at index {i} exceeds 10000 chars: {len(text)}"
                )
            if not text.strip():
                raise ValueError(f"Text at index {i} is empty or whitespace only")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Limit metadata size to prevent memory issues."""
        if len(str(v)) > 10000:
            raise ValueError("Metadata too large (>10KB)")
        return v


class EmbeddingVector(BaseModel):
    """Single embedding vector."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    vector: tuple[float, ...] = Field(..., min_length=1)
    model_id: str
    namespace: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    """Typed result from embedding generation."""

    model_config = ConfigDict(frozen=True)

    vectors: tuple[EmbeddingVector, ...] = Field(..., min_length=1)
    model_id: str
    namespace: str
    processing_time_ms: float = Field(..., ge=0)
    gpu_memory_used_mb: int | None = Field(default=None, ge=0)
    correlation_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def vector_count(self) -> int:
        return len(self.vectors)

    @property
    def per_namespace_counts(self) -> dict[str, int]:
        """Count vectors by namespace."""
        counts: dict[str, int] = {}
        for vec in self.vectors:
            counts[vec.namespace] = counts.get(vec.namespace, 0) + 1
        return counts


class EmbeddingError(Exception):
    """Base exception for embedding operations."""

    def __init__(self, message: str, *, correlation_id: str | None = None):
        super().__init__(message)
        self.correlation_id = correlation_id


class EmbeddingValidationError(EmbeddingError):
    """Validation failed for embedding request."""

    def __init__(self, message: str, *, field: str | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.field = field


class EmbeddingProcessingError(EmbeddingError):
    """Processing failed during embedding generation."""

    def __init__(
        self,
        message: str,
        *,
        retry_possible: bool = False,
        error_type: str = "unknown",
        **kwargs: Any
    ):
        super().__init__(message, **kwargs)
        self.retry_possible = retry_possible
        self.error_type = error_type


class EmbeddingServiceUnavailableError(EmbeddingProcessingError):
    """Embedding service is unavailable."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(
            message,
            retry_possible=True,
            error_type="service_unavailable",
            **kwargs
        )
