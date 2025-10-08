"""Stage contract protocols for Dagster-based orchestration.

This module provides typed boundaries around the ingestion pipeline so that
stage implementations can be swapped without forcing changes to the orchestration
engine.  The contracts mirror the requirements captured in the OpenSpec change
proposal and intentionally avoid importing Dagster so that the stage layer
remains framework agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from Medical_KG_rev.adapters.plugins.models import AdapterRequest
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.models.entities import Claim, Entity
from Medical_KG_rev.models.ir import Document


RawPayload = dict[str, Any]


@dataclass(slots=True)
class EmbeddingVector:
    """Represents a single embedding vector produced during the embed stage."""

    id: str
    values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddingBatch:
    """Container returned by the embed stage."""

    vectors: tuple[EmbeddingVector, ...]
    model: str
    tenant_id: str


@dataclass(slots=True)
class IndexReceipt:
    """Acknowledgement returned by the index stage."""

    chunks_indexed: int
    opensearch_ok: bool
    faiss_ok: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphWriteReceipt:
    """Result returned by the knowledge graph stage."""

    nodes_written: int
    edges_written: int
    correlation_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageContext:
    """Immutable context shared across stage boundaries."""

    tenant_id: str
    job_id: str | None = None
    doc_id: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pipeline_name: str | None = None
    pipeline_version: str | None = None

    def with_metadata(self, **values: Any) -> StageContext:
        """Return a new context instance with additional metadata."""

        updated = dict(self.metadata)
        updated.update(values)
        return StageContext(
            tenant_id=self.tenant_id,
            job_id=self.job_id,
            doc_id=self.doc_id,
            correlation_id=self.correlation_id,
            metadata=updated,
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
        )


@runtime_checkable
class IngestStage(Protocol):
    """Fetch raw payloads from the configured adapter."""

    def execute(self, ctx: StageContext, request: AdapterRequest) -> list[RawPayload]: ...


@runtime_checkable
class ParseStage(Protocol):
    """Transform raw payloads into the canonical IR document."""

    def execute(self, ctx: StageContext, payloads: list[RawPayload]) -> Document: ...


@runtime_checkable
class ChunkStage(Protocol):
    """Split an IR document into retrieval-ready chunks."""

    def execute(self, ctx: StageContext, document: Document) -> list[Chunk]: ...


@runtime_checkable
class EmbedStage(Protocol):
    """Generate dense and/or sparse embeddings for a batch of chunks."""

    def execute(self, ctx: StageContext, chunks: list[Chunk]) -> EmbeddingBatch: ...


@runtime_checkable
class IndexStage(Protocol):
    """Persist embeddings into the vector and lexical indices."""

    def execute(self, ctx: StageContext, batch: EmbeddingBatch) -> IndexReceipt: ...


@runtime_checkable
class ExtractStage(Protocol):
    """Run extraction models over the IR document."""

    def execute(self, ctx: StageContext, document: Document) -> tuple[list[Entity], list[Claim]]: ...


@runtime_checkable
class KGStage(Protocol):
    """Write extracted entities and claims into the knowledge graph."""

    def execute(self, ctx: StageContext, entities: list[Entity], claims: list[Claim]) -> GraphWriteReceipt: ...


__all__ = [
    "ChunkStage",
    "EmbedStage",
    "EmbeddingBatch",
    "EmbeddingVector",
    "ExtractStage",
    "GraphWriteReceipt",
    "IngestStage",
    "IndexReceipt",
    "IndexStage",
    "KGStage",
    "ParseStage",
    "RawPayload",
    "StageContext",
]
