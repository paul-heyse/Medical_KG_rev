"""Stage contract protocols for Dagster-based orchestration.

This module provides typed boundaries around the ingestion pipeline so that
stage implementations can be swapped without forcing changes to the orchestration
engine.  The contracts mirror the requirements captured in the OpenSpec change
proposal and intentionally avoid importing Dagster so that the stage layer
remains framework agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

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

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation of the context."""

        return {
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "doc_id": self.doc_id,
            "correlation_id": self.correlation_id,
            "metadata": dict(self.metadata),
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StageContext:
        """Rehydrate a context from a mapping payload."""

        return cls(
            tenant_id=str(payload.get("tenant_id")),
            job_id=payload.get("job_id"),
            doc_id=payload.get("doc_id"),
            correlation_id=payload.get("correlation_id"),
            metadata=dict(payload.get("metadata", {})),
            pipeline_name=payload.get("pipeline_name"),
            pipeline_version=payload.get("pipeline_version"),
        )


@dataclass(slots=True)
class StageResultSnapshot:
    """Aggregated metadata describing a stage execution."""

    stage: str
    stage_type: str
    attempts: int | None = None
    duration_ms: int | None = None
    output_count: int | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_type": self.stage_type,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "output_count": self.output_count,
            "error": self.error,
        }


@dataclass(slots=True)
class PipelineState:
    """Strongly-typed representation of the orchestration pipeline state."""

    context: StageContext
    adapter_request: AdapterRequest
    payload: dict[str, Any] = field(default_factory=dict)
    payloads: tuple[RawPayload, ...] = ()
    document: Document | None = None
    chunks: tuple[Chunk, ...] = ()
    embedding_batch: EmbeddingBatch | None = None
    entities: tuple[Entity, ...] = ()
    claims: tuple[Claim, ...] = ()
    index_receipt: IndexReceipt | None = None
    graph_receipt: GraphWriteReceipt | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    stage_results: dict[str, StageResultSnapshot] = field(default_factory=dict)
    schema_version: str = "v1"
    job_id: str | None = None

    @classmethod
    def initialise(
        cls,
        *,
        context: StageContext,
        adapter_request: AdapterRequest,
        payload: Mapping[str, Any] | None = None,
    ) -> PipelineState:
        """Factory helper used during bootstrap to create a state instance."""

        return cls(
            context=context,
            adapter_request=adapter_request,
            payload=dict(payload or {}),
            job_id=context.job_id,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_payloads(self) -> tuple[RawPayload, ...]:
        return self.payloads

    def set_payloads(self, payloads: Sequence[RawPayload]) -> None:
        self.payloads = tuple(payloads)

    def require_payloads(self) -> tuple[RawPayload, ...]:
        if not self.payloads:
            raise ValueError("PipelineState requires payloads before parse stage execution")
        return self.payloads

    def has_document(self) -> bool:
        return self.document is not None

    def set_document(self, document: Document) -> None:
        self.document = document

    def require_document(self) -> Document:
        if self.document is None:
            raise ValueError("PipelineState does not contain a parsed document")
        return self.document

    def has_chunks(self) -> bool:
        return bool(self.chunks)

    def set_chunks(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = tuple(chunks)

    def require_chunks(self) -> tuple[Chunk, ...]:
        if not self.chunks:
            raise ValueError("PipelineState does not contain document chunks")
        return self.chunks

    def has_embeddings(self) -> bool:
        return self.embedding_batch is not None and bool(self.embedding_batch.vectors)

    def set_embedding_batch(self, batch: EmbeddingBatch) -> None:
        self.embedding_batch = batch

    def require_embedding_batch(self) -> EmbeddingBatch:
        if self.embedding_batch is None:
            raise ValueError("PipelineState does not contain embedding results")
        return self.embedding_batch

    def set_entities_and_claims(
        self,
        entities: Sequence[Entity],
        claims: Sequence[Claim],
    ) -> None:
        self.entities = tuple(entities)
        self.claims = tuple(claims)

    def has_entities(self) -> bool:
        return bool(self.entities)

    def has_claims(self) -> bool:
        return bool(self.claims)

    def require_entities(self) -> tuple[Entity, ...]:
        if not self.entities:
            raise ValueError("PipelineState does not contain extracted entities")
        return self.entities

    def require_claims(self) -> tuple[Claim, ...]:
        if not self.claims:
            raise ValueError("PipelineState does not contain extracted claims")
        return self.claims

    def set_index_receipt(self, receipt: IndexReceipt) -> None:
        self.index_receipt = receipt

    def set_graph_receipt(self, receipt: GraphWriteReceipt) -> None:
        self.graph_receipt = receipt

    def ensure_ready_for(self, stage_type: str) -> None:
        """Validate preconditions required by the requested stage type."""

        if stage_type in {"parse", "ir-validation"}:
            self.require_payloads()
        elif stage_type == "chunk":
            self.require_document()
        elif stage_type == "embed":
            self.require_chunks()
        elif stage_type == "index":
            self.require_embedding_batch()
        elif stage_type == "extract":
            self.require_document()
        elif stage_type == "knowledge-graph":
            # Extraction stages may legitimately produce empty collections but the
            # state must contain the tuple marker.
            if self.entities is None or self.claims is None:
                raise ValueError("PipelineState requires extraction outputs before KG stage")

    # ------------------------------------------------------------------
    # Stage bookkeeping
    # ------------------------------------------------------------------
    @staticmethod
    def _stage_state_key(stage_type: str) -> str:
        return {
            "ingest": "payloads",
            "parse": "document",
            "ir-validation": "document",
            "chunk": "chunks",
            "embed": "embedding_batch",
            "index": "index_receipt",
            "extract": "extraction",
            "knowledge-graph": "graph_receipt",
        }.get(stage_type, stage_type)

    def apply_stage_output(self, stage_type: str, stage_name: str, output: Any) -> None:
        """Persist a stage output onto the typed state structure."""

        key = self._stage_state_key(stage_type)
        if stage_type == "ingest":
            values = output or []
            if not isinstance(values, Sequence):
                raise TypeError("Ingest stage must return a sequence of payloads")
            self.set_payloads(values)
        elif stage_type in {"parse", "ir-validation"}:
            if not isinstance(output, Document):
                raise TypeError("Parse stages must return a Document instance")
            self.set_document(output)
        elif stage_type == "chunk":
            if not isinstance(output, Sequence):
                raise TypeError("Chunk stage must return a sequence of Chunk instances")
            self.set_chunks(output)
        elif stage_type == "embed":
            if not isinstance(output, EmbeddingBatch):
                raise TypeError("Embed stage must return an EmbeddingBatch")
            self.set_embedding_batch(output)
        elif stage_type == "index":
            if not isinstance(output, IndexReceipt):
                raise TypeError("Index stage must return an IndexReceipt")
            self.set_index_receipt(output)
        elif stage_type == "extract":
            if (
                not isinstance(output, tuple)
                or len(output) != 2
                or not isinstance(output[0], Sequence)
                or not isinstance(output[1], Sequence)
            ):
                raise TypeError("Extract stage must return a tuple of entity and claim sequences")
            entities, claims = output
            self.set_entities_and_claims(entities, claims)
        elif stage_type == "knowledge-graph":
            if not isinstance(output, GraphWriteReceipt):
                raise TypeError("Knowledge graph stage must return a GraphWriteReceipt")
            self.set_graph_receipt(output)
        else:
            self.metadata[key] = output

        self.stage_results[stage_name] = StageResultSnapshot(stage=stage_name, stage_type=stage_type)

    def infer_output_count(self, stage_type: str, output: Any) -> int:
        if output is None:
            return 0
        if stage_type in {"ingest", "chunk"} and isinstance(output, Sequence):
            return len(output)
        if stage_type in {"parse", "ir-validation"}:
            return 1
        if stage_type == "embed" and isinstance(output, EmbeddingBatch):
            return len(output.vectors)
        if stage_type == "index" and isinstance(output, IndexReceipt):
            return output.chunks_indexed
        if stage_type == "extract" and isinstance(output, tuple) and len(output) == 2:
            entities, claims = output
            entity_count = len(entities) if isinstance(entities, Sequence) else 0
            claim_count = len(claims) if isinstance(claims, Sequence) else 0
            return entity_count + claim_count
        if stage_type == "knowledge-graph" and isinstance(output, GraphWriteReceipt):
            return output.nodes_written
        return 1

    def record_stage_metrics(
        self,
        stage_name: str,
        *,
        stage_type: str | None = None,
        attempts: int | None = None,
        duration_ms: int | None = None,
        output_count: int | None = None,
        error: str | None = None,
    ) -> None:
        snapshot = self.stage_results.setdefault(
            stage_name,
            StageResultSnapshot(stage=stage_name, stage_type="unknown"),
        )
        if stage_type:
            snapshot.stage_type = stage_type
        snapshot.attempts = attempts
        snapshot.duration_ms = duration_ms
        snapshot.output_count = output_count
        snapshot.error = error

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def serialise(self) -> dict[str, Any]:
        """Return a metadata snapshot suitable for logging or Kafka payloads."""

        return {
            "version": self.schema_version,
            "job_id": self.job_id,
            "context": self.context.to_dict(),
            "adapter_request": self.adapter_request.model_dump(),
            "payload": dict(self.payload),
            "payload_count": len(self.payloads),
            "document_id": getattr(self.document, "id", None),
            "chunk_count": len(self.chunks),
            "embedding_count": len(self.embedding_batch.vectors)
            if self.embedding_batch
            else 0,
            "entity_count": len(self.entities),
            "claim_count": len(self.claims),
            "index_receipt": self.index_receipt.metadata if self.index_receipt else None,
            "graph_receipt": self.graph_receipt.metadata if self.graph_receipt else None,
            "stage_results": {name: result.as_dict() for name, result in self.stage_results.items()},
        }

    @classmethod
    def recover(
        cls,
        payload: Mapping[str, Any],
        *,
        context: StageContext,
        adapter_request: AdapterRequest,
    ) -> PipelineState:
        """Best-effort recovery for pipeline state snapshots."""

        state = cls.initialise(context=context, adapter_request=adapter_request, payload=payload.get("payload"))
        state.schema_version = str(payload.get("version", "v1"))
        state.job_id = payload.get("job_id") or context.job_id
        state.metadata.update(dict(payload.get("metadata", {})))
        return state


@runtime_checkable
class IngestStage(Protocol):
    """Fetch raw payloads from the configured adapter."""

    def execute(self, ctx: StageContext, state: PipelineState) -> list[RawPayload]: ...


@runtime_checkable
class ParseStage(Protocol):
    """Transform raw payloads into the canonical IR document."""

    def execute(self, ctx: StageContext, state: PipelineState) -> Document: ...


@runtime_checkable
class ChunkStage(Protocol):
    """Split an IR document into retrieval-ready chunks."""

    def execute(self, ctx: StageContext, state: PipelineState) -> list[Chunk]: ...


@runtime_checkable
class EmbedStage(Protocol):
    """Generate dense and/or sparse embeddings for a batch of chunks."""

    def execute(self, ctx: StageContext, state: PipelineState) -> EmbeddingBatch: ...


@runtime_checkable
class IndexStage(Protocol):
    """Persist embeddings into the vector and lexical indices."""

    def execute(self, ctx: StageContext, state: PipelineState) -> IndexReceipt: ...


@runtime_checkable
class ExtractStage(Protocol):
    """Run extraction models over the IR document."""

    def execute(self, ctx: StageContext, state: PipelineState) -> tuple[list[Entity], list[Claim]]: ...


@runtime_checkable
class KGStage(Protocol):
    """Write extracted entities and claims into the knowledge graph."""

    def execute(self, ctx: StageContext, state: PipelineState) -> GraphWriteReceipt: ...


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
    "PipelineState",
    "KGStage",
    "StageResultSnapshot",
    "ParseStage",
    "RawPayload",
    "StageContext",
]
