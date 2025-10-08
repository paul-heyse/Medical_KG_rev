"""Stage contract protocols for Dagster-based orchestration.

This module provides typed boundaries around the ingestion pipeline so that
stage implementations can be swapped without forcing changes to the orchestration
engine.  The contracts mirror the requirements captured in the OpenSpec change
proposal and intentionally avoid importing Dagster so that the stage layer
remains framework agnostic.
"""

from __future__ import annotations

import base64
import copy
import orjson
import structlog
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

from Medical_KG_rev.adapters.plugins.models import AdapterRequest
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.models.entities import Claim, Entity
from Medical_KG_rev.models.ir import Document
from Medical_KG_rev.observability.metrics import PIPELINE_STATE_SERIALISATIONS
from attrs import define, field as attrs_field
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass


RawPayload = dict[str, Any]


logger = structlog.get_logger(__name__)


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


@pydantic_dataclass(config=ConfigDict(validate_assignment=True))
class StageContext:
    """Immutable context shared across stage boundaries."""

    tenant_id: str
    job_id: str | None = None
    doc_id: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
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


@define(slots=True)
class StageResultSnapshot:
    """Aggregated metadata describing a stage execution."""

    stage: str
    stage_type: str
    attempts: int | None = attrs_field(default=None)
    duration_ms: int | None = attrs_field(default=None)
    output_count: int | None = attrs_field(default=None)
    error: str | None = attrs_field(default=None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_type": self.stage_type,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "output_count": self.output_count,
            "error": self.error,
        }


@define(slots=True, frozen=True)
class DownloadArtifact:
    """Represents a downloaded PDF artefact available for downstream stages."""

    document_id: str
    tenant_id: str
    uri: str
    metadata: dict[str, Any] = attrs_field(factory=dict)


@define(slots=True, frozen=True)
class GateDecision:
    """Represents the outcome of a gate stage evaluation."""

    name: str
    ready: bool
    metadata: dict[str, Any] = attrs_field(factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineStateSnapshot:
    """Immutable snapshot used for state rollback and diagnostics."""

    payloads: tuple[RawPayload, ...]
    document: Document | None
    chunks: tuple[Chunk, ...]
    embedding_batch: EmbeddingBatch | None
    entities: tuple[Entity, ...]
    claims: tuple[Claim, ...]
    index_receipt: IndexReceipt | None
    graph_receipt: GraphWriteReceipt | None
    downloads: tuple[DownloadArtifact, ...]
    gate_decisions: dict[str, GateDecision]
    metadata: dict[str, Any]
    stage_results: dict[str, StageResultSnapshot]
    job_id: str | None


class PipelineStateValidationError(ValueError):
    """Raised when the pipeline state fails validation."""

    def __init__(self, message: str, *, rule: str | None = None) -> None:
        super().__init__(message)
        self.rule = rule


class PipelineGateNotReady(RuntimeError):
    """Raised when a gate stage determines processing must pause."""

    def __init__(self, message: str, *, gate: str) -> None:
        super().__init__(message)
        self.gate = gate


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
    downloads: tuple[DownloadArtifact, ...] = ()
    gate_decisions: dict[str, GateDecision] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    stage_results: dict[str, StageResultSnapshot] = field(default_factory=dict)
    schema_version: str = "v1"
    job_id: str | None = None
    _dirty: bool = field(default=True, init=False, repr=False)
    _serialised_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _tenant_id: str = field(init=False, repr=False)
    _checkpoints: dict[str, PipelineStateSnapshot] = field(default_factory=dict, init=False, repr=False)

    _VALIDATORS: ClassVar[list[tuple[str | None, Callable[["PipelineState"], None]]]] = []

    def __post_init__(self) -> None:
        self._tenant_id = self.context.tenant_id

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
    def _mark_dirty(self) -> None:
        self._dirty = True
        self._serialised_cache = None

    def is_dirty(self) -> bool:
        """Return whether the state has pending changes since last snapshot."""

        return self._dirty

    def clone(self) -> "PipelineState":
        """Return a deep copy of the current pipeline state."""

        context_copy = StageContext.from_dict(self.context.to_dict())
        adapter_copy = self.adapter_request.model_copy(deep=True)
        clone = PipelineState.initialise(
            context=context_copy,
            adapter_request=adapter_copy,
            payload=copy.deepcopy(self.payload),
        )
        clone.payloads = tuple(copy.deepcopy(self.payloads))
        clone.document = copy.deepcopy(self.document)
        clone.chunks = tuple(copy.deepcopy(self.chunks))
        clone.embedding_batch = copy.deepcopy(self.embedding_batch)
        clone.entities = tuple(copy.deepcopy(self.entities))
        clone.claims = tuple(copy.deepcopy(self.claims))
        clone.index_receipt = copy.deepcopy(self.index_receipt)
        clone.graph_receipt = copy.deepcopy(self.graph_receipt)
        clone.downloads = tuple(copy.deepcopy(self.downloads))
        clone.gate_decisions = {
            name: copy.deepcopy(decision) for name, decision in self.gate_decisions.items()
        }
        clone.metadata = copy.deepcopy(self.metadata)
        clone.stage_results = {
            name: copy.deepcopy(result) for name, result in self.stage_results.items()
        }
        clone.schema_version = self.schema_version
        clone.job_id = self.job_id
        clone._dirty = self._dirty
        clone._serialised_cache = (
            copy.deepcopy(self._serialised_cache) if self._serialised_cache is not None else None
        )
        clone._checkpoints = {
            label: copy.deepcopy(snapshot) for label, snapshot in self._checkpoints.items()
        }
        return clone

    @property
    def tenant_id(self) -> str:
        """Return the tenant that owns the current state."""

        return self._tenant_id

    @staticmethod
    def _checkpoint_label(label: str | None) -> str:
        return label or "__default__"

    def create_checkpoint(
        self,
        label: str | None = None,
        *,
        include_stage_results: bool = True,
    ) -> PipelineStateSnapshot:
        """Capture and store a checkpoint snapshot for later rollback."""

        snapshot = self.snapshot(include_stage_results=include_stage_results)
        self._checkpoints[self._checkpoint_label(label)] = snapshot
        logger.debug(
            "pipeline_state.checkpoint_created",
            label=self._checkpoint_label(label),
            tenant_id=self._tenant_id,
        )
        return snapshot

    def get_checkpoint(self, label: str | None = None) -> PipelineStateSnapshot | None:
        """Return a previously captured checkpoint snapshot if it exists."""

        return self._checkpoints.get(self._checkpoint_label(label))

    def has_checkpoint(self, label: str | None = None) -> bool:
        return self._checkpoint_label(label) in self._checkpoints

    def rollback_to(
        self,
        label: str | None = None,
        *,
        restore_stage_results: bool = True,
    ) -> PipelineStateSnapshot | None:
        """Restore the pipeline to a stored checkpoint if available."""

        snapshot = self.get_checkpoint(label)
        if snapshot is not None:
            self.restore(snapshot, restore_stage_results=restore_stage_results)
            logger.debug(
                "pipeline_state.checkpoint_restored",
                label=self._checkpoint_label(label),
                tenant_id=self._tenant_id,
            )
        return snapshot

    def clear_checkpoint(self, label: str | None = None) -> None:
        """Drop a previously stored checkpoint."""

        self._checkpoints.pop(self._checkpoint_label(label), None)

    def clear_checkpoints(self) -> None:
        """Remove all stored checkpoints."""

        self._checkpoints.clear()

    def ensure_tenant_scope(self, tenant_id: str) -> None:
        """Validate that the state is being accessed by the owning tenant."""

        if tenant_id != self._tenant_id:
            raise PipelineStateValidationError(
                f"PipelineState initialised for tenant '{self._tenant_id}' cannot be reused for tenant '{tenant_id}'"
            )

    def get_payloads(self) -> tuple[RawPayload, ...]:
        return self.payloads

    def set_payloads(self, payloads: Sequence[RawPayload]) -> None:
        self.payloads = tuple(payloads)
        self._mark_dirty()

    def require_payloads(self) -> tuple[RawPayload, ...]:
        if not self.payloads:
            raise ValueError("PipelineState requires payloads before parse stage execution")
        return self.payloads

    def has_document(self) -> bool:
        return self.document is not None

    def set_document(self, document: Document) -> None:
        self.document = document
        self._mark_dirty()

    def require_document(self) -> Document:
        if self.document is None:
            raise ValueError("PipelineState does not contain a parsed document")
        return self.document

    def has_chunks(self) -> bool:
        return bool(self.chunks)

    def set_chunks(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = tuple(chunks)
        self._mark_dirty()

    def require_chunks(self) -> tuple[Chunk, ...]:
        if not self.chunks:
            raise ValueError("PipelineState does not contain document chunks")
        return self.chunks

    def has_embeddings(self) -> bool:
        return self.embedding_batch is not None and bool(self.embedding_batch.vectors)

    def set_embedding_batch(self, batch: EmbeddingBatch) -> None:
        self.embedding_batch = batch
        self._mark_dirty()

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
        self._mark_dirty()

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
        self._mark_dirty()

    def set_graph_receipt(self, receipt: GraphWriteReceipt) -> None:
        self.graph_receipt = receipt
        self._mark_dirty()

    def set_downloads(self, artifacts: Sequence[DownloadArtifact]) -> None:
        self.downloads = tuple(artifacts)
        self._mark_dirty()

    def require_downloads(self) -> tuple[DownloadArtifact, ...]:
        if not self.downloads:
            raise ValueError("PipelineState does not contain download artefacts")
        return self.downloads

    def record_gate_decision(self, decision: GateDecision) -> None:
        self.gate_decisions[decision.name] = decision
        self._mark_dirty()

    def get_gate_decision(self, name: str) -> GateDecision | None:
        return self.gate_decisions.get(name)

    def ensure_ready_for(self, stage_type: str) -> None:
        """Validate preconditions required by the requested stage type."""

        if stage_type in {"parse", "ir-validation"}:
            self.require_payloads()
        elif stage_type == "download":
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
        elif stage_type == "gate":
            self.require_downloads()

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
            "download": "downloads",
            "gate": "gate_decisions",
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
        elif stage_type == "download":
            if not isinstance(output, Sequence) or not all(
                isinstance(item, DownloadArtifact) for item in output
            ):
                raise TypeError("Download stage must return DownloadArtifact instances")
            self.set_downloads(output)
        elif stage_type == "gate":
            if not isinstance(output, GateDecision):
                raise TypeError("Gate stage must return a GateDecision")
            self.record_gate_decision(output)
        else:
            self.metadata[key] = output

        self.stage_results[stage_name] = StageResultSnapshot(stage=stage_name, stage_type=stage_type)
        self._mark_dirty()
        logger.debug(
            "pipeline_state.stage_output_applied",
            stage=stage_name,
            stage_type=stage_type,
            tenant_id=self._tenant_id,
        )

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
        if stage_type == "download" and isinstance(output, Sequence):
            return len(output)
        if stage_type == "gate" and isinstance(output, GateDecision):
            return int(output.ready)
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
        self._mark_dirty()

    def mark_stage_failed(
        self,
        stage_name: str,
        *,
        error: str,
        stage_type: str | None = None,
    ) -> None:
        """Record failure metadata for a stage."""

        self.record_stage_metrics(
            stage_name,
            stage_type=stage_type,
            attempts=None,
            duration_ms=None,
            output_count=None,
            error=error,
        )

    def cleanup_stage(self, stage_type: str) -> None:
        """Drop large stage outputs to allow garbage collection."""

        key = self._stage_state_key(stage_type)
        if key == "payloads":
            self.payloads = ()
        elif key == "document":
            self.document = None
        elif key == "chunks":
            self.chunks = ()
        elif key == "embedding_batch":
            self.embedding_batch = None
        elif key == "index_receipt":
            self.index_receipt = None
        elif key == "extraction":
            self.entities = ()
            self.claims = ()
        elif key == "graph_receipt":
            self.graph_receipt = None
        else:
            self.metadata.pop(key, None)
        self._mark_dirty()

    # ------------------------------------------------------------------
    # Snapshot & rollback helpers
    # ------------------------------------------------------------------
    def snapshot(self, *, include_stage_results: bool = True) -> PipelineStateSnapshot:
        """Capture an immutable snapshot for later rollback or diagnostics."""

        stage_payload: dict[str, StageResultSnapshot] = {}
        if include_stage_results:
            stage_payload = {
                name: copy.deepcopy(result)
                for name, result in self.stage_results.items()
            }
        return PipelineStateSnapshot(
            payloads=tuple(copy.deepcopy(self.payloads)),
            document=copy.deepcopy(self.document),
            chunks=tuple(copy.deepcopy(self.chunks)),
            embedding_batch=copy.deepcopy(self.embedding_batch),
            entities=tuple(copy.deepcopy(self.entities)),
            claims=tuple(copy.deepcopy(self.claims)),
            index_receipt=copy.deepcopy(self.index_receipt),
            graph_receipt=copy.deepcopy(self.graph_receipt),
            downloads=tuple(copy.deepcopy(self.downloads)),
            gate_decisions={
                name: copy.deepcopy(decision) for name, decision in self.gate_decisions.items()
            },
            metadata=copy.deepcopy(self.metadata),
            stage_results=stage_payload,
            job_id=self.job_id,
        )

    def restore(
        self,
        snapshot: PipelineStateSnapshot,
        *,
        restore_stage_results: bool = True,
    ) -> None:
        """Restore the state from a previously captured snapshot."""

        self.payloads = tuple(copy.deepcopy(snapshot.payloads))
        self.document = copy.deepcopy(snapshot.document)
        self.chunks = tuple(copy.deepcopy(snapshot.chunks))
        self.embedding_batch = copy.deepcopy(snapshot.embedding_batch)
        self.entities = tuple(copy.deepcopy(snapshot.entities))
        self.claims = tuple(copy.deepcopy(snapshot.claims))
        self.index_receipt = copy.deepcopy(snapshot.index_receipt)
        self.graph_receipt = copy.deepcopy(snapshot.graph_receipt)
        self.downloads = tuple(copy.deepcopy(snapshot.downloads))
        self.gate_decisions = {
            name: copy.deepcopy(decision) for name, decision in snapshot.gate_decisions.items()
        }
        self.metadata = copy.deepcopy(snapshot.metadata)
        if restore_stage_results:
            self.stage_results = {
                name: copy.deepcopy(result) for name, result in snapshot.stage_results.items()
            }
        self.job_id = snapshot.job_id
        self._mark_dirty()

    def rollback(self, snapshot: PipelineStateSnapshot) -> None:
        """Alias for :meth:`restore` to emphasise rollback semantics."""

        self.restore(snapshot)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def serialise(
        self,
        *,
        include_stage_results: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Return a metadata snapshot suitable for logging or Kafka payloads."""

        if use_cache and not self._dirty and self._serialised_cache is not None:
            return copy.deepcopy(self._serialised_cache)

        snapshot: dict[str, Any] = {
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
            "download_count": len(self.downloads),
            "gate_status": {name: decision.ready for name, decision in self.gate_decisions.items()},
        }
        if include_stage_results:
            snapshot["stage_results"] = {
                name: result.as_dict() for name, result in self.stage_results.items()
            }
        if use_cache:
            self._serialised_cache = snapshot
            self._dirty = False
        serialised = copy.deepcopy(snapshot)
        PIPELINE_STATE_SERIALISATIONS.inc()
        logger.debug(
            "pipeline_state.serialised",
            tenant_id=self._tenant_id,
            payload_count=len(self.payloads),
            stage_count=len(self.stage_results),
        )
        return serialised

    def serialise_json(self) -> str:
        """Return a JSON encoded snapshot of the state."""

        return orjson.dumps(self.serialise()).decode("utf-8")

    def serialise_compressed(self) -> bytes:
        """Compress the JSON snapshot for efficient transport."""

        return zlib.compress(self.serialise_json().encode("utf-8"))

    def serialise_base64(self) -> str:
        """Return a base64 encoded compressed snapshot."""

        return base64.b64encode(self.serialise_compressed()).decode("ascii")

    def diff(self, other: PipelineState) -> dict[str, tuple[Any, Any]]:
        """Produce a minimal diff between two states."""

        entries: dict[str, tuple[Any, Any]] = {}
        if len(self.payloads) != len(other.payloads):
            entries["payload_count"] = (len(self.payloads), len(other.payloads))
        if len(self.chunks) != len(other.chunks):
            entries["chunk_count"] = (len(self.chunks), len(other.chunks))
        self_embeddings = (
            len(self.embedding_batch.vectors) if self.embedding_batch else 0
        )
        other_embeddings = (
            len(other.embedding_batch.vectors) if other.embedding_batch else 0
        )
        if self_embeddings != other_embeddings:
            entries["embedding_count"] = (self_embeddings, other_embeddings)
        if len(self.entities) != len(other.entities):
            entries["entity_count"] = (len(self.entities), len(other.entities))
        if len(self.claims) != len(other.claims):
            entries["claim_count"] = (len(self.claims), len(other.claims))
        if len(self.downloads) != len(other.downloads):
            entries["download_count"] = (len(self.downloads), len(other.downloads))
        self_gates = {name: decision.ready for name, decision in self.gate_decisions.items()}
        other_gates = {name: decision.ready for name, decision in other.gate_decisions.items()}
        if self_gates != other_gates:
            entries["gate_status"] = (self_gates, other_gates)
        if self.context.pipeline_version != other.context.pipeline_version:
            entries["pipeline_version"] = (
                self.context.pipeline_version,
                other.context.pipeline_version,
            )
        if self.job_id != other.job_id:
            entries["job_id"] = (self.job_id, other.job_id)
        return entries

    @classmethod
    def recover(
        cls,
        payload: Mapping[str, Any] | bytes | str,
        *,
        context: StageContext,
        adapter_request: AdapterRequest,
    ) -> PipelineState:
        """Best-effort recovery for pipeline state snapshots."""

        if isinstance(payload, (bytes, bytearray)):
            decoded = zlib.decompress(bytes(payload))
            recovered = orjson.loads(decoded)
        elif isinstance(payload, str):
            try:
                compressed = base64.b64decode(payload)
            except (ValueError, TypeError):
                recovered = orjson.loads(payload)
            else:
                decoded = zlib.decompress(compressed)
                recovered = orjson.loads(decoded)
        else:
            recovered = payload

        if not isinstance(recovered, Mapping):
            recovered = {}

        state = cls.initialise(
            context=context,
            adapter_request=adapter_request,
            payload=recovered.get("payload"),
        )
        state.schema_version = str(recovered.get("version", "v1"))
        state.job_id = recovered.get("job_id") or context.job_id
        state.metadata.update(dict(recovered.get("metadata", {})))
        stage_payload = recovered.get("stage_results")
        if isinstance(stage_payload, Mapping):
            for name, payload_data in stage_payload.items():
                if isinstance(payload_data, Mapping):
                    state.stage_results[name] = StageResultSnapshot(
                        stage=str(payload_data.get("stage", name)),
                        stage_type=str(payload_data.get("stage_type", "unknown")),
                        attempts=payload_data.get("attempts"),
                        duration_ms=payload_data.get("duration_ms"),
                        output_count=payload_data.get("output_count"),
                        error=payload_data.get("error"),
                    )
        gates = recovered.get("gate_status")
        if isinstance(gates, Mapping):
            for name, ready in gates.items():
                state.gate_decisions[str(name)] = GateDecision(
                    name=str(name), ready=bool(ready)
                )
        state._dirty = False
        state._serialised_cache = copy.deepcopy(dict(recovered))
        return state

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @classmethod
    def register_validator(
        cls,
        validator: Callable[["PipelineState"], None],
        *,
        name: str | None = None,
    ) -> None:
        cls._VALIDATORS.append((name, validator))

    @classmethod
    def clear_validators(cls) -> None:
        cls._VALIDATORS.clear()

    def validate(
        self,
        *,
        extra_rules: Sequence[Callable[["PipelineState"], None]] | None = None,
    ) -> None:
        """Run registered validators against the state."""

        for name, validator in self._VALIDATORS:
            try:
                validator(self)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise PipelineStateValidationError(str(exc), rule=name) from exc
        if extra_rules:
            for rule in extra_rules:
                try:
                    rule(self)
                except Exception as exc:  # pragma: no cover - defensive guard
                    raise PipelineStateValidationError(str(exc)) from exc

    def validate_transition(self, stage_type: str) -> None:
        """Ensure the state is ready for the requested stage transition."""

        try:
            self.ensure_ready_for(stage_type)
        except ValueError as exc:
            raise PipelineStateValidationError(
                f"State missing prerequisites for stage '{stage_type}': {exc}"
            ) from exc


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


@runtime_checkable
class DownloadStage(Protocol):
    """Download raw assets required for downstream processing."""

    def execute(self, ctx: StageContext, state: PipelineState) -> list[DownloadArtifact]: ...


@runtime_checkable
class GateStage(Protocol):
    """Enforce conditional progression based on external readiness signals."""

    def execute(self, ctx: StageContext, state: PipelineState) -> GateDecision: ...


__all__ = [
    "ChunkStage",
    "EmbedStage",
    "EmbeddingBatch",
    "EmbeddingVector",
    "DownloadArtifact",
    "GateDecision",
    "ExtractStage",
    "GraphWriteReceipt",
    "IngestStage",
    "IndexReceipt",
    "IndexStage",
    "DownloadStage",
    "GateStage",
    "PipelineGateNotReady",
    "PipelineState",
    "PipelineStateSnapshot",
    "PipelineStateValidationError",
    "KGStage",
    "StageResultSnapshot",
    "ParseStage",
    "RawPayload",
    "StageContext",
]
