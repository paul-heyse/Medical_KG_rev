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
import json as _json
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import util as importlib_util
from typing import Any, ClassVar, Protocol, runtime_checkable

import structlog
from prometheus_client import Counter, Histogram

if importlib_util.find_spec("pydantic") is not None:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError  # type: ignore[import]

    PYDANTIC_AVAILABLE = True
else:  # pragma: no cover - fallback for lightweight environments
    PYDANTIC_AVAILABLE = False

    class ValidationError(ValueError):
        """Fallback validation error when Pydantic is unavailable."""

    class BaseModel:  # type: ignore[override]
        model_config: dict[str, Any] = {}

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> BaseModel:
            payload = self.model_dump(mode="python")
            if update:
                payload.update(update)
            return self.__class__(**payload)

        def model_dump(self, mode: str = "python") -> dict[str, Any]:  # pragma: no cover - trivial shim
            return {k: copy.deepcopy(v) for k, v in self.__dict__.items() if not k.startswith("_")}

        @classmethod
        def model_validate(cls, payload: Mapping[str, Any]) -> BaseModel:
            return cls(**dict(payload))

    def Field(*, default: Any | None = None, default_factory: Callable[[], Any] | None = None) -> Any:  # type: ignore[override]
        if default_factory is not None:
            return default_factory()
        return default

    def ConfigDict(**kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return dict(kwargs)

if PYDANTIC_AVAILABLE:
    from Medical_KG_rev.adapters.plugins.models import AdapterRequest  # type: ignore[import]
else:  # pragma: no cover - fallback for lightweight test environments

    @dataclass(slots=True)
    class AdapterRequest:
        tenant_id: str
        correlation_id: str | None = None
        domain: str | None = None
        parameters: dict[str, Any] = field(default_factory=dict)

        @classmethod
        def model_validate(cls, payload: Mapping[str, Any]) -> AdapterRequest:
            return cls(
                tenant_id=str(payload.get("tenant_id", "")),
                correlation_id=payload.get("correlation_id"),
                domain=payload.get("domain"),
                parameters=dict(payload.get("parameters", {})),
            )

        def model_copy(self, *, deep: bool = False) -> AdapterRequest:
            return AdapterRequest(
                tenant_id=self.tenant_id,
                correlation_id=self.correlation_id,
                domain=self.domain,
                parameters=copy.deepcopy(self.parameters) if deep else dict(self.parameters),
            )

        def model_dump(self, mode: str = "json") -> dict[str, Any]:  # pragma: no cover - simple shim
            return {
                "tenant_id": self.tenant_id,
                "correlation_id": self.correlation_id,
                "domain": self.domain,
                "parameters": copy.deepcopy(self.parameters),
            }


if PYDANTIC_AVAILABLE:
    from Medical_KG_rev.chunking.models import Chunk  # type: ignore[import]
else:  # pragma: no cover - fallback shim

    @dataclass(slots=True)
    class Chunk:
        chunk_id: str
        doc_id: str
        tenant_id: str
        body: str
        title_path: tuple[str, ...]
        section: str
        start_char: int
        end_char: int
        granularity: str
        chunker: str
        chunker_version: str

        @classmethod
        def model_validate(cls, payload: Mapping[str, Any]) -> Chunk:
            return cls(
                chunk_id=str(payload.get("chunk_id")),
                doc_id=str(payload.get("doc_id")),
                tenant_id=str(payload.get("tenant_id", "")),
                body=str(payload.get("body", "")),
                title_path=tuple(payload.get("title_path", ())),
                section=str(payload.get("section", "")),
                start_char=int(payload.get("start_char", 0)),
                end_char=int(payload.get("end_char", 0)),
                granularity=str(payload.get("granularity", "")),
                chunker=str(payload.get("chunker", "")),
                chunker_version=str(payload.get("chunker_version", "")),
            )

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return asdict(self)


if PYDANTIC_AVAILABLE:
    from Medical_KG_rev.models.entities import Claim, Entity  # type: ignore[import]
else:  # pragma: no cover - fallback shim

    @dataclass(slots=True)
    class Entity:
        id: str
        type: str
        text: str

        @classmethod
        def model_validate(cls, payload: Mapping[str, Any]) -> Entity:
            return cls(id=str(payload.get("id", "")), type=str(payload.get("type", "")), text=str(payload.get("text", "")))

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return asdict(self)


    @dataclass(slots=True)
    class Claim:
        id: str
        statement: str

        @classmethod
        def model_validate(cls, payload: Mapping[str, Any]) -> Claim:
            return cls(id=str(payload.get("id", "")), statement=str(payload.get("statement", "")))

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return asdict(self)


if PYDANTIC_AVAILABLE:
    from Medical_KG_rev.models.ir import Document  # type: ignore[import]
else:  # pragma: no cover - fallback shim

    @dataclass(slots=True)
    class Document:
        id: str
        source: str
        sections: tuple[Any, ...] = ()
        metadata: dict[str, Any] = field(default_factory=dict)

        @classmethod
        def model_validate(cls, payload: Mapping[str, Any]) -> Document:
            return cls(
                id=str(payload.get("id", "")),
                source=str(payload.get("source", "")),
                sections=tuple(payload.get("sections", ())),
                metadata=dict(payload.get("metadata", {})),
            )

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "id": self.id,
                "source": self.source,
                "sections": list(self.sections),
                "metadata": copy.deepcopy(self.metadata),
            }
from dataclasses import dataclass, field

import orjson
from attrs import asdict as attr_asdict
from attrs import define
from attrs import define as attr_define
from attrs import field as attr_field
from attrs import field as attrs_field
from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

from Medical_KG_rev.adapters.plugins.models import AdapterRequest
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.models.entities import Claim, Entity
from Medical_KG_rev.models.ir import Document
from Medical_KG_rev.observability.metrics import PIPELINE_STATE_SERIALISATIONS
from Medical_KG_rev.orchestration.state import (
    PipelineStateCache,
    PipelineStateModel,
    dumps_json,
    dumps_orjson,
    encode_base64,
    record_stage_metrics,
    serialise_payload,
)

RawPayload = dict[str, Any]


_state_logger = structlog.get_logger("Medical_KG_rev.pipeline_state")

_STATE_MUTATIONS = Counter(
    "orchestration_pipeline_state_mutations_total",
    "Total number of mutating operations executed against PipelineState",
    labelnames=("operation",),
)
_STATE_SERIALISATION_LATENCY = Histogram(
    "orchestration_pipeline_state_serialisation_seconds",
    "Latency of PipelineState serialisation operations",
    labelnames=("format",),
    buckets=(
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
    ),
)
_STATE_DEPENDENCY_FAILURES = Counter(
    "orchestration_pipeline_state_dependency_failures_total",
    "Number of times a stage attempted to execute without satisfying dependencies",
    labelnames=("stage_type",),
)


if importlib_util.find_spec("orjson") is not None:
    import orjson as _orjson  # type: ignore[import]
    ORJSON_AVAILABLE = True
else:  # pragma: no cover - fallback path for minimal environments
    ORJSON_AVAILABLE = False

    class _OrjsonShim:
        JSONDecodeError = ValueError

        @staticmethod
        def dumps(payload: Any) -> bytes:
            return _json.dumps(payload).encode("utf-8")

        @staticmethod
        def loads(data: Any) -> Any:
            if isinstance(data, (bytes, bytearray)):
                return _json.loads(bytes(data).decode("utf-8"))
            return _json.loads(data)

    _orjson = _OrjsonShim()


if importlib_util.find_spec("pydantic") is not None:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError  # type: ignore[import]

    PYDANTIC_AVAILABLE = True
else:  # pragma: no cover - fallback for lightweight environments
    PYDANTIC_AVAILABLE = False

    class ValidationError(ValueError):
        """Fallback validation error when Pydantic is unavailable."""

    class BaseModel:  # type: ignore[override]
        """Minimal shim providing the interface we rely on for tests."""

        model_config: dict[str, Any] = {}

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> BaseModel:
            payload = self.model_dump(mode="python")
            if update:
                payload.update(update)
            return self.__class__(**payload)

        def model_dump(self, mode: str = "python") -> dict[str, Any]:  # pragma: no cover - trivial shim
            return {k: copy.deepcopy(v) for k, v in self.__dict__.items() if not k.startswith("_")}

        @classmethod
        def model_validate(cls, payload: Mapping[str, Any]) -> BaseModel:
            return cls(**dict(payload))

    def Field(*, default: Any | None = None, default_factory: Callable[[], Any] | None = None) -> Any:  # type: ignore[override]
        if default_factory is not None:
            return default_factory()
        return default

    def ConfigDict(**kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return dict(kwargs)


if importlib_util.find_spec("attrs") is not None:
    from attrs import asdict as attr_asdict  # type: ignore[import]
    from attrs import define as attr_define  # type: ignore[import]
    from attrs import field as attr_field  # type: ignore[import]
else:  # pragma: no cover - fallback for environments without attrs
    attr_asdict = asdict

    def attr_define(*, slots: bool = False):  # type: ignore[override]
        def _decorator(cls):
            return dataclass(cls, slots=slots)

        return _decorator

    def attr_field(*, default: Any | None = None, factory: Callable[[], Any] | None = None):  # type: ignore[override]
        if factory is not None:
            return field(default_factory=factory)
        return field(default=default)


if importlib_util.find_spec("tenacity") is not None:
    from tenacity import (  # type: ignore[import]
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
else:  # pragma: no cover - fallback for environments without tenacity

    def stop_after_attempt(count: int) -> int:
        return max(1, int(count))

    def wait_exponential(**_: Any) -> None:  # pragma: no cover - compatibility shim
        return None

    def retry_if_exception_type(exc_type: type[BaseException]) -> type[BaseException]:  # pragma: no cover - shim
        return exc_type

    def retry(*, reraise: bool | None = None, stop: Any | None = None, wait: Any | None = None, retry: Any | None = None):
        max_attempts = stop if isinstance(stop, int) else 1

        def _decorator(func):
            def _wrapped(*args, **kwargs):
                attempts = 0
                while True:
                    try:
                        attempts += 1
                        return func(*args, **kwargs)
                    except Exception:  # pragma: no cover - simple retry shim
                        if attempts >= max_attempts:
                            raise

            return _wrapped

        return _decorator


@attr_define(slots=True)
class PDFStateTracker:
    """Track PDF download and gate status for long-running ingest jobs."""

    downloads: tuple[str, ...] = ()
    last_attempt_ms: int | None = None
    gate_open: bool = False
    gate_reason: str | None = None
    ledger_reference: str | None = None

    def record_download(self, asset_id: str, *, attempt_ms: int | None = None) -> None:
        if asset_id not in self.downloads:
            self.downloads = (*self.downloads, asset_id)
        if attempt_ms is not None:
            self.last_attempt_ms = attempt_ms

    def mark_gate(self, *, opened: bool, reason: str | None = None) -> None:
        self.gate_open = opened
        self.gate_reason = reason

    def set_ledger_reference(self, reference: str | None) -> None:
        self.ledger_reference = reference

    def as_dict(self) -> dict[str, Any]:
        return {
            "downloads": list(self.downloads),
            "last_attempt_ms": self.last_attempt_ms,
            "gate_open": self.gate_open,
            "gate_reason": self.gate_reason,
            "ledger_reference": self.ledger_reference,
        }


if PYDANTIC_AVAILABLE:

    class StageContext(BaseModel):
        """Immutable context shared across stage boundaries."""

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

        tenant_id: str
        job_id: str | None = None
        doc_id: str | None = None
        correlation_id: str | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)
        pipeline_name: str | None = None
        pipeline_version: str | None = None

        def with_metadata(self, **values: Any) -> StageContext:
            """Return a new context instance with additional metadata."""
            updated = {**self.metadata, **values}
            return self.model_copy(update={"metadata": updated})

        def to_dict(self) -> dict[str, Any]:
            """Return a serialisable representation of the context."""
            return self.model_dump(mode="json")

        @classmethod
        def from_dict(cls, payload: Mapping[str, Any]) -> StageContext:
            """Rehydrate a context from a mapping payload."""
            return cls.model_validate(payload)

else:

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
            return cls(
                tenant_id=str(payload.get("tenant_id")),
                job_id=payload.get("job_id"),
                doc_id=payload.get("doc_id"),
                correlation_id=payload.get("correlation_id"),
                metadata=dict(payload.get("metadata", {})),
                pipeline_name=payload.get("pipeline_name"),
                pipeline_version=payload.get("pipeline_version"),
            )

        def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> StageContext:
            payload = self.to_dict()
            if update:
                payload.update(update)
            return StageContext.from_dict(payload)

        def model_dump(self, mode: str = "python") -> dict[str, Any]:
            return self.to_dict()

        @classmethod
        def model_validate(cls, payload: Mapping[str, Any]) -> StageContext:
            return cls.from_dict(payload)
logger = structlog.get_logger(__name__)
_STATE_SERIALISE_COUNTER = Counter(
    "medical_kg_pipeline_state_serialise_total",
    "Total pipeline state serialisation attempts",
    ("format",),
)
_STATE_SERIALISE_LATENCY = Histogram(
    "medical_kg_pipeline_state_serialise_seconds",
    "Pipeline state serialisation latency",
    ("format",),
)
_STATE_PERSIST_COUNTER = Counter(
    "medical_kg_pipeline_state_persist_total",
    "Pipeline state persistence attempts",
    ("format", "status"),
)
_state_logger = structlog.get_logger(__name__).bind(component="PipelineState")


class PdfAssetModel(BaseModel):
    asset_id: str
    uri: str
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@attr_define(slots=True)
class PdfAsset:
    """Representation of a downloaded PDF asset associated with a job."""

    asset_id: str
    uri: str
    checksum: str | None = None
    metadata: dict[str, Any] = attr_field(factory=dict)


@attr_define(slots=True)
class _StateCache:
    payload: dict[str, Any] | None = None
    json_bytes: bytes | None = None
    compressed: bytes | None = None
    base64_payload: str | None = None


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


@attr_define(slots=True)
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
        return attr_asdict(self)


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
@dataclass(slots=True)
class PdfGateState:
    """Represents the progress of the PDF two-phase pipeline gate."""

    downloaded: bool = False
    ir_ready: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "downloaded": self.downloaded,
            "ir_ready": self.ir_ready,
            "metadata": dict(self.metadata),
        }

    def merge_metadata(self, values: Mapping[str, Any] | None) -> None:
        if not values:
            return
        self.metadata.update(dict(values))


@dataclass(slots=True)
class StagePerformanceSample:
    """Profiled metrics captured during stage lifecycle notifications."""

    stage: str
    stage_type: str
    duration_ms: int
    measured_ms: int
    attempts: int
    output_count: int
    timestamp: float


@dataclass(slots=True)
class PipelineStateLifecycleHook:
    """Callback hooks executed when stages progress."""

    on_started: Callable[[PipelineState, str, str], None] | None = None
    on_completed: Callable[[
        PipelineState,
        str,
        str,
        int,
        int,
        int,
    ], None] | None = None
    on_failed: Callable[[PipelineState, str, str, BaseException], None] | None = None


@dataclass(slots=True)
class PipelineStateProfiler:
    """Collects lightweight profiling statistics for pipeline stages."""

    samples: list[StagePerformanceSample] = field(default_factory=list)
    _inflight: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def on_stage_started(self, _state: PipelineState, stage: str, stage_type: str) -> None:
        self._inflight[stage] = time.perf_counter()

    def on_stage_completed(
        self,
        _state: PipelineState,
        stage: str,
        stage_type: str,
        duration_ms: int,
        attempts: int,
        output_count: int,
    ) -> None:
        start = self._inflight.pop(stage, None)
        measured_ms = duration_ms
        if start is not None:
            measured_ms = int((time.perf_counter() - start) * 1000)
        self.samples.append(
            StagePerformanceSample(
                stage=stage,
                stage_type=stage_type,
                duration_ms=duration_ms,
                measured_ms=measured_ms,
                attempts=attempts,
                output_count=output_count,
                timestamp=time.time(),
            )
        )

    def on_stage_failed(
        self,
        _state: PipelineState,
        stage: str,
        _stage_type: str,
        _error: BaseException,
    ) -> None:
        self._inflight.pop(stage, None)

    def summary(self) -> dict[str, Any]:
        totals: dict[str, dict[str, Any]] = {}
        for sample in self.samples:
            entry = totals.setdefault(
                sample.stage,
                {
                    "stage_type": sample.stage_type,
                    "count": 0,
                    "duration_ms": 0,
                    "measured_ms": 0,
                    "attempts": 0,
                    "outputs": 0,
                },
            )
            entry["count"] += 1
            entry["duration_ms"] += sample.duration_ms
            entry["measured_ms"] += sample.measured_ms
            entry["attempts"] += sample.attempts
            entry["outputs"] += sample.output_count
        return {
            "total_stages": len(self.samples),
            "stages": totals,
        }


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
    pdf_assets: tuple[PdfAsset, ...]
    gate_status: dict[str, bool]
    metadata: dict[str, Any]
    stage_results: dict[str, StageResultSnapshot]
    job_id: str | None
    pdf_tracker: PDFStateTracker
    pdf_gate: PdfGateState


class PipelineStateValidationError(ValueError):
    """Raised when the pipeline state fails validation."""

    def __init__(self, message: str, *, rule: str | None = None) -> None:
        super().__init__(message)
        self.rule = rule


if PYDANTIC_AVAILABLE:

    class PipelineStateEnvelope(BaseModel):
        """Pydantic schema enforcing PipelineState invariants."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        context: StageContext
        adapter_request: AdapterRequest
        payload: dict[str, Any] = Field(default_factory=dict)
        payloads: tuple[RawPayload, ...] = ()
        document: Document | None = None
        chunks: tuple[Chunk, ...] = ()
        embedding_batch: EmbeddingBatch | None = None
        entities: tuple[Entity, ...] = ()
        claims: tuple[Claim, ...] = ()
        index_receipt: IndexReceipt | None = None
        graph_receipt: GraphWriteReceipt | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)
        stage_results: dict[str, StageResultSnapshot] = Field(default_factory=dict)
        schema_version: str = "v1"
        job_id: str | None = None
        pdf_tracker: PDFStateTracker = Field(default_factory=PDFStateTracker)

else:

    class PipelineStateEnvelope:
        """Fallback schema shim when Pydantic is not installed."""

        @staticmethod
        def model_validate(payload: Mapping[str, Any]) -> Mapping[str, Any]:  # pragma: no cover - trivial
            return payload
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
    pdf_assets: tuple[PdfAsset, ...] = ()
    gate_status: dict[str, bool] = field(default_factory=dict)
    schema_version: str = "v1"
    job_id: str | None = None
    pdf_tracker: PDFStateTracker = field(default_factory=PDFStateTracker)
    pdf_gate: PdfGateState = field(default_factory=PdfGateState)
    _dirty: bool = field(default=True, init=False, repr=False)
    _cache: _StateCache = field(default_factory=_StateCache, init=False, repr=False)
    _tenant_id: str = field(init=False, repr=False)
    _checkpoints: dict[str, PipelineStateSnapshot] = field(default_factory=dict, init=False, repr=False)
    _lifecycle_hooks: list[PipelineStateLifecycleHook] = field(
        default_factory=list, init=False, repr=False
    )
    _profiler: PipelineStateProfiler = field(default_factory=PipelineStateProfiler, init=False, repr=False)

    _VALIDATORS: ClassVar[list[tuple[str | None, Callable[[PipelineState], None]]]] = []
    _STAGE_DEPENDENCIES: ClassVar[dict[str, tuple[str, ...]]] = {
        "parse": ("ingest",),
        "ir-validation": ("parse",),
        "pdf-download": ("ingest",),
        "pdf-gate": ("pdf-download",),
    }
    _SERIALISATION_CACHE: ClassVar[PipelineStateCache] = PipelineStateCache(ttl_seconds=120.0)

    def __post_init__(self) -> None:
        if PYDANTIC_AVAILABLE:
            try:
                PipelineStateEnvelope.model_validate(
                    {
                        "context": self.context,
                        "adapter_request": self.adapter_request,
                        "payload": self.payload,
                        "payloads": self.payloads,
                        "document": self.document,
                        "chunks": self.chunks,
                        "embedding_batch": self.embedding_batch,
                        "entities": self.entities,
                        "claims": self.claims,
                        "index_receipt": self.index_receipt,
                        "graph_receipt": self.graph_receipt,
                        "metadata": self.metadata,
                        "stage_results": self.stage_results,
                        "schema_version": self.schema_version,
                        "job_id": self.job_id,
                        "pdf_tracker": self.pdf_tracker,
                    }
                )
            except ValidationError as exc:  # pragma: no cover - defensive guard
                raise PipelineStateValidationError(str(exc)) from exc
        self._tenant_id = self.context.tenant_id
        self.register_lifecycle_hook(
            PipelineStateLifecycleHook(
                on_started=self._profiler.on_stage_started,
                on_completed=self._profiler.on_stage_completed,
                on_failed=self._profiler.on_stage_failed,
            )
        )

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

    @classmethod
    def required_stage_types(cls, stage_type: str) -> tuple[str, ...]:
        """Return the upstream stage types required before executing ``stage_type``."""
        return cls._STAGE_DEPENDENCIES.get(stage_type, ())

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def _mark_dirty(self) -> None:
        self._dirty = True
        self._serialised_cache = None
        _STATE_MUTATIONS.labels(operation="mutate").inc()
        self._cache = _StateCache()

    def is_dirty(self) -> bool:
        """Return whether the state has pending changes since last snapshot."""
        return self._dirty

    def clone(self) -> PipelineState:
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
        clone.pdf_tracker = copy.deepcopy(self.pdf_tracker)
        clone._dirty = self._dirty
        clone._serialised_cache = (
            copy.deepcopy(self._serialised_cache) if self._serialised_cache is not None else None
        )
        clone.pdf_gate = PdfGateState(
            downloaded=self.pdf_gate.downloaded,
            ir_ready=self.pdf_gate.ir_ready,
            metadata=copy.deepcopy(self.pdf_gate.metadata),
        )
        clone._dirty = self._dirty
        clone._cache = copy.deepcopy(self._cache)
        clone._checkpoints = {
            label: copy.deepcopy(snapshot) for label, snapshot in self._checkpoints.items()
        }
        return clone

    def register_lifecycle_hook(self, hook: PipelineStateLifecycleHook) -> None:
        """Register a lifecycle hook that observes stage progress."""
        self._lifecycle_hooks.append(hook)

    def notify_stage_started(self, stage: str, stage_type: str) -> None:
        for hook in self._lifecycle_hooks:
            if hook.on_started:
                hook.on_started(self, stage, stage_type)

    def notify_stage_completed(
        self,
        stage: str,
        stage_type: str,
        *,
        duration_ms: int,
        attempts: int,
        output_count: int,
    ) -> None:
        for hook in self._lifecycle_hooks:
            if hook.on_completed:
                hook.on_completed(
                    self,
                    stage,
                    stage_type,
                    duration_ms,
                    attempts,
                    output_count,
                )

    def notify_stage_failed(self, stage: str, stage_type: str, error: BaseException) -> None:
        for hook in self._lifecycle_hooks:
            if hook.on_failed:
                hook.on_failed(self, stage, stage_type, error)

    def profiling_summary(self) -> dict[str, Any]:
        """Return aggregated profiling metrics recorded for the pipeline run."""
        return self._profiler.summary()

    def profiling_samples(self) -> tuple[StagePerformanceSample, ...]:
        return tuple(self._profiler.samples)

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
    def has_pdf_assets(self) -> bool:
        return bool(self.pdf_assets)

    def set_pdf_assets(self, assets: Sequence[PdfAsset | Mapping[str, Any]]) -> None:
        converted: list[PdfAsset] = []
        for asset in assets:
            if isinstance(asset, PdfAsset):
                converted.append(asset)
            elif isinstance(asset, Mapping):
                model = PdfAssetModel.model_validate(asset)
                converted.append(
                    PdfAsset(
                        asset_id=model.asset_id,
                        uri=model.uri,
                        checksum=model.checksum,
                        metadata=model.metadata,
                    )
                )
            else:
                raise TypeError("PDF assets must be PdfAsset instances or mappings")
        self.pdf_assets = tuple(converted)
        self.metadata.setdefault("pdf", {})["assets"] = [
            attr_asdict(asset) for asset in converted
        ]
        self._mark_dirty()
        _state_logger.debug("pipeline_state.pdf_assets.set", count=len(converted))

    @property
    def is_pdf_ready(self) -> bool:
        return any(self.gate_status.values())

    def record_gate_status(self, stage_name: str, ready: bool) -> None:
        self.gate_status[stage_name] = ready
        gates = self.metadata.setdefault("gates", {})
        gates[stage_name] = {"ready": ready, "timestamp": time.time()}
        self._mark_dirty()
        _state_logger.debug("pipeline_state.gate.recorded", stage=stage_name, ready=ready)

    def ensure_ready_for(self, stage_type: str) -> None:
        """Validate preconditions required by the requested stage type."""
        if stage_type in {"parse", "ir-validation"} or stage_type == "download":
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
        elif stage_type == "pdf-gate":
            if not self.pdf_tracker.downloads:
                raise ValueError("PDF gate cannot run before any downloads are recorded")
        elif stage_type == "gate":
            self.require_downloads()
        elif stage_type == "download":
            self.require_document()
        elif stage_type == "gate":
            if not self.pdf_assets:
                raise ValueError("PipelineState requires PDF assets before gate stage")

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
            "pdf-download": "metadata",
            "pdf-ir-gate": "metadata",
            "download": "pdf_assets",
            "gate": "gate_status",
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
        elif stage_type == "pdf-download":
            assets = output or ()
            if isinstance(assets, Sequence) and not isinstance(assets, (str, bytes)):
                for asset in assets:
                    self.pdf_tracker.record_download(str(asset))
            else:
                self.pdf_tracker.record_download(str(assets))
            self._mark_dirty()
        elif stage_type == "pdf-gate":
            decision = getattr(output, "allowed", None)
            if decision is None:
                decision = bool(output)
            reason = getattr(output, "reason", None)
            self.pdf_tracker.mark_gate(opened=bool(decision), reason=reason)
            reference = getattr(output, "ledger_reference", None)
            if reference is not None:
                self.pdf_tracker.set_ledger_reference(str(reference))
            self._mark_dirty()
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
        elif stage_type == "pdf-download":
            metadata = output if isinstance(output, Mapping) else None
            self.mark_pdf_downloaded(metadata=metadata)
        elif stage_type == "pdf-ir-gate":
            metadata = output if isinstance(output, Mapping) else None
            self.mark_pdf_ir_ready(metadata=metadata)
        elif stage_type == "download":
            if not isinstance(output, Sequence):
                raise TypeError("Download stage must return a sequence of PDF assets")
            self.set_pdf_assets(output)
        elif stage_type == "gate":
            if not isinstance(output, bool):
                raise TypeError("Gate stage must return a boolean readiness flag")
            self.record_gate_status(stage_name, output)
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
        if stage_type == "pdf-download" and isinstance(output, Sequence):
            return len(tuple(output))
        if stage_type == "download" and isinstance(output, Sequence):
            return len(output)
        if stage_type == "gate" and isinstance(output, GateDecision):
            return int(output.ready)
        if stage_type == "gate" and isinstance(output, bool):
            return 1
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
        record_stage_metrics(
            pipeline=self.context.pipeline_name,
            stage=stage_name,
            stage_type=snapshot.stage_type,
            attempts=attempts,
            duration_ms=duration_ms,
            error=error,
        )
        self._mark_dirty()
        _state_logger.debug(
            "pipeline_state.stage_metrics_recorded",
            stage=stage_name,
            stage_type=snapshot.stage_type,
            attempts=attempts,
            duration_ms=duration_ms,
            output_count=output_count,
            error=error,
        )

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

    def dependencies_satisfied(self, dependencies: Sequence[str]) -> bool:
        """Return True when all dependency stages have completed successfully."""
        for dependency in dependencies:
            snapshot = self.stage_results.get(dependency)
            if snapshot is None or snapshot.error:
                return False
        return True

    def ensure_dependencies(self, stage_name: str, dependencies: Sequence[str]) -> None:
        """Raise if any dependency is missing or failed."""
        unmet: list[str] = []
        for dependency in dependencies:
            snapshot = self.stage_results.get(dependency)
            if snapshot is None:
                unmet.append(dependency)
            elif snapshot.error:
                unmet.append(f"{dependency} (failed)")
        if unmet:
            dependency_list = ", ".join(sorted(unmet))
            raise ValueError(
                f"Stage '{stage_name}' cannot execute until dependencies are satisfied: {dependency_list}"
            )

    def mark_pdf_downloaded(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        """Flag the PDF gate as downloaded and merge metadata."""
        self.pdf_gate.downloaded = True
        self.pdf_gate.merge_metadata(metadata)
        self.metadata.setdefault("pdf", {})["downloaded"] = True
        self._mark_dirty()

    def mark_pdf_ir_ready(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        """Flag the PDF gate as ready for IR stage and merge metadata."""
        self.pdf_gate.ir_ready = True
        self.pdf_gate.merge_metadata(metadata)
        self.metadata.setdefault("pdf", {})["ir_ready"] = True
        self._mark_dirty()

    def reset_pdf_gate(self) -> None:
        """Reset the PDF gate state."""
        self.pdf_gate = PdfGateState()
        if "pdf" in self.metadata:
            self.metadata.pop("pdf", None)
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
            pdf_assets=tuple(copy.deepcopy(self.pdf_assets)),
            gate_status=dict(self.gate_status),
            metadata=copy.deepcopy(self.metadata),
            stage_results=stage_payload,
            job_id=self.job_id,
            pdf_tracker=copy.deepcopy(self.pdf_tracker),
            pdf_gate=PdfGateState(
                downloaded=self.pdf_gate.downloaded,
                ir_ready=self.pdf_gate.ir_ready,
                metadata=copy.deepcopy(self.pdf_gate.metadata),
            ),
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
        self.pdf_assets = tuple(copy.deepcopy(snapshot.pdf_assets))
        self.gate_status = dict(snapshot.gate_status)
        self.metadata = copy.deepcopy(snapshot.metadata)
        if restore_stage_results:
            self.stage_results = {
                name: copy.deepcopy(result) for name, result in snapshot.stage_results.items()
            }
        self.job_id = snapshot.job_id
        self.pdf_tracker = copy.deepcopy(snapshot.pdf_tracker)
        self.pdf_gate = PdfGateState(
            downloaded=snapshot.pdf_gate.downloaded,
            ir_ready=snapshot.pdf_gate.ir_ready,
            metadata=copy.deepcopy(snapshot.pdf_gate.metadata),
        )
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
            _STATE_SERIALISATION_LATENCY.labels(format="dict").observe(0.0)
            return copy.deepcopy(self._serialised_cache)
        if use_cache and not self._dirty and self._cache.payload is not None:
            return copy.deepcopy(self._cache.payload)

        start = time.perf_counter()
        snapshot: dict[str, Any] = {
            "version": self.schema_version,
            "job_id": self.job_id,
            "context": self.context.to_dict(),
            "adapter_request": self.adapter_request.model_dump(mode="json"),
            "payload": dict(self.payload),
            "payload_count": len(self.payloads),
            "document_id": getattr(self.document, "id", None),
            "chunk_count": len(self.chunks),
            "embedding_count": len(self.embedding_batch.vectors)
            if self.embedding_batch
            else 0,
            "entity_count": len(self.entities),
            "claim_count": len(self.claims),
            "metadata": copy.deepcopy(self.metadata),
            "index_receipt": self.index_receipt.metadata if self.index_receipt else None,
            "graph_receipt": self.graph_receipt.metadata if self.graph_receipt else None,
            "pdf": self.pdf_tracker.as_dict(),
            "download_count": len(self.downloads),
            "gate_status": {name: decision.ready for name, decision in self.gate_decisions.items()},
            "gate_status": dict(self.gate_status),
        }
        if self.pdf_assets:
            snapshot["pdf_assets"] = [
                {
                    "asset_id": asset.asset_id,
                    "uri": asset.uri,
                    "checksum": asset.checksum,
                }
                for asset in self.pdf_assets
            ]
            snapshot["pdf_asset_count"] = len(self.pdf_assets)
        if include_stage_results:
            snapshot["stage_results"] = {
                name: result.as_dict() for name, result in self.stage_results.items()
            }
        elapsed = time.perf_counter() - start
        _STATE_SERIALISATION_LATENCY.labels(format="dict").observe(elapsed)
        if self.pdf_gate:
            snapshot["pdf_gate"] = self.pdf_gate.as_dict()

        model = serialise_payload(snapshot)
        payload = model.model_dump(mode="json")

        if use_cache:
            self._cache.payload = copy.deepcopy(snapshot)
            self._serialised_cache = copy.deepcopy(payload)
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

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return a dictionary compatible with legacy dict-based state consumers."""
        payload: dict[str, Any] = {
            "version": self.schema_version,
            "job_id": self.job_id,
            "context": self.context.to_dict(),
            "adapter_request": self.adapter_request.model_dump(mode="json"),
            "payload": copy.deepcopy(self.payload),
            "payloads": [copy.deepcopy(item) for item in self.payloads],
            "metadata": copy.deepcopy(self.metadata),
            "stage_results": {
                name: result.as_dict() for name, result in self.stage_results.items()
            },
        }
        if self.document is not None:
            payload["document"] = self.document.model_dump(mode="json")
        if self.chunks:
            payload["chunks"] = [chunk.model_dump(mode="json") for chunk in self.chunks]
        if self.embedding_batch is not None:
            payload["embedding_batch"] = {
                "vectors": [
                    {
                        "id": vector.id,
                        "values": list(vector.values),
                        "metadata": dict(vector.metadata),
                    }
                    for vector in self.embedding_batch.vectors
                ],
                "model": self.embedding_batch.model,
                "tenant_id": self.embedding_batch.tenant_id,
            }
        if self.entities:
            payload["entities"] = [entity.model_dump(mode="json") for entity in self.entities]
        if self.claims:
            payload["claims"] = [claim.model_dump(mode="json") for claim in self.claims]
        if self.index_receipt is not None:
            payload["index_receipt"] = asdict(self.index_receipt)
        if self.graph_receipt is not None:
            payload["graph_receipt"] = asdict(self.graph_receipt)
        payload["pdf"] = self.pdf_tracker.as_dict()
        payload["pdf_gate"] = self.pdf_gate.as_dict()
        if self.pdf_assets:
            payload["pdf_assets"] = [attr_asdict(asset) for asset in self.pdf_assets]
        if self.gate_status:
            payload["gate_status"] = dict(self.gate_status)
        return payload

    def _cache_key(self) -> str:
        return (
            self.job_id
            or self.context.job_id
            or self.context.correlation_id
            or f"state-{id(self)}"
        )

        start = time.perf_counter()
        payload = self.serialise()
        encoded = orjson.dumps(payload).decode("utf-8")
        elapsed = time.perf_counter() - start
        _STATE_SERIALISATION_LATENCY.labels(format="json").observe(elapsed)
        return encoded
        return orjson.dumps(self.serialise()).decode("utf-8")
    def to_model(self) -> PipelineStateModel:
        """Return the Pydantic representation for the current state."""
        if not self._dirty and self._cache.json_bytes is not None:
            return self._cache.json_bytes.decode("utf-8")
        payload = self.serialise()
        start = time.perf_counter()
        json_bytes = orjson.dumps(payload)
        duration = time.perf_counter() - start
        _STATE_SERIALISE_COUNTER.labels(format="json").inc()
        _STATE_SERIALISE_LATENCY.labels(format="json").observe(duration)
        self._cache.json_bytes = json_bytes
        return json_bytes.decode("utf-8")

        payload = self.serialise()
        return PipelineStateModel.model_validate(payload)

    def serialise_json(self, *, use_cache: bool = True) -> str:
        """Return a JSON encoded snapshot of the state."""
        payload = self.serialise(use_cache=use_cache)
        if use_cache and not self._dirty:
            cached = self._SERIALISATION_CACHE.get(self._cache_key())
            if cached is not None:
                return cached.decode("utf-8")
            blob = dumps_orjson(payload)
            self._SERIALISATION_CACHE.set(self._cache_key(), blob)
            return blob.decode("utf-8")
        return dumps_json(payload)

    def serialise_compressed(self, *, use_cache: bool = True) -> bytes:
        """Compress the JSON snapshot for efficient transport."""
        if not self._dirty and self._cache.compressed is not None:
            return self._cache.compressed
        json_bytes = self._cache.json_bytes
        if json_bytes is None or self._dirty:
            json_bytes = orjson.dumps(self.serialise())
            self._cache.json_bytes = json_bytes
        start = time.perf_counter()
        compressed = zlib.compress(json_bytes)
        duration = time.perf_counter() - start
        _STATE_SERIALISE_COUNTER.labels(format="compressed").inc()
        _STATE_SERIALISE_LATENCY.labels(format="compressed").observe(duration)
        self._cache.compressed = compressed
        start = time.perf_counter()
        encoded = self.serialise_json().encode("utf-8")
        compressed = zlib.compress(encoded)
        elapsed = time.perf_counter() - start
        _STATE_SERIALISATION_LATENCY.labels(format="compressed").observe(elapsed)
        return compressed

    def serialise_base64(self) -> str:
        """Return a base64 encoded compressed snapshot."""
        start = time.perf_counter()
        compressed = self.serialise_compressed()
        encoded = base64.b64encode(compressed).decode("ascii")
        elapsed = time.perf_counter() - start
        _STATE_SERIALISATION_LATENCY.labels(format="base64").observe(elapsed)
        return encoded

    def persist_with_retry(self, persist: Callable[[dict[str, Any]], None]) -> None:
        """Persist the current snapshot using tenacity-backed retries."""
        snapshot = self.serialise()

        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.1, min=0.1, max=1.0),
            retry=retry_if_exception_type(Exception),
        )
        def _attempt(payload: dict[str, Any]) -> None:
            persist(payload)

        _attempt(snapshot)
        _STATE_MUTATIONS.labels(operation="persist").inc()
        _state_logger.debug(
            "pipeline_state.persisted",
            tenant_id=self._tenant_id,
            job_id=self.job_id,
        )
        if not self._dirty and self._cache.compressed is not None:
            return self._cache.compressed
        json_bytes = self._cache.json_bytes
        if json_bytes is None or self._dirty:
            json_bytes = orjson.dumps(self.serialise())
            self._cache.json_bytes = json_bytes
        start = time.perf_counter()
        compressed = zlib.compress(json_bytes)
        duration = time.perf_counter() - start
        _STATE_SERIALISE_COUNTER.labels(format="compressed").inc()
        _STATE_SERIALISE_LATENCY.labels(format="compressed").observe(duration)
        self._cache.compressed = compressed
        return compressed

    def serialise_base64(self, *, use_cache: bool = True) -> str:
        """Return a base64 encoded compressed snapshot."""
        payload = self.serialise(use_cache=use_cache)
        blob = dumps_orjson(payload)
        if use_cache and not self._dirty:
            self._SERIALISATION_CACHE.set(self._cache_key(), blob)
        return encode_base64(blob)
        if not self._dirty and self._cache.base64_payload is not None:
            return self._cache.base64_payload
        compressed = self.serialise_compressed()
        encoded = base64.b64encode(compressed).decode("ascii")
        self._cache.base64_payload = encoded
        return encoded

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.2, max=2.0))
    def persist_with_retry(
        self,
        writer: Callable[[bytes], Any],
        *,
        format: str = "json",
    ) -> Any:
        """Persist the state payload using the supplied writer with retries."""
        if format == "json":
            payload = self.serialise_json().encode("utf-8")
        elif format == "compressed":
            payload = self.serialise_compressed()
        else:
            raise ValueError(f"Unsupported persistence format '{format}'")
        try:
            result = writer(payload)
        except Exception as exc:
            _STATE_PERSIST_COUNTER.labels(format=format, status="error").inc()
            _state_logger.error(
                "pipeline_state.persist.failure",
                format=format,
                error=str(exc),
            )
            raise
        _STATE_PERSIST_COUNTER.labels(format=format, status="success").inc()
        _state_logger.debug(
            "pipeline_state.persist.success",
            format=format,
            size=len(payload),
        )
        return result

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
        if len(self.pdf_assets) != len(other.pdf_assets):
            entries["pdf_asset_count"] = (len(self.pdf_assets), len(other.pdf_assets))
        if self.gate_status != other.gate_status:
            entries["gate_status"] = (dict(self.gate_status), dict(other.gate_status))
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
            decoded = zlib.decompress(bytes(payload)).decode("utf-8")
            decoded = zlib.decompress(bytes(payload))
            recovered = orjson.loads(decoded)
        elif isinstance(payload, str):
            try:
                compressed = base64.b64decode(payload)
            except (ValueError, TypeError):
                recovered = orjson.loads(payload)
            else:
                decoded = zlib.decompress(compressed).decode("utf-8")
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
        pdf_payload = recovered.get("pdf")
        if isinstance(pdf_payload, Mapping):
            state.pdf_tracker.downloads = tuple(str(item) for item in pdf_payload.get("downloads", ()))
            state.pdf_tracker.last_attempt_ms = pdf_payload.get("last_attempt_ms")
            state.pdf_tracker.gate_open = bool(pdf_payload.get("gate_open", False))
            state.pdf_tracker.gate_reason = pdf_payload.get("gate_reason")
            state.pdf_tracker.ledger_reference = pdf_payload.get("ledger_reference")
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
        pdf_payload = recovered.get("pdf_assets")
        if isinstance(pdf_payload, Sequence):
            state.set_pdf_assets(pdf_payload)
        gate_payload = recovered.get("gate_status")
        if isinstance(gate_payload, Mapping):
            for stage_name, ready in gate_payload.items():
                if isinstance(ready, bool):
                    state.gate_status[str(stage_name)] = ready
            state.metadata.setdefault("gates", {})
            for name, ready in state.gate_status.items():
                state.metadata["gates"][name] = {"ready": ready, "timestamp": time.time()}
        state._dirty = False
        recovered_dict = dict(recovered)
        state._cache.payload = copy.deepcopy(recovered_dict)
        state._cache.json_bytes = orjson.dumps(recovered_dict)
        state._cache.compressed = zlib.compress(state._cache.json_bytes)
        state._cache.base64_payload = base64.b64encode(state._cache.compressed).decode("ascii")
        return state

    def hydrate_legacy(self, payload: Mapping[str, Any]) -> None:
        """Populate the state using a legacy dictionary payload."""
        self.payload = dict(payload.get("payload", {}))
        raw_payloads = payload.get("payloads")
        if isinstance(raw_payloads, Sequence):
            self.payloads = tuple(copy.deepcopy(list(raw_payloads)))
        document_payload = payload.get("document")
        if document_payload:
            self.document = Document.model_validate(document_payload)
        else:
            self.document = None
        chunk_payload = payload.get("chunks")
        if isinstance(chunk_payload, Sequence):
            self.chunks = tuple(Chunk.model_validate(item) for item in chunk_payload)
        else:
            self.chunks = ()
        embedding_payload = payload.get("embedding_batch")
        if isinstance(embedding_payload, Mapping):
            vectors = []
            for vector in embedding_payload.get("vectors", []):
                if isinstance(vector, Mapping):
                    values = tuple(float(v) for v in vector.get("values", ()))
                    vectors.append(
                        EmbeddingVector(
                            id=str(vector.get("id")),
                            values=values,
                            metadata=dict(vector.get("metadata", {})),
                        )
                    )
            self.embedding_batch = EmbeddingBatch(
                vectors=tuple(vectors),
                model=str(embedding_payload.get("model", "")),
                tenant_id=str(embedding_payload.get("tenant_id", self.context.tenant_id)),
            )
        else:
            self.embedding_batch = None
        entities_payload = payload.get("entities")
        if isinstance(entities_payload, Sequence):
            self.entities = tuple(Entity.model_validate(item) for item in entities_payload)
        else:
            self.entities = ()
        claims_payload = payload.get("claims")
        if isinstance(claims_payload, Sequence):
            self.claims = tuple(Claim.model_validate(item) for item in claims_payload)
        else:
            self.claims = ()
        index_payload = payload.get("index_receipt")
        if isinstance(index_payload, Mapping):
            self.index_receipt = IndexReceipt(
                chunks_indexed=int(index_payload.get("chunks_indexed", 0)),
                opensearch_ok=bool(index_payload.get("opensearch_ok", False)),
                faiss_ok=bool(index_payload.get("faiss_ok", False)),
                metadata=dict(index_payload.get("metadata", {})),
            )
        else:
            self.index_receipt = None
        graph_payload = payload.get("graph_receipt")
        if isinstance(graph_payload, Mapping):
            self.graph_receipt = GraphWriteReceipt(
                nodes_written=int(graph_payload.get("nodes_written", 0)),
                edges_written=int(graph_payload.get("edges_written", 0)),
                correlation_id=str(graph_payload.get("correlation_id", "")),
                metadata=dict(graph_payload.get("metadata", {})),
            )
        else:
            self.graph_receipt = None
        self.metadata = copy.deepcopy(payload.get("metadata", {}))
        pdf_payload = payload.get("pdf_gate") or payload.get("pdf")
        if isinstance(pdf_payload, Mapping):
            self.pdf_gate = PdfGateState(
                downloaded=bool(pdf_payload.get("downloaded", False)),
                ir_ready=bool(pdf_payload.get("ir_ready", False)),
                metadata=dict(pdf_payload.get("metadata", {})),
            )
        else:
            self.pdf_gate = PdfGateState()
        pdf_payload = payload.get("pdf_assets")
        if isinstance(pdf_payload, Sequence):
            self.set_pdf_assets(pdf_payload)
        else:
            self.pdf_assets = ()
        gate_payload = payload.get("gate_status")
        if isinstance(gate_payload, Mapping):
            self.gate_status = {str(name): bool(value) for name, value in gate_payload.items()}
        else:
            self.gate_status = {}
        self.metadata.setdefault("gates", {})
        for name, ready in self.gate_status.items():
            self.metadata["gates"][name] = {"ready": ready, "timestamp": time.time()}
        stage_payload = payload.get("stage_results")
        if isinstance(stage_payload, Mapping):
            self.stage_results = {}
            for name, payload_data in stage_payload.items():
                if isinstance(payload_data, Mapping):
                    self.stage_results[str(name)] = StageResultSnapshot(
                        stage=str(payload_data.get("stage", name)),
                        stage_type=str(payload_data.get("stage_type", "unknown")),
                        attempts=payload_data.get("attempts"),
                        duration_ms=payload_data.get("duration_ms"),
                        output_count=payload_data.get("output_count"),
                        error=payload_data.get("error"),
                    )
        self.job_id = payload.get("job_id") or self.context.job_id
        self.schema_version = str(payload.get("version", self.schema_version))
        pdf_payload = payload.get("pdf")
        if isinstance(pdf_payload, Mapping):
            self.pdf_tracker.downloads = tuple(str(item) for item in pdf_payload.get("downloads", ()))
            self.pdf_tracker.last_attempt_ms = pdf_payload.get("last_attempt_ms")
            self.pdf_tracker.gate_open = bool(pdf_payload.get("gate_open", False))
            self.pdf_tracker.gate_reason = pdf_payload.get("gate_reason")
            self.pdf_tracker.ledger_reference = pdf_payload.get("ledger_reference")
        self._dirty = True
        self._cache = _StateCache()
        self.clear_checkpoints()

    @classmethod
    def from_legacy(
        cls,
        payload: Mapping[str, Any],
        *,
        context: StageContext,
        adapter_request: AdapterRequest,
    ) -> PipelineState:
        """Rehydrate a typed state from a legacy dictionary payload."""
        state = cls.initialise(
            context=context,
            adapter_request=adapter_request,
            payload=payload.get("payload"),
        )
        state.hydrate_legacy(payload)
        return state

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @classmethod
    def register_validator(
        cls,
        validator: Callable[[PipelineState], None],
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
        extra_rules: Sequence[Callable[[PipelineState], None]] | None = None,
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
        for dependency, satisfied in self._dependency_status(stage_type).items():
            if not satisfied:
                _STATE_DEPENDENCY_FAILURES.labels(stage_type=stage_type).inc()
                raise PipelineStateValidationError(
                    f"Stage '{stage_type}' requires '{dependency}' to complete before execution"
                )
        try:
            self.ensure_ready_for(stage_type)
        except ValueError as exc:
            raise PipelineStateValidationError(
                f"State missing prerequisites for stage '{stage_type}': {exc}"
            ) from exc

    def _dependency_status(self, stage_type: str) -> dict[str, bool]:
        requirements = self._DEPENDENCIES.get(stage_type, ())
        if not requirements:
            return {}
        statuses: dict[str, bool] = {}
        known_stage_types = {result.stage_type for result in self.stage_results.values()}
        for requirement in requirements:
            if requirement == "ingest":
                statuses[requirement] = bool(self.payloads)
            elif requirement == "parse":
                statuses[requirement] = self.document is not None
            elif requirement == "chunk":
                statuses[requirement] = bool(self.chunks)
            elif requirement == "embed":
                statuses[requirement] = self.embedding_batch is not None
            elif requirement == "extract":
                statuses[requirement] = bool(self.entities or self.claims)
            elif requirement == "pdf-download":
                statuses[requirement] = bool(self.pdf_tracker.downloads)
            else:
                statuses[requirement] = requirement in known_stage_types
        return statuses


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
    "DownloadArtifact",
    "DownloadStage",
    "EmbedStage",
    "EmbeddingBatch",
    "EmbeddingVector",
    "ExtractStage",
    "GateDecision",
    "GateStage",
    "GraphWriteReceipt",
    "IndexReceipt",
    "IndexStage",
    "IngestStage",
    "KGStage",
    "ParseStage",
    "PipelineGateNotReady",
    "PipelineState",
    "PipelineStateSnapshot",
    "PipelineStateValidationError",
    "RawPayload",
    "StageContext",
    "StageResultSnapshot",
]
