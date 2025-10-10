"""Lightweight OpenLineage emitter used by the Dagster orchestrator."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, ClassVar

from Medical_KG_rev.orchestration.ledger import JobLedgerEntry
from Medical_KG_rev.utils.logging import get_logger

try:  # pragma: no cover - optional dependency
    from openlineage.client import OpenLineageClient  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenLineageClient = None  # type: ignore


logger = get_logger(__name__)


class RunState(str, Enum):
    """Minimal subset of OpenLineage run states used for job events."""

    START = "START"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"


@dataclass(slots=True)
class BaseFacet:
    """Base helper providing OpenLineage facet conversion."""

    facet_name: ClassVar[str]
    schema_url: ClassVar[str]

    def as_dict(self, producer: str) -> dict[str, Any]:
        payload = self._payload()
        return {
            "_producer": producer,
            "_schemaURL": self.schema_url,
            **payload,
        }

    def _payload(self) -> Mapping[str, Any]:  # pragma: no cover - override hook
        raise NotImplementedError


@dataclass(slots=True)
class GPUUtilizationFacet(BaseFacet):
    """Facet describing GPU utilisation for a job run."""

    facet_name: ClassVar[str] = "gpuUtilization"
    schema_url: ClassVar[str] = "https://openlineage.io/spec/facets/1-0-0/GpuUtilizationFacet.json"

    gpu_memory_used_mb: float | None = None
    gpu_utilization_percent: float | None = None

    def _payload(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {}
        if self.gpu_memory_used_mb is not None:
            payload["gpuMemoryUsedMb"] = self.gpu_memory_used_mb
        if self.gpu_utilization_percent is not None:
            payload["gpuUtilizationPercent"] = self.gpu_utilization_percent
        return payload


@dataclass(slots=True)
class ModelVersionFacet(BaseFacet):
    """Facet linking a run to the model version responsible for outputs."""

    facet_name: ClassVar[str] = "modelVersion"
    schema_url: ClassVar[str] = "https://openlineage.io/spec/facets/1-0-0/ModelVersionFacet.json"

    model_name: str
    model_version: str | None = None

    def _payload(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {"modelName": self.model_name}
        if self.model_version:
            payload["modelVersion"] = self.model_version
        return payload


@dataclass(slots=True)
class RetryAttemptFacet(BaseFacet):
    """Facet capturing retry counts per stage for a run."""

    facet_name: ClassVar[str] = "retryAttempts"
    schema_url: ClassVar[str] = "https://openlineage.io/spec/facets/1-0-0/RetryAttemptFacet.json"

    attempts: Mapping[str, int]

    def _payload(self) -> Mapping[str, Any]:
        return {"attempts": dict(self.attempts)}


@dataclass(slots=True)
class OpenLineageEvent:
    """Simple representation of an OpenLineage RunEvent payload."""

    event_type: RunState
    event_time: str
    job_name: str
    namespace: str
    run_id: str
    producer: str
    run_facets: Mapping[str, Any]
    job_facets: Mapping[str, Any]
    inputs: Sequence[Mapping[str, Any]]
    outputs: Sequence[Mapping[str, Any]]
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "eventType": self.event_type.value,
            "eventTime": self.event_time,
            "job": {"namespace": self.namespace, "name": self.job_name},
            "run": {"runId": self.run_id, "facets": dict(self.run_facets)},
            "producer": self.producer,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "jobFacets": dict(self.job_facets),
        }
        if self.message:
            payload["message"] = self.message
        return payload


class OpenLineageEmitter:
    """Emit OpenLineage-compatible run events with optional client delivery."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        enabled: bool | None = None,
        namespace: str = "medical-kg",
        producer: str = "medical-kg.orchestration",
    ) -> None:
        self.namespace = namespace
        self.producer = producer
        self._events: list[dict[str, Any]] = []
        self.enabled = bool(enabled) if enabled is not None else _env_enabled()
        if client is not None:
            self._client = client
        elif OpenLineageClient is not None and self.enabled:
            try:  # pragma: no cover - guarded by optional dependency
                self._client = OpenLineageClient()  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning("openlineage.client.init_failed", error=str(exc))
                self._client = None
                self.enabled = False
        else:
            self._client = None

    @property
    def events(self) -> Sequence[Mapping[str, Any]]:
        """Expose emitted events for testing and diagnostics."""
        return tuple(self._events)

    def clear(self) -> None:
        self._events.clear()

    # ------------------------------------------------------------------
    # Emission helpers
    # ------------------------------------------------------------------
    def emit_run_started(
        self,
        pipeline: str,
        *,
        run_id: str,
        context: Any,
        attempt: int,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._emit(
            RunState.START,
            pipeline,
            run_id=run_id,
            attempt=attempt,
            context=context,
            ledger_entry=None,
            run_metadata=run_metadata,
        )

    def emit_run_completed(
        self,
        pipeline: str,
        *,
        run_id: str,
        context: Any,
        attempt: int,
        ledger_entry: JobLedgerEntry | None,
        run_metadata: Mapping[str, Any] | None,
        duration_ms: int,
    ) -> Mapping[str, Any]:
        extra_metadata: dict[str, Any] = {"duration_ms": duration_ms}
        return self._emit(
            RunState.COMPLETE,
            pipeline,
            run_id=run_id,
            attempt=attempt,
            context=context,
            ledger_entry=ledger_entry,
            run_metadata=run_metadata,
            extra_metadata=extra_metadata,
        )

    def emit_run_failed(
        self,
        pipeline: str,
        *,
        run_id: str,
        context: Any,
        attempt: int,
        ledger_entry: JobLedgerEntry | None,
        run_metadata: Mapping[str, Any] | None,
        error: str,
    ) -> Mapping[str, Any]:
        return self._emit(
            RunState.FAIL,
            pipeline,
            run_id=run_id,
            attempt=attempt,
            context=context,
            ledger_entry=ledger_entry,
            run_metadata=run_metadata,
            error=error,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _emit(
        self,
        state: RunState,
        pipeline: str,
        *,
        run_id: str,
        attempt: int,
        context: Any,
        ledger_entry: JobLedgerEntry | None,
        run_metadata: Mapping[str, Any] | None,
        extra_metadata: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> Mapping[str, Any]:
        event_time = datetime.now(UTC).isoformat()
        run_facets, job_facets = self._build_facets(
            ledger_entry=ledger_entry,
            run_metadata=run_metadata,
            attempt=attempt,
            extra_metadata=extra_metadata,
        )
        event = OpenLineageEvent(
            state,
            event_time,
            pipeline,
            self.namespace,
            run_id,
            self.producer,
            run_facets,
            job_facets,
            inputs=[],
            outputs=[],
            message=error,
        )
        payload = event.to_dict()
        self._send(payload)
        return payload

    def _send(self, payload: Mapping[str, Any]) -> None:
        if self.enabled and self._client is not None:
            try:  # pragma: no cover - external dependency path
                self._client.emit(payload)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("openlineage.client.emit_failed", error=str(exc))
        self._events.append(dict(payload))

    def _build_facets(
        self,
        *,
        ledger_entry: JobLedgerEntry | None,
        run_metadata: Mapping[str, Any] | None,
        attempt: int,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[MutableMapping[str, Any], MutableMapping[str, Any]]:
        run_facets: MutableMapping[str, Any] = {}
        job_facets: MutableMapping[str, Any] = {}

        if ledger_entry and ledger_entry.retry_count_per_stage:
            retry_facet = RetryAttemptFacet(ledger_entry.retry_count_per_stage)
            run_facets[retry_facet.facet_name] = retry_facet.as_dict(self.producer)

        metadata = dict(run_metadata or {})
        metadata.setdefault("attempt", attempt)
        if extra_metadata:
            metadata.update(extra_metadata)

        metrics = metadata.get("metrics")
        if isinstance(metrics, Mapping):
            gpu_facet = GPUUtilizationFacet(
                gpu_memory_used_mb=_coerce_float(metrics.get("gpu_memory_mb")),
                gpu_utilization_percent=_coerce_float(metrics.get("gpu_utilization_percent")),
            )
            if gpu_facet._payload():
                job_facets[gpu_facet.facet_name] = gpu_facet.as_dict(self.producer)

        models = metadata.get("models")
        if isinstance(models, Mapping):
            for _, details in models.items():
                if not isinstance(details, Mapping):
                    continue
                model_name = str(details.get("name", "unknown"))
                model_version = details.get("version")
                facet = ModelVersionFacet(model_name=model_name, model_version=model_version)
                job_facets[facet.facet_name] = facet.as_dict(self.producer)
                break

        return run_facets, job_facets


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_enabled() -> bool:
    value = os.getenv("MK_ENABLE_OPENLINEAGE")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "GPUUtilizationFacet",
    "ModelVersionFacet",
    "OpenLineageEmitter",
    "RetryAttemptFacet",
    "RunState",
]
