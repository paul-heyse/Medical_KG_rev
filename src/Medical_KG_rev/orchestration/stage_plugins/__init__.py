"""Built-in plugin registrations for pluggable orchestration stages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.stage_registry import (
    StageMetadata,
    StageRegistration,
)
from Medical_KG_rev.orchestration.stages.contracts import StageContext

logger = structlog.get_logger(__name__)


class GateConditionError(RuntimeError):
    """Raised when a gate stage condition fails."""


def _sequence_length(value: Any) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def _handle_download_output(state: dict[str, Any], _: str, output: Any) -> None:
    state["downloaded_files"] = output


def _handle_gate_output(state: dict[str, Any], _: str, output: Any) -> None:  # pragma: no cover - no-op
    return None


@dataclass(slots=True)
class DownloadStage:
    """Example download stage that records configured sources."""

    name: str
    sources: list[dict[str, Any]]

    def execute(self, ctx: StageContext, upstream: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, source in enumerate(self.sources):
            record = {
                "id": f"{self.name}:{index}",
                "tenant_id": ctx.tenant_id,
                "source": dict(source),
                "status": "skipped",
            }
            results.append(record)
        if not results and upstream:
            results.append(
                {
                    "id": f"{self.name}:0",
                    "tenant_id": ctx.tenant_id,
                    "source": {"upstream": upstream},
                    "status": "forwarded",
                }
            )
        logger.debug(
            "dagster.stage.download.completed",
            stage=self.name,
            tenant_id=ctx.tenant_id,
            files=len(results),
        )
        return results


@dataclass(slots=True)
class GateCondition:
    key: str
    expected: Any = True


@dataclass(slots=True)
class GateStage:
    """Gate stage validating state conditions before proceeding."""

    name: str
    conditions: tuple[GateCondition, ...]

    def execute(self, ctx: StageContext, upstream: Any) -> None:
        state = upstream if isinstance(upstream, dict) else {"value": upstream}
        for condition in self.conditions:
            value = state
            for part in condition.key.split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = getattr(value, part, None)
            if value != condition.expected:
                logger.warning(
                    "dagster.stage.gate.blocked",
                    stage=self.name,
                    tenant_id=ctx.tenant_id,
                    key=condition.key,
                    expected=condition.expected,
                    actual=value,
                )
                raise GateConditionError(
                    f"Gate '{self.name}' blocked: expected {condition.key} == {condition.expected!r}"
                )
        logger.debug(
            "dagster.stage.gate.passed",
            stage=self.name,
            tenant_id=ctx.tenant_id,
            conditions=len(self.conditions),
        )


def register_download_stage() -> StageRegistration:
    """Register the built-in download stage plugin."""

    def _builder(definition: StageDefinition) -> DownloadStage:
        config = definition.config or {}
        sources = config.get("sources") or config.get("urls") or []
        normalised: list[dict[str, Any]] = []
        if isinstance(sources, dict):
            normalised.append(dict(sources))
        elif isinstance(sources, Iterable) and not isinstance(sources, (str, bytes)):
            for item in sources:
                if isinstance(item, dict):
                    normalised.append(dict(item))
                else:
                    normalised.append({"value": item})
        return DownloadStage(name=definition.name, sources=normalised)

    metadata = StageMetadata(
        stage_type="download",
        state_key="downloaded_files",
        output_handler=_handle_download_output,
        output_counter=_sequence_length,
        description="Downloads external resources referenced by upstream payloads",
        dependencies=("ingest",),
    )
    return StageRegistration(metadata=metadata, builder=_builder)


def register_gate_stage() -> StageRegistration:
    """Register the built-in gate stage plugin."""

    def _builder(definition: StageDefinition) -> GateStage:
        config = definition.config or {}
        conditions_config = config.get("conditions") or []
        parsed: list[GateCondition] = []
        for entry in conditions_config:
            if isinstance(entry, Mapping):
                key = entry.get("key") or "value"
                parsed.append(GateCondition(key=str(key), expected=entry.get("expected", True)))
            elif isinstance(entry, str):
                parsed.append(GateCondition(key=entry, expected=True))
        if not parsed:
            parsed.append(GateCondition(key="value", expected=True))
        return GateStage(name=definition.name, conditions=tuple(parsed))

    metadata = StageMetadata(
        stage_type="gate",
        state_key=None,
        output_handler=_handle_gate_output,
        output_counter=lambda _: 0,
        description="Halts pipeline execution until configured conditions are met",
        dependencies=("download",),
    )
    return StageRegistration(metadata=metadata, builder=_builder)


__all__ = [
    "DownloadStage",
    "GateCondition",
    "GateConditionError",
    "GateStage",
    "register_download_stage",
    "register_gate_stage",
]
