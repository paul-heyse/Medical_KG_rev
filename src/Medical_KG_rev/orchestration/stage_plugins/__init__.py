"""Built-in plugin registrations for pluggable orchestration stages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import structlog

from Medical_KG_rev.orchestration.dagster.configuration import (
    GateDefinition,
    StageDefinition,
)
from Medical_KG_rev.orchestration.dagster.gates import (
    GateConditionError,
    GateConditionEvaluator,
    GateEvaluationResult,
    GateTimeoutError,
    build_gate_result,
)
from Medical_KG_rev.orchestration.dagster.stage_registry import (
    StageMetadata,
    StageRegistration,
)
from Medical_KG_rev.orchestration.stages.contracts import StageContext

if TYPE_CHECKING:  # pragma: no cover - hints only
    from Medical_KG_rev.orchestration.ledger import JobLedger

logger = structlog.get_logger(__name__)


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
class GateStage:
    """Gate stage validating ledger state before proceeding."""

    name: str
    definition: GateDefinition
    evaluator: GateConditionEvaluator
    timeout_seconds: int | None
    poll_interval: float
    max_attempts: int | None
    retry_backoff: float

    def evaluate(
        self,
        ctx: StageContext,
        ledger: "JobLedger",
        state: dict[str, Any],
    ) -> GateEvaluationResult:
        job_id = ctx.job_id or state.get("job_id")
        if not job_id:
            raise GateConditionError(f"Gate '{self.name}' cannot evaluate without a job identifier")

        gate_state = state.setdefault("gates", {}).setdefault(self.definition.name, {})
        attempts = 0
        start_time = time.perf_counter()
        deadline = (
            start_time + self.timeout_seconds if self.timeout_seconds is not None else None
        )

        while True:
            attempts += 1
            entry = ledger.get(job_id)
            if entry is None:
                raise GateConditionError(
                    f"Gate '{self.name}' could not locate job '{job_id}' in the ledger"
                )

            satisfied, details = self.evaluator.evaluate(entry, state, gate_state)
            gate_state.update(
                {
                    "last_details": details,
                    "attempts": attempts,
                    "status": "passed" if satisfied else "waiting",
                }
            )

            if satisfied:
                result = build_gate_result(
                    self.definition,
                    True,
                    attempts,
                    start_time,
                    details,
                )
                gate_state["result"] = result.details
                return result

            failure_reason = _describe_gate_failure(details)
            gate_state["status"] = "failed"
            gate_state["error"] = failure_reason
            failure_result = build_gate_result(
                self.definition,
                False,
                attempts,
                start_time,
                details,
                last_error=failure_reason,
            )
            gate_state["result"] = failure_result.details

            current_time = time.perf_counter()
            if deadline is not None and current_time >= deadline:
                raise GateTimeoutError(
                    f"Gate '{self.name}' timed out after {self.timeout_seconds} seconds",
                    result=failure_result,
                )
            if self.max_attempts is not None and attempts >= self.max_attempts:
                raise GateConditionError(
                    f"Gate '{self.name}' failed after {attempts} attempts: {failure_reason}",
                    result=failure_result,
                )

            sleep_for = self.poll_interval if attempts == 1 else max(self.poll_interval, self.retry_backoff)
            logger.debug(
                "dagster.stage.gate.retrying",
                stage=self.name,
                tenant_id=ctx.tenant_id,
                attempts=attempts,
                sleep_seconds=sleep_for,
                reason=failure_reason,
            )
            time.sleep(sleep_for)


def _describe_gate_failure(details: Mapping[str, Any]) -> str:
    clauses = details.get("clauses", []) if isinstance(details, Mapping) else []
    for clause in clauses:
        if not isinstance(clause, Mapping):
            continue
        for predicate in clause.get("all", []) or []:
            if isinstance(predicate, Mapping) and not predicate.get("passed", True):
                field = predicate.get("field")
                expected = predicate.get("expected")
                actual = predicate.get("actual")
                operator = predicate.get("operator")
                return (
                    f"{field}={actual!r} did not satisfy {operator} {expected!r}"
                )
        for predicate in clause.get("any", []) or []:
            if isinstance(predicate, Mapping) and not predicate.get("passed", True):
                field = predicate.get("field")
                expected = predicate.get("expected")
                actual = predicate.get("actual")
                operator = predicate.get("operator")
                return (
                    f"{field}={actual!r} did not satisfy {operator} {expected!r}"
                )
    return "gate conditions not satisfied"


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
        gate_payload = config.get("definition") or config
        gate_definition = GateDefinition.model_validate(gate_payload)
        evaluator = GateConditionEvaluator(gate_definition)
        retry = gate_definition.retry
        return GateStage(
            name=definition.name,
            definition=gate_definition,
            evaluator=evaluator,
            timeout_seconds=gate_definition.timeout_seconds,
            poll_interval=gate_definition.poll_interval_seconds,
            max_attempts=retry.max_attempts if retry else None,
            retry_backoff=retry.backoff_seconds if retry else gate_definition.poll_interval_seconds,
        )

    metadata = StageMetadata(
        stage_type="gate",
        state_key=None,
        output_handler=_handle_gate_output,
        output_counter=lambda _: 0,
        description="Halts pipeline execution until configured conditions are met",
        dependencies=(),
    )
    return StageRegistration(metadata=metadata, builder=_builder)


__all__ = [
    "DownloadStage",
    "GateConditionError",
    "GateStage",
    "GateTimeoutError",
    "register_download_stage",
    "register_gate_stage",
]
