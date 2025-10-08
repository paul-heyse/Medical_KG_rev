"""Built-in orchestration stage plugins (download and gate)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import structlog

from Medical_KG_rev.orchestration.dagster.configuration import (
    GateConditionClause,
    GateConditionMatch,
    GateConditionOperator,
    GateDefinition,
    GateRetryConfig,
    StageDefinition,
)
from Medical_KG_rev.orchestration.dagster.stage_registry import (
    StageMetadata,
    StageRegistration,
)
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stages.contracts import StageContext
from Medical_KG_rev.observability.metrics import (
    record_gate_evaluation,
    record_gate_timeout,
)

logger = structlog.get_logger(__name__)


class GateConditionError(RuntimeError):
    """Raised when a gate stage condition fails."""


def _sequence_length(value: Any) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def _handle_download_output(state: dict[str, Any], _: str, output: Any) -> None:
    state["downloaded_files"] = output


def _handle_gate_output(state: dict[str, Any], _: str, output: Any) -> None:
    if hasattr(output, "gate"):
        gates = state.setdefault("gate_results", {})
        gates[output.gate] = output.as_dict()


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
class GateEvaluationResult:
    """Structured result returned by :class:`GateStage` evaluations."""

    gate: str
    resume_stage: str
    attempts: int
    elapsed_seconds: float
    details: dict[str, Any]
    skip_download_on_resume: bool
    phase: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "resume_stage": self.resume_stage,
            "attempts": self.attempts,
            "elapsed_seconds": self.elapsed_seconds,
            "details": dict(self.details),
            "skip_download_on_resume": self.skip_download_on_resume,
            "phase": self.phase,
        }


@dataclass(slots=True)
class GateStage:
    """Gate stage validating ledger conditions before proceeding."""

    definition: GateDefinition
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic

    def execute(
        self,
        ctx: StageContext,
        state: Mapping[str, Any],
        ledger: JobLedger,
    ) -> GateEvaluationResult:
        job_id = ctx.job_id
        if not job_id:
            raise GateConditionError(
                f"Gate '{self.definition.name}' requires a job identifier in the stage context"
            )

        pipeline_name = ctx.pipeline_name or state.get("pipeline") or "unknown"
        retry = self.definition.retry or GateRetryConfig()
        timeout = self.definition.condition.timeout_seconds
        deadline = None if timeout is None else self.monotonic() + timeout
        attempts = 0
        start = self.monotonic()
        last_details: dict[str, Any] = {}

        while True:
            attempts += 1
            entry = ledger.get(job_id)
            if entry is None:
                raise GateConditionError(f"Gate '{self.definition.name}' missing ledger entry")

            satisfied, details = self._evaluate(entry)
            last_details = details

            if satisfied:
                elapsed = self.monotonic() - start
                record_gate_evaluation(self.definition.name, pipeline_name, "success")
                self._update_metadata(
                    ledger,
                    job_id,
                    status="passed",
                    attempts=attempts,
                    elapsed=elapsed,
                    details=details,
                    phase=state.get("_phase"),
                )
                logger.debug(
                    "dagster.stage.gate.passed",
                    stage=self.definition.name,
                    tenant_id=ctx.tenant_id,
                    attempts=attempts,
                    resume_stage=self.definition.resume_stage,
                )
                return GateEvaluationResult(
                    gate=self.definition.name,
                    resume_stage=self.definition.resume_stage,
                    attempts=attempts,
                    elapsed_seconds=elapsed,
                    details=details,
                    skip_download_on_resume=self.definition.skip_download_on_resume,
                    phase=state.get("_phase"),
                )

            record_gate_evaluation(self.definition.name, pipeline_name, "failure")
            now = self.monotonic()
            timed_out = deadline is not None and now >= deadline
            exceeded_attempts = retry.max_attempts and attempts >= retry.max_attempts

            if timed_out:
                record_gate_timeout(self.definition.name, pipeline_name)
                self._update_metadata(
                    ledger,
                    job_id,
                    status="timeout",
                    attempts=attempts,
                    elapsed=now - start,
                    details=details,
                    phase=state.get("_phase"),
                    error="timeout",
                )
                logger.warning(
                    "dagster.stage.gate.timeout",
                    stage=self.definition.name,
                    tenant_id=ctx.tenant_id,
                    timeout_seconds=timeout,
                )
                raise GateConditionError(
                    f"Gate '{self.definition.name}' timed out after {timeout} seconds"
                )

            if exceeded_attempts and timeout is None:
                self._update_metadata(
                    ledger,
                    job_id,
                    status="failed",
                    attempts=attempts,
                    elapsed=now - start,
                    details=details,
                    phase=state.get("_phase"),
                    error="max_attempts",
                )
                logger.warning(
                    "dagster.stage.gate.attempts_exhausted",
                    stage=self.definition.name,
                    tenant_id=ctx.tenant_id,
                    attempts=attempts,
                )
                raise GateConditionError(
                    f"Gate '{self.definition.name}' failed after {attempts} attempts"
                )

            sleep_for = max(self.definition.condition.poll_interval_seconds, retry.delay_seconds)
            if deadline is not None:
                remaining = max(deadline - now, 0.0)
                sleep_for = min(sleep_for, remaining)
                if sleep_for <= 0:
                    record_gate_timeout(self.definition.name, pipeline_name)
                    self._update_metadata(
                        ledger,
                        job_id,
                        status="timeout",
                        attempts=attempts,
                        elapsed=now - start,
                        details=details,
                        phase=state.get("_phase"),
                        error="timeout",
                    )
                    raise GateConditionError(
                        f"Gate '{self.definition.name}' timed out before meeting conditions"
                    )

            logger.debug(
                "dagster.stage.gate.waiting",
                stage=self.definition.name,
                tenant_id=ctx.tenant_id,
                attempts=attempts,
                sleep_seconds=sleep_for,
            )
            self.sleep(sleep_for)

    def _evaluate(self, entry) -> tuple[bool, dict[str, Any]]:
        clause_results: dict[str, Any] = {}
        matches: list[bool] = []
        for clause in self.definition.condition.clauses:
            value = self._resolve_field(entry, clause.field)
            passed = self._evaluate_clause(clause, value)
            clause_results[clause.field] = {
                "operator": clause.operator.value,
                "expected": clause.value,
                "previous": clause.previous_value,
                "actual": value,
                "passed": passed,
            }
            matches.append(passed)
        if self.definition.condition.match is GateConditionMatch.ANY:
            satisfied = any(matches)
        else:
            satisfied = all(matches)
        clause_results["summary"] = {
            "match": self.definition.condition.match.value,
            "total": len(matches),
            "passed": sum(1 for item in matches if item),
        }
        return satisfied, clause_results

    @staticmethod
    def _resolve_field(entry, path: str) -> Any:
        value: Any = entry
        for part in path.split("."):
            if isinstance(value, Mapping):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                break
        return value

    @staticmethod
    def _evaluate_clause(clause: GateConditionClause, value: Any) -> bool:
        if clause.operator is GateConditionOperator.EQUALS:
            return value == clause.value
        if clause.operator is GateConditionOperator.NOT_EQUALS:
            return value != clause.value
        if clause.operator is GateConditionOperator.EXISTS:
            return value is not None
        if clause.operator is GateConditionOperator.NOT_EXISTS:
            return value is None
        if clause.operator is GateConditionOperator.IN:
            assert clause.value is not None
            return value in clause.value  # type: ignore[operator]
        if clause.operator is GateConditionOperator.NOT_IN:
            assert clause.value is not None
            return value not in clause.value  # type: ignore[operator]
        if clause.operator is GateConditionOperator.CHANGED:
            previous = clause.previous_value
            if clause.value is not None:
                return value == clause.value and value != previous
            return value != previous
        return False

    def _update_metadata(
        self,
        ledger: JobLedger,
        job_id: str,
        *,
        status: str,
        attempts: int,
        elapsed: float,
        details: Mapping[str, Any],
        phase: str | None,
        error: str | None = None,
    ) -> None:
        entry = ledger.get(job_id)
        existing_gate = {}
        if entry is not None:
            gate_meta = entry.metadata.get("gate")
            if isinstance(gate_meta, Mapping):
                existing_gate = dict(gate_meta)
        existing_gate[self.definition.name] = {
            "status": status,
            "attempts": attempts,
            "elapsed_seconds": elapsed,
            "resume_stage": self.definition.resume_stage,
            "skip_download_on_resume": self.definition.skip_download_on_resume,
            "details": dict(details),
            "phase": phase,
            "error": error,
        }
        payload: dict[str, Any] = {"gate": existing_gate}
        if phase:
            payload["phase"] = phase
        ledger.update_metadata(job_id, payload)


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
        gate_config = config.get("gate")
        if not gate_config:
            raise ValueError(f"Stage '{definition.name}' requires a gate configuration block")
        gate_definition = GateDefinition.model_validate(gate_config)
        return GateStage(definition=gate_definition)

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
    "GateConditionError",
    "GateEvaluationResult",
    "GateStage",
    "register_download_stage",
    "register_gate_stage",
]
