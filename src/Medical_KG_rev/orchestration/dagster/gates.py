"""Gate execution primitives for Dagster pipelines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import structlog

from Medical_KG_rev.orchestration.dagster.configuration import (
    GateDefinition,
    GateOperator,
    StageDefinition,
)
from Medical_KG_rev.orchestration.ledger import JobLedger, JobLedgerEntry
from Medical_KG_rev.orchestration.stages.contracts import StageContext

logger = structlog.get_logger(__name__)


class GateConditionError(RuntimeError):
    """Raised when a gate condition fails or times out."""

    def __init__(self, gate_name: str, message: str, *, status: str = "failed", attempts: int = 0):
        super().__init__(message)
        self.gate_name = gate_name
        self.status = status
        self.attempts = attempts


@dataclass(slots=True)
class GateEvaluationResult:
    """Structured result emitted by gate evaluation."""

    gate_name: str
    status: str
    satisfied: bool
    attempts: int
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def should_resume(self) -> bool:
        return self.satisfied and self.status == "satisfied"


@dataclass(slots=True)
class _ClauseResult:
    satisfied: bool
    fatal: bool
    reason: str | None
    details: dict[str, Any]


class GateConditionEvaluator:
    """Evaluate gate definitions against the Job Ledger."""

    def __init__(self, ledger: JobLedger) -> None:
        self._ledger = ledger

    def evaluate(self, job_id: str, gate: GateDefinition) -> GateEvaluationResult:
        start = time.perf_counter()
        attempts = 0
        while True:
            attempts += 1
            entry = self._ledger.get(job_id)
            if entry is None:
                raise GateConditionError(gate.name, f"job '{job_id}' not found in ledger", attempts=attempts)
            clause_result = self._evaluate_clauses(entry, gate)
            elapsed = time.perf_counter() - start
            if clause_result.fatal:
                self._ledger.record_gate_state(
                    job_id,
                    gate.name,
                    status="failed",
                    reason=clause_result.reason,
                    attempts=attempts,
                    elapsed_seconds=elapsed,
                    extra={"last_values": clause_result.details.get("last_values", {})},
                )
                raise GateConditionError(
                    gate.name,
                    clause_result.reason or "gate evaluation failed",
                    status="failed",
                    attempts=attempts,
                )
            if clause_result.satisfied:
                self._ledger.record_gate_state(
                    job_id,
                    gate.name,
                    status="satisfied",
                    attempts=attempts,
                    elapsed_seconds=elapsed,
                    extra={"last_values": clause_result.details.get("last_values", {})},
                )
                return GateEvaluationResult(
                    gate_name=gate.name,
                    status="satisfied",
                    satisfied=True,
                    attempts=attempts,
                    elapsed_seconds=elapsed,
                    metadata={"mode": gate.condition.mode},
                )
            if elapsed >= gate.timeout_seconds:
                reason = clause_result.reason or "gate condition not met before timeout"
                self._ledger.record_gate_state(
                    job_id,
                    gate.name,
                    status="timeout",
                    reason=reason,
                    attempts=attempts,
                    elapsed_seconds=elapsed,
                    extra={"last_values": clause_result.details.get("last_values", {})},
                )
                raise GateConditionError(
                    gate.name,
                    reason,
                    status="timeout",
                    attempts=attempts,
                )
            time.sleep(gate.poll_interval_seconds)

    def _evaluate_clauses(
        self,
        entry: JobLedgerEntry,
        gate: GateDefinition,
    ) -> _ClauseResult:
        satisfied: list[bool] = []
        fatal = False
        reason: str | None = None
        last_values = self._last_values(entry, gate.name)
        new_last_values = dict(last_values)

        for clause in gate.condition.clauses:
            value = self._resolve_field(entry, clause.field)
            clause_ok = False
            clause_reason: str | None = None
            if clause.operator == GateOperator.EXISTS:
                clause_ok = value is not None
                if not clause_ok:
                    clause_reason = f"field '{clause.field}' is missing"
            elif clause.operator == GateOperator.EQUALS:
                clause_ok = value == clause.value
                if not clause_ok:
                    clause_reason = (
                        f"field '{clause.field}' expected '{clause.value}' but found '{value}'"
                    )
            elif clause.operator == GateOperator.CHANGED:
                previous = last_values.get(clause.field)
                clause_ok = previous is not None and previous != value
                clause_reason = (
                    f"field '{clause.field}' has not changed from '{previous}'"
                    if not clause_ok
                    else None
                )
                new_last_values[clause.field] = value
            else:  # pragma: no cover - defensive
                fatal = True
                clause_reason = f"unsupported gate operator '{clause.operator}'"
            satisfied.append(clause_ok)
            if clause_reason and reason is None:
                reason = clause_reason

        overall = all(satisfied) if gate.condition.mode == "all" else any(satisfied)
        details = {"last_values": new_last_values}
        return _ClauseResult(satisfied=overall, fatal=fatal, reason=reason, details=details)

    def _resolve_field(self, entry: JobLedgerEntry, field_path: str) -> Any:
        parts = field_path.split(".")
        current: Any = entry
        for part in parts:
            if isinstance(current, Mapping):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
            if current is None:
                break
        return current

    def _last_values(self, entry: JobLedgerEntry, gate_name: str) -> dict[str, Any]:
        state = entry.gate_state.get(gate_name, {})
        last_values = state.get("last_values")
        if isinstance(last_values, Mapping):
            return dict(last_values)
        return {}


class GateStage:
    """Runtime stage that evaluates a gate without producing downstream output."""

    def __init__(self, definition: StageDefinition, gate: GateDefinition) -> None:
        self._definition = definition
        self._gate = gate

    @property
    def gate(self) -> GateDefinition:
        return self._gate

    def execute(
        self,
        ctx: StageContext,
        state: Mapping[str, Any],
        *,
        ledger: JobLedger,
    ) -> GateEvaluationResult:
        job_id = ctx.job_id or state.get("job_id")
        if not isinstance(job_id, str):
            raise GateConditionError(self._gate.name, "gate evaluation requires a job identifier")
        evaluator = GateConditionEvaluator(ledger)
        result = evaluator.evaluate(job_id, self._gate)
        ledger.set_phase(job_id, f"phase-{self._definition.phase_index + 1}")
        return result


__all__ = [
    "GateConditionEvaluator",
    "GateConditionError",
    "GateEvaluationResult",
    "GateStage",
]
