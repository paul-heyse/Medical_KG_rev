"""Gate evaluation helpers used by Dagster orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

from Medical_KG_rev.orchestration.dagster.configuration import (
    GateCondition,
    GateConditionOperator,
    GateDefinition,
    GatePredicate,
)
from Medical_KG_rev.orchestration.ledger import JobLedgerEntry


class GateConditionError(RuntimeError):
    """Raised when a gate stage fails to satisfy its conditions."""

    def __init__(self, message: str, *, result: GateEvaluationResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class GateTimeoutError(GateConditionError):
    """Raised when a gate stage times out while polling for conditions."""


@dataclass(slots=True)
class GateEvaluationResult:
    """Outcome of evaluating gate conditions."""

    gate: str
    satisfied: bool
    attempts: int
    duration_seconds: float
    details: dict[str, Any]
    last_error: str | None = None


class GateConditionEvaluator:
    """Evaluate declarative gate conditions against ledger and state."""

    def __init__(self, definition: GateDefinition) -> None:
        self.definition = definition

    def evaluate(
        self,
        entry: JobLedgerEntry | None,
        state: Mapping[str, Any],
        gate_state: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        clause_results: list[dict[str, Any]] = []
        for clause in self.definition.conditions:
            result = self._evaluate_clause(clause, entry, state, gate_state)
            clause_results.append(result)
            if not result["passed"]:
                return False, {"clauses": clause_results}
        return True, {"clauses": clause_results}

    def _evaluate_clause(
        self,
        clause: GateCondition,
        entry: JobLedgerEntry | None,
        state: Mapping[str, Any],
        gate_state: dict[str, Any],
    ) -> dict[str, Any]:
        all_results: list[dict[str, Any]] = []
        any_results: list[dict[str, Any]] = []
        passed = True

        if clause.all:
            for predicate in clause.all:
                outcome = self._evaluate_predicate(predicate, entry, state, gate_state)
                all_results.append(outcome)
            if clause.all and not all(item["passed"] for item in all_results):
                passed = False

        if clause.any:
            for predicate in clause.any:
                outcome = self._evaluate_predicate(predicate, entry, state, gate_state)
                any_results.append(outcome)
            if clause.any and not any(item["passed"] for item in any_results):
                passed = False

        return {
            "description": clause.description,
            "all": all_results,
            "any": any_results,
            "passed": passed,
        }

    def _evaluate_predicate(
        self,
        predicate: GatePredicate,
        entry: JobLedgerEntry | None,
        state: Mapping[str, Any],
        gate_state: dict[str, Any],
    ) -> dict[str, Any]:
        field = predicate.field
        operator = predicate.operator
        value = predicate.value

        resolved = self._resolve_value(field, entry, state)
        passed = False
        reason: str | None = None

        if operator is GateConditionOperator.EQUALS:
            passed = resolved == value
        elif operator is GateConditionOperator.NOT_EQUALS:
            passed = resolved != value
        elif operator is GateConditionOperator.EXISTS:
            exists = resolved is not None
            expected = bool(value) if value is not None else True
            passed = exists if expected else not exists
        elif operator is GateConditionOperator.IN:
            try:
                passed = resolved in value  # type: ignore[operator]
            except TypeError:
                passed = False
                reason = "membership check failed"
        elif operator is GateConditionOperator.CHANGED:
            previous_values = gate_state.setdefault("last_values", {})
            previous = previous_values.get(field)
            previous_values[field] = resolved
            changed = previous is not None and previous != resolved
            if previous is None:
                changed = resolved is not None
            expected = bool(value) if value is not None else True
            passed = changed if expected else not changed
        else:  # pragma: no cover - exhaustive guard
            reason = f"unsupported operator {operator.value}"

        return {
            "field": field,
            "operator": operator.value,
            "expected": value,
            "actual": resolved,
            "passed": passed,
            "reason": reason,
        }

    def _resolve_value(
        self,
        field: str,
        entry: JobLedgerEntry | None,
        state: Mapping[str, Any],
    ) -> Any:
        parts = field.split(".")
        value: Any = entry
        for part in parts:
            if isinstance(value, JobLedgerEntry):
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    metadata = getattr(value, "metadata", {})
                    if isinstance(metadata, Mapping) and part in metadata:
                        value = metadata.get(part)
                    else:
                        value = getattr(value, part, None)
            elif isinstance(value, Mapping):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                break

        if value is not None:
            return value

        # Fall back to execution state lookup
        state_value: Any = state
        for part in parts:
            if isinstance(state_value, Mapping):
                state_value = state_value.get(part)
            else:
                state_value = getattr(state_value, part, None)
            if state_value is None:
                break
        return state_value


def build_gate_result(
    gate: GateDefinition,
    satisfied: bool,
    attempts: int,
    start_time: float,
    details: dict[str, Any],
    last_error: str | None = None,
) -> GateEvaluationResult:
    duration = time.perf_counter() - start_time
    return GateEvaluationResult(
        gate=gate.name,
        satisfied=satisfied,
        attempts=attempts,
        duration_seconds=duration,
        details=details,
        last_error=last_error,
    )


__all__ = [
    "GateConditionEvaluator",
    "GateConditionError",
    "GateEvaluationResult",
    "GateTimeoutError",
    "build_gate_result",
]
