"""CLI helpers for inspecting gated Dagster pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from Medical_KG_rev.orchestration.dagster.configuration import (
    GateDefinition,
    PipelineConfigLoader,
    PipelineTopologyConfig,
)
from Medical_KG_rev.orchestration.dagster.runtime import GateConditionEvaluator
from Medical_KG_rev.orchestration.ledger import JobLedgerEntry
from Medical_KG_rev.utils.logging import get_logger

logger = get_logger(__name__)


def _topological_order(stages: Iterable[tuple[str, list[str]]]) -> list[str]:
    graph: dict[str, set[str]] = {name: set(deps) for name, deps in stages}
    resolved: list[str] = []
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(node: str) -> None:
        if node in permanent:
            return
        if node in temporary:
            raise ValueError(f"cycle detected involving stage '{node}'")
        temporary.add(node)
        for dep in graph.get(node, set()):
            if dep in graph:
                visit(dep)
        temporary.remove(node)
        permanent.add(node)
        resolved.append(node)

    for node in graph:
        visit(node)
    return resolved


def _compute_phases(topology: PipelineTopologyConfig) -> dict[str, int]:
    stage_pairs = [(stage.name, stage.depends_on) for stage in topology.stages]
    order = _topological_order(stage_pairs)
    lookup = {stage.name: stage for stage in topology.stages}
    phase_index = 1
    phase_map: dict[str, int] = {}
    for name in order:
        stage = lookup[name]
        phase_map[name] = phase_index
        if stage.stage_type == "gate":
            phase_index += 1
    return phase_map


def _load_pipeline(base_path: Path, name: str) -> PipelineTopologyConfig:
    loader = PipelineConfigLoader(base_path)
    return loader.load(name)


def _print(msg: str) -> None:
    sys.stdout.write(msg + "\n")


def _phase_summary(topology: PipelineTopologyConfig) -> str:
    phase_map = _compute_phases(topology)
    buckets: dict[int, list[str]] = defaultdict(list)
    for stage in topology.stages:
        buckets[phase_map[stage.name]].append(stage.name)
    lines = [f"Pipeline: {topology.name} (version {topology.version})"]
    for phase in sorted(buckets):
        lines.append(f"  Phase {phase}:")
        for stage_name in buckets[phase]:
            stage = next(item for item in topology.stages if item.name == stage_name)
            marker = " [gate]" if stage.stage_type == "gate" else ""
            deps = f" (depends on {', '.join(stage.depends_on)})" if stage.depends_on else ""
            lines.append(f"    - {stage_name} [{stage.stage_type}]{marker}{deps}")
    if topology.gates:
        lines.append("  Gates:")
        for gate in topology.gates:
            clause_parts = []
            for clause in gate.condition.clauses:
                value = json.dumps(clause.value)
                clause_parts.append(f"{clause.field} {clause.operator.value} {value}")
            lines.append(
                "    - {name}: resume='{resume}' logic={logic} clauses=[{clauses}]".format(
                    name=gate.name,
                    resume=gate.resume_stage,
                    logic=gate.condition.logic,
                    clauses="; ".join(clause_parts),
                )
            )
    return "\n".join(lines)


def _load_json(path: str) -> dict[str, Any]:
    if path == "-":
        payload = sys.stdin.read()
    else:
        payload = Path(path).read_text()
    return json.loads(payload)


def _build_entry(data: dict[str, Any]) -> JobLedgerEntry:
    required = {"job_id", "doc_key", "tenant_id"}
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"ledger entry missing required fields: {', '.join(missing)}")
    allowed = set(JobLedgerEntry.__annotations__.keys())
    skip = {"created_at", "updated_at", "completed_at"}
    kwargs = {key: data[key] for key in allowed if key in data and key not in skip}
    entry = JobLedgerEntry(**kwargs)
    if "metadata" in data and not isinstance(entry.metadata, dict):
        entry.metadata = dict(data["metadata"])
    return entry


def _describe_gate(gate: GateDefinition) -> str:
    lines = [f"Gate '{gate.name}' → resume '{gate.resume_stage}'"]
    lines.append(f"  logic: {gate.condition.logic}")
    if gate.timeout_seconds:
        lines.append(f"  timeout_seconds: {gate.timeout_seconds}")
    if gate.retry:
        lines.append(
            f"  retry: attempts={gate.retry.max_attempts} backoff={gate.retry.backoff_seconds}s"
        )
    for clause in gate.condition.clauses:
        lines.append(
            f"  clause: field='{clause.field}' operator={clause.operator.value} value={clause.value!r}"
        )
    return "\n".join(lines)


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        topology = _load_pipeline(args.base_path, args.pipeline)
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("gate_tool.validate.error", pipeline=args.pipeline, error=str(exc))
        _print(f"Validation failed: {exc}")
        return 1
    _print("Validation succeeded.")
    _print(_phase_summary(topology))
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    topology = _load_pipeline(args.base_path, args.pipeline)
    _print(_phase_summary(topology))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    topology = _load_pipeline(args.base_path, args.pipeline)
    gate = next((item for item in topology.gates if item.name == args.gate), None)
    if gate is None:
        _print(f"Gate '{args.gate}' not found in pipeline '{args.pipeline}'.")
        return 1
    entry_payload = _load_json(args.ledger)
    previous_payload = _load_json(args.previous) if args.previous else {}
    try:
        entry = _build_entry(entry_payload)
    except ValueError as exc:
        _print(f"Invalid ledger entry: {exc}")
        return 1
    evaluator = GateConditionEvaluator(gate)
    satisfied, observed = evaluator.evaluate(entry, previous_observed=previous_payload)
    _print(_describe_gate(gate))
    _print("Observed values:")
    for field, value in observed.items():
        _print(f"  {field}: {value!r}")
    _print(f"Gate result: {'PASSED' if satisfied else 'BLOCKED'}")
    return 0 if satisfied or args.allow_failure else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dagster gate utilities")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("config/orchestration/pipelines"),
        help="Pipeline topology directory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate pipeline configuration")
    validate_parser.add_argument("pipeline", help="Pipeline name (YAML stem)")
    validate_parser.set_defaults(func=cmd_validate)

    viz_parser = subparsers.add_parser("visualize", help="Visualize phase ordering")
    viz_parser.add_argument("pipeline", help="Pipeline name (YAML stem)")
    viz_parser.set_defaults(func=cmd_visualize)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a gate against a ledger entry")
    eval_parser.add_argument("pipeline", help="Pipeline name (YAML stem)")
    eval_parser.add_argument("gate", help="Gate identifier")
    eval_parser.add_argument("--ledger", required=True, help="Path to ledger entry JSON (use '-' for stdin)")
    eval_parser.add_argument(
        "--previous",
        help="Optional JSON with previous observed values for 'changed' clauses",
    )
    eval_parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Return success even if the gate conditions are not satisfied",
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
