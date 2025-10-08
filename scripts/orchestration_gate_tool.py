"""CLI utilities for validating, visualising, and debugging gated pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from pydantic import ValidationError

from Medical_KG_rev.orchestration.dagster.configuration import (
    GateDefinition,
    PipelineTopologyConfig,
)
from Medical_KG_rev.orchestration.dagster.gates import GateConditionEvaluator
from Medical_KG_rev.orchestration.ledger import JobLedgerEntry


def _load_topology(path: Path) -> PipelineTopologyConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Topology file {path} did not contain a mapping")
    return PipelineTopologyConfig.model_validate(data)


def _load_json_input(value: str | None) -> Any:
    if not value:
        return {}
    candidate = Path(value)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
    else:
        text = value
    return json.loads(text)


def _build_ledger_entry(payload: Mapping[str, Any] | None) -> JobLedgerEntry | None:
    if not payload:
        return None
    metadata = payload.get("metadata")
    entry = JobLedgerEntry(
        job_id=str(payload.get("job_id", "debug-job")),
        doc_key=str(payload.get("doc_key", payload.get("document_id", "debug-doc"))),
        tenant_id=str(payload.get("tenant_id", "debug-tenant")),
        pipeline=payload.get("pipeline"),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )
    if "status" in payload:
        entry.status = str(payload["status"])
    if "stage" in payload:
        entry.stage = str(payload["stage"])
    if "current_stage" in payload:
        entry.current_stage = str(payload["current_stage"])
    if "pipeline_name" in payload:
        entry.pipeline_name = str(payload["pipeline_name"])
    if "pdf_downloaded" in payload:
        entry.pdf_downloaded = bool(payload["pdf_downloaded"])
    if "pdf_ir_ready" in payload:
        entry.pdf_ir_ready = bool(payload["pdf_ir_ready"])
    if "attempts" in payload:
        entry.attempts = int(payload["attempts"])
    return entry


def _format_conditions(gate: GateDefinition) -> str:
    lines: list[str] = []
    for index, clause in enumerate(gate.conditions, start=1):
        lines.append(f"      Clause {index}:")
        if clause.description:
            lines.append(f"        description: {clause.description}")
        if clause.all:
            lines.append("        all:")
            for predicate in clause.all:
                lines.append(
                    f"          - {predicate.field} {predicate.operator.value} {predicate.value!r}"
                )
        if clause.any:
            lines.append("        any:")
            for predicate in clause.any:
                lines.append(
                    f"          - {predicate.field} {predicate.operator.value} {predicate.value!r}"
                )
    return "\n".join(lines)


def _visualise_pipeline(config: PipelineTopologyConfig, *, show_conditions: bool) -> None:
    plan = config.build_phase_plan()
    stage_map = {stage.name: stage for stage in config.stages}
    gate_map = {gate.name: gate for gate in config.gates}

    print(f"Pipeline: {config.name} (version {config.version})")
    print("Phases:")
    for order, phase in enumerate(plan.phases, start=1):
        print(f"  {order}. {phase}")
        for stage_name in plan.phase_to_stages.get(phase, ()):  # pragma: no branch - iteration
            stage = stage_map[stage_name]
            prefix = "    -"
            description = f"{prefix} {stage.name} [{stage.stage_type}]"
            if stage.stage_type == "gate":
                gate_name = stage.gate or stage.config.get("gate") or stage.name
                gate = gate_map.get(gate_name)
                if gate:
                    description += (
                        f" gate='{gate.name}' resume→{gate.resume_stage}"
                        f"@{gate.resume_phase or plan.stage_to_phase.get(gate.resume_stage, '?')}"
                    )
            print(description)
            if stage.depends_on:
                print(f"      depends_on: {', '.join(stage.depends_on)}")
            if show_conditions and stage.stage_type == "gate" and gate:
                formatted = _format_conditions(gate)
                if formatted:
                    print(formatted)
        resume_gate = plan.gate_for_phase.get(phase)
        if resume_gate:
            print(
                "      resumes from gate: "
                f"{resume_gate.name} (stage={resume_gate.resume_stage}, phase={resume_gate.resume_phase})"
            )


def handle_validate(args: argparse.Namespace) -> int:
    try:
        config = _load_topology(Path(args.pipeline))
        plan = config.build_phase_plan()
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Validation failed: {exc}")
        if isinstance(exc, ValidationError):
            print(exc)
        return 1

    print(f"Pipeline '{config.name}' validated successfully.")
    print(f"  phases: {', '.join(plan.phases)}")
    print(f"  gates: {', '.join(gate.name for gate in config.gates) or 'none'}")
    return 0


def handle_visualise(args: argparse.Namespace) -> int:
    try:
        config = _load_topology(Path(args.pipeline))
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Failed to load pipeline: {exc}")
        return 1

    _visualise_pipeline(config, show_conditions=args.show_conditions)
    return 0


def handle_debug(args: argparse.Namespace) -> int:
    try:
        config = _load_topology(Path(args.pipeline))
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Failed to load pipeline: {exc}")
        return 1

    gate_map = {gate.name: gate for gate in config.gates}
    if args.gate not in gate_map:
        print(f"Gate '{args.gate}' not found in pipeline {config.name}")
        return 1

    gate_definition = gate_map[args.gate]
    evaluator = GateConditionEvaluator(gate_definition)

    ledger_payload = _load_json_input(args.ledger)
    if ledger_payload and not isinstance(ledger_payload, Mapping):
        print("Ledger payload must be a JSON object")
        return 1
    entry = _build_ledger_entry(ledger_payload)

    state_payload = _load_json_input(args.state)
    if state_payload and not isinstance(state_payload, Mapping):
        print("Execution state must be a JSON object")
        return 1

    gate_state_payload = _load_json_input(args.gate_state)
    if gate_state_payload and not isinstance(gate_state_payload, Mapping):
        print("Gate state must be a JSON object")
        return 1

    gate_state: dict[str, Any] = dict(gate_state_payload) if gate_state_payload else {}

    passed, details = evaluator.evaluate(entry, state_payload, gate_state)
    outcome = "PASSED" if passed else "BLOCKED"
    print(f"Gate '{gate_definition.name}' evaluation: {outcome}")
    print(json.dumps(details, indent=2, sort_keys=True))

    if gate_state:
        print("Updated gate state:")
        print(json.dumps(gate_state, indent=2, sort_keys=True))
        if args.dump_gate_state:
            Path(args.dump_gate_state).write_text(
                json.dumps(gate_state, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"Gate state written to {args.dump_gate_state}")

    return 0 if passed or args.allow_failure else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a pipeline topology")
    validate_parser.add_argument("pipeline", help="Path to the pipeline topology YAML file")
    validate_parser.set_defaults(func=handle_validate)

    visualise_parser = subparsers.add_parser(
        "visualise", help="Render a human-readable view of pipeline phases and gates"
    )
    visualise_parser.add_argument("pipeline", help="Path to the pipeline topology YAML file")
    visualise_parser.add_argument(
        "--show-conditions",
        action="store_true",
        help="Include the detailed gate condition clauses in the output",
    )
    visualise_parser.set_defaults(func=handle_visualise)

    debug_parser = subparsers.add_parser(
        "debug", help="Evaluate a gate definition against ledger and state payloads"
    )
    debug_parser.add_argument("pipeline", help="Path to the pipeline topology YAML file")
    debug_parser.add_argument("gate", help="Name of the gate to evaluate")
    debug_parser.add_argument(
        "--ledger",
        default=None,
        help="JSON payload or path describing the ledger entry to evaluate",
    )
    debug_parser.add_argument(
        "--state",
        default=None,
        help="JSON payload or path describing the execution state (defaults to empty state)",
    )
    debug_parser.add_argument(
        "--gate-state",
        default=None,
        help="Optional JSON payload or path describing persisted gate evaluation state",
    )
    debug_parser.add_argument(
        "--dump-gate-state",
        default=None,
        help="Persist the updated gate state to the supplied file",
    )
    debug_parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Return a zero exit code even if the gate evaluation fails",
    )
    debug_parser.set_defaults(func=handle_debug)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
