"""CLI utilities for inspecting pipeline gate configuration."""

from __future__ import annotations

import argparse
import json
from typing import Any

from Medical_KG_rev.orchestration.dagster.configuration import (
    PipelineConfigLoader,
    PipelinePhase,
)


def _format_clause(clause: dict[str, Any]) -> str:
    operator = clause.get("operator", "equals")
    field = clause.get("field", "<field>")
    expected = clause.get("value")
    previous = clause.get("previous_value")
    rendered = f"{field} {operator} {expected!r}"
    if previous is not None and operator == "changed":
        rendered += f" (previous={previous!r})"
    return rendered


def _summarise_gate(config, gate) -> dict[str, Any]:
    stage = next((stage for stage in config.stages if stage.name == gate.stage), None)
    clauses = [clause.model_dump() for clause in gate.condition.clauses]
    return {
        "name": gate.name,
        "stage": gate.stage,
        "resume_stage": gate.resume_stage,
        "phase": stage.phase.value if stage and stage.phase else None,
        "skip_download_on_resume": gate.skip_download_on_resume,
        "timeout_seconds": gate.condition.timeout_seconds,
        "poll_interval_seconds": gate.condition.poll_interval_seconds,
        "match": gate.condition.match.value,
        "retry": gate.retry.model_dump() if gate.retry else None,
        "clauses": clauses,
    }


def inspect_gates(pipeline: str, *, base_path: str | None = None) -> dict[str, Any]:
    loader = PipelineConfigLoader(base_path=base_path)
    config = loader.load(pipeline)
    gate_summaries = [_summarise_gate(config, gate) for gate in config.gates]
    phases: dict[str, list[str]] = {phase.value: [] for phase in PipelinePhase}
    for stage in config.stages:
        phase_name = stage.phase.value if stage.phase else PipelinePhase.PRE_GATE.value
        phases.setdefault(phase_name, []).append(stage.name)
    return {
        "pipeline": config.name,
        "version": config.version,
        "gates": gate_summaries,
        "phases": {key: value for key, value in phases.items() if value},
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        f"Pipeline: {report['pipeline']} ({report['version']})",
        "",
        "Execution phases:",
    ]
    for phase, stages in report["phases"].items():
        lines.append(f"  - {phase}: {', '.join(stages)}")
    if not report["gates"]:
        lines.append("\nNo gates configured.")
        return "\n".join(lines)
    lines.append("\nGates:")
    for gate in report["gates"]:
        lines.append(
            f"  - {gate['name']} (stage={gate['stage']} → resume {gate['resume_stage']}, phase={gate['phase']})"
        )
        lines.append(
            f"      skip_download_on_resume={gate['skip_download_on_resume']} | timeout={gate['timeout_seconds']}s | poll={gate['poll_interval_seconds']}s"
        )
        if gate["retry"]:
            retry = gate["retry"]
            lines.append(
                f"      retry: max_attempts={retry.get('max_attempts')} delay={retry.get('delay_seconds')}s"
            )
        lines.append(f"      clauses ({gate['match']}):")
        for clause in gate["clauses"]:
            lines.append(f"        * {_format_clause(clause)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect gate configuration for a pipeline")
    parser.add_argument("pipeline", help="Pipeline topology name (e.g. pdf-two-phase)")
    parser.add_argument("--base-path", dest="base_path", help="Override topology directory", default=None)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    args = parser.parse_args(argv)

    report = inspect_gates(args.pipeline, base_path=args.base_path)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI execution
    raise SystemExit(main())
