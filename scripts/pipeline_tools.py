"""CLI utilities for working with pipeline and gate definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from Medical_KG_rev.orchestration.dagster.configuration import PipelineTopologyConfig
from Medical_KG_rev.orchestration.dagster.gates import GateConditionError, GateStage
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stages.contracts import StageContext


def _load_pipeline(path: Path) -> PipelineTopologyConfig:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Pipeline file '{path}' is empty or invalid")
    return PipelineTopologyConfig.model_validate(data)


def _load_ledger_entries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [entry for entry in raw if isinstance(entry, dict)]
    raise ValueError("Ledger snapshot must be a JSON object or array")


def _stage_summary(pipeline: PipelineTopologyConfig) -> str:
    groups: dict[int, list[str]] = {}
    for stage in pipeline.stages:
        groups.setdefault(stage.phase_index, []).append(stage.name)
    lines = [f"Pipeline: {pipeline.name} (v{pipeline.version})"]
    for phase in sorted(groups):
        names = " → ".join(groups[phase])
        lines.append(f"  phase-{phase}: {names}")
    for gate in pipeline.gates:
        lines.append(
            f"  gate {gate.name}: resume='{gate.resume_stage}', timeout={gate.timeout_seconds}s"
        )
    return "\n".join(lines)


def cmd_validate(args: argparse.Namespace) -> int:
    pipeline = _load_pipeline(Path(args.pipeline))
    print(_stage_summary(pipeline))
    print("Validation succeeded: dependency graph and gates are consistent.")
    return 0


def cmd_test_gate(args: argparse.Namespace) -> int:
    pipeline = _load_pipeline(Path(args.pipeline))
    gate = next((item for item in pipeline.gates if item.name == args.gate), None)
    if gate is None:
        raise SystemExit(f"Gate '{args.gate}' not found in pipeline '{pipeline.name}'")
    entries = _load_ledger_entries(Path(args.ledger))
    ledger = JobLedger()
    stage_def = next(stage for stage in pipeline.stages if stage.gate == gate.name)
    stage = GateStage(stage_def, gate)
    for payload in entries:
        job_id = str(payload.get("job_id") or payload.get("id") or "job")
        entry = ledger.create(
            job_id=job_id,
            doc_key=str(payload.get("doc_key", job_id)),
            tenant_id=str(payload.get("tenant_id", "tenant")),
            pipeline=pipeline.name,
            metadata=payload.get("metadata", {}),
        )
        entry.pdf_ir_ready = bool(payload.get("pdf_ir_ready", False))
        entry.pdf_downloaded = bool(payload.get("pdf_downloaded", False))
        ledger._entries[entry.job_id] = entry  # type: ignore[attr-defined]
        try:
            result = stage.execute(
                StageContext(tenant_id=entry.tenant_id, job_id=entry.job_id),
                {},
                ledger=ledger,
            )
        except GateConditionError as exc:  # pragma: no cover - exercised in CLI usage
            print(f"job={entry.job_id} gate={gate.name} status={exc.status} reason={exc}")
            continue
        print(
            "job={job} gate={gate} status={status} attempts={attempts}".format(
                job=entry.job_id,
                gate=gate.name,
                status=result.status,
                attempts=result.attempts,
            )
        )
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    pipeline = _load_pipeline(Path(args.pipeline))
    print(_stage_summary(pipeline))
    return 0


def cmd_debug_gate(args: argparse.Namespace) -> int:
    entries = _load_ledger_entries(Path(args.ledger))
    for payload in entries:
        job_id = payload.get("job_id") or payload.get("id") or "job"
        gates = {
            key.split("gate.", 1)[1]: value
            for key, value in payload.get("metadata", {}).items()
            if isinstance(key, str) and key.startswith("gate.")
        }
        print(f"job={job_id}")
        if not gates:
            print("  no gate metadata recorded")
            continue
        for name, value in gates.items():
            print(f"  {name}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline utility toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate a pipeline topology")
    validate.add_argument("pipeline", help="Path to the pipeline YAML file")
    validate.set_defaults(func=cmd_validate)

    test_gate = sub.add_parser("test-gate", help="Evaluate a gate against a ledger snapshot")
    test_gate.add_argument("pipeline", help="Path to the pipeline YAML file")
    test_gate.add_argument("gate", help="Gate name to evaluate")
    test_gate.add_argument("ledger", help="Path to a JSON ledger entry or list")
    test_gate.set_defaults(func=cmd_test_gate)

    visualize = sub.add_parser("visualize", help="Print stage order grouped by phase")
    visualize.add_argument("pipeline", help="Path to the pipeline YAML file")
    visualize.set_defaults(func=cmd_visualize)

    debug = sub.add_parser("debug", help="Inspect gate metadata in a ledger snapshot")
    debug.add_argument("ledger", help="Path to a JSON ledger entry or list")
    debug.set_defaults(func=cmd_debug_gate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
