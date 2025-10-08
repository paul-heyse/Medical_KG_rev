"""Utility script to deploy embedding services to Kubernetes environments."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


OVERLAYS = {
    "staging": Path("ops/k8s/overlays/staging"),
    "production": Path("ops/k8s/overlays/production"),
}


def build_kubectl_command(environment: str, *, dry_run: bool) -> list[str]:
    overlay = OVERLAYS.get(environment)
    if overlay is None:
        raise ValueError(f"Unknown environment '{environment}'")
    command = ["kubectl", "apply"]
    if dry_run:
        command.extend(["--dry-run=server"])
    command.extend(["-k", str(overlay)])
    return command


def deploy(environment: str, *, dry_run: bool = False) -> None:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl must be installed to run deployments")
    command = build_kubectl_command(environment, dry_run=dry_run)
    subprocess.run(command, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the embedding stack to Kubernetes")
    parser.add_argument(
        "environment",
        choices=sorted(OVERLAYS.keys()),
        help="Target environment to deploy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a server-side dry run without persisting changes",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    deploy(args.environment, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    main()
