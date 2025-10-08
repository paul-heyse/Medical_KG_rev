#!/usr/bin/env python
"""Audit namespace configuration for tenant-isolated embedding access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Medical_KG_rev.services.embedding.namespace.loader import load_namespace_configs


def _audit_namespaces(config_dir: Path | None = None) -> list[str]:
    configs = load_namespace_configs(config_dir)
    issues: list[str] = []
    for namespace, config in configs.items():
        tenants = set(config.allowed_tenants)
        scopes = set(config.allowed_scopes)
        if not tenants:
            issues.append(f"{namespace}: missing allowed_tenants entry")
        if "all" not in tenants and len(tenants) == 0:
            issues.append(f"{namespace}: no tenants authorised")
        if "embed:write" not in scopes and "*" not in scopes:
            issues.append(f"{namespace}: embed:write scope missing")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace-dir",
        type=Path,
        default=None,
        help="Optional path to the namespace directory (defaults to config/embedding/namespaces)",
    )
    args = parser.parse_args()

    issues = _audit_namespaces(args.namespace_dir)
    if issues:
        print("Found potential tenant isolation issues:")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print("All namespaces define tenant access controls and embed:write scope.")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
