"""Utility script to sync legacy adapter entry points with the plugin framework."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Python >=3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore

from Medical_KG_rev.adapters.plugins.bootstrap import get_plugin_manager


def _load_pyproject(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _generate_entry_points() -> dict[str, str]:
    manager = get_plugin_manager(refresh=True)
    entry_points: dict[str, str] = {}
    for metadata in manager.list_metadata():
        plugin = manager._adapters.get(metadata.name)  # type: ignore[attr-defined]
        if plugin is None:
            continue
        entry_points[metadata.name] = f"{plugin.__class__.__module__}:{plugin.__class__.__qualname__}"
    return dict(sorted(entry_points.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml to inspect",
    )
    parser.add_argument(
        "--print", action="store_true", help="Print recommended entry points to stdout"
    )
    args = parser.parse_args(argv)

    pyproject = _load_pyproject(args.pyproject)
    entry_points = _generate_entry_points()

    if args.print or not pyproject:
        for name, target in entry_points.items():
            print(f"{name} = {target}")
        return 0

    project = pyproject.setdefault("project", {})  # type: ignore[assignment]
    entry_section = project.setdefault("entry-points", {})  # type: ignore[assignment]
    adapters = entry_section.setdefault("medical_kg.adapters", {})  # type: ignore[assignment]
    adapters.update(entry_points)
    print("Updated entry points for medical_kg.adapters. Persist changes manually if desired.")
    for name, target in entry_points.items():
        print(f" - {name}: {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
