"""Detect imports referencing legacy embedding modules."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

LEGACY_MODULES = {
    "bge_embedder",
    "splade_embedder",
    "manual_batching",
    "token_counter",
}


def _contains_legacy_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if any(part in LEGACY_MODULES for part in alias.name.split(".")):
                return True
    if isinstance(node, ast.ImportFrom):
        if node.module and any(part in LEGACY_MODULES for part in node.module.split(".")):
            return True
        for alias in node.names:
            if alias.name in LEGACY_MODULES:
                return True
    return False


def check_file(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=str(path))
    for node in ast.walk(tree):
        if _contains_legacy_import(node):
            print(f"{path}:{getattr(node, 'lineno', 0)}: dangling legacy embedding import detected")
            return False
    return True


def main() -> int:
    root = Path("src")
    if not root.exists():
        print("src directory not found; nothing to scan")
        return 0
    success = True
    for file_path in root.rglob("*.py"):
        if not check_file(file_path):
            success = False
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
