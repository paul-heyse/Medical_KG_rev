#!/usr/bin/env python3
"""Find duplicate code patterns across the repository.

This script uses AST analysis to identify duplicate functions, classes, and
imports across the Medical_KG_rev repository.

Usage:
    python scripts/find_duplicate_code.py [path]

Examples:
    python scripts/find_duplicate_code.py src/Medical_KG_rev/
    python scripts/find_duplicate_code.py tests/
"""

import argparse
import ast
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


class DuplicateDetector:
    """Detects duplicate code patterns using AST analysis."""

    def __init__(self) -> None:
        self.functions: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.classes: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.imports: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.errors: List[str] = []

    def analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for duplicates."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            self._visit_node(tree, file_path)

        except SyntaxError as e:
            self.errors.append(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            self.errors.append(f"Error analyzing {file_path}: {e}")

    def _visit_node(self, node: ast.AST, file_path: Path) -> None:
        """Visit AST nodes and collect function/class signatures."""
        if isinstance(node, ast.FunctionDef):
            signature = self._get_function_signature(node)
            self.functions[signature].append((str(file_path), node.lineno))

        elif isinstance(node, ast.ClassDef):
            signature = self._get_class_signature(node)
            self.classes[signature].append((str(file_path), node.lineno))

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            signature = self._get_import_signature(node)
            self.imports[signature].append((str(file_path), node.lineno))

        # Recursively visit child nodes
        for child in ast.iter_child_nodes(node):
            self._visit_node(child, file_path)

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get a normalized function signature."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        # Include decorators in signature
        decorators = [ast.unparse(d) for d in node.decorator_list]
        decorator_str = f"@{','.join(decorators)}" if decorators else ""

        return f"{decorator_str}def {node.name}({','.join(args)})"

    def _get_class_signature(self, node: ast.ClassDef) -> str:
        """Get a normalized class signature."""
        bases = [ast.unparse(base) for base in node.bases]
        bases_str = f"({','.join(bases)})" if bases else ""

        return f"class {node.name}{bases_str}"

    def _get_import_signature(self, node: ast.AST) -> str:
        """Get a normalized import signature."""
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            return f"import {','.join(names)}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            return f"from {module} import {','.join(names)}"
        return ""

    def find_duplicates(self) -> Dict[str, List[Tuple[str, int]]]:
        """Find all duplicate patterns."""
        duplicates = {}

        # Find duplicate functions
        for signature, locations in self.functions.items():
            if len(locations) > 1:
                duplicates[f"function:{signature}"] = locations

        # Find duplicate classes
        for signature, locations in self.classes.items():
            if len(locations) > 1:
                duplicates[f"class:{signature}"] = locations

        # Find duplicate imports
        for signature, locations in self.imports.items():
            if len(locations) > 1:
                duplicates[f"import:{signature}"] = locations

        return duplicates

    def generate_report(self) -> str:
        """Generate a comprehensive duplicate code report."""
        duplicates = self.find_duplicates()

        report = []
        report.append("=" * 60)
        report.append("DUPLICATE CODE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        if self.errors:
            report.append("ERRORS:")
            for error in self.errors:
                report.append(f"  ❌ {error}")
            report.append("")

        if not duplicates:
            report.append("✅ No duplicate code patterns found!")
            return "\n".join(report)

        report.append(f"Found {len(duplicates)} duplicate patterns:")
        report.append("")

        for pattern_type, locations in duplicates.items():
            report.append(f"🔍 {pattern_type}")
            for file_path, line_num in locations:
                report.append(f"  📁 {file_path}:{line_num}")
            report.append("")

        # Summary statistics
        function_duplicates = sum(1 for k in duplicates.keys() if k.startswith("function:"))
        class_duplicates = sum(1 for k in duplicates.keys() if k.startswith("class:"))
        import_duplicates = sum(1 for k in duplicates.keys() if k.startswith("import:"))

        report.append("SUMMARY:")
        report.append(f"  Function duplicates: {function_duplicates}")
        report.append(f"  Class duplicates: {class_duplicates}")
        report.append(f"  Import duplicates: {import_duplicates}")
        report.append(f"  Total duplicates: {len(duplicates)}")

        return "\n".join(report)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Find duplicate code patterns")
    parser.add_argument("path", nargs="?", default="src/Medical_KG_rev/",
                       help="Path to analyze (default: src/Medical_KG_rev/)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path {path} does not exist")
        sys.exit(1)

    detector = DuplicateDetector()

    # Find all Python files
    python_files = []
    if path.is_file() and path.suffix == ".py":
        python_files = [path]
    else:
        python_files = list(path.rglob("*.py"))

    print(f"Analyzing {len(python_files)} Python files...")

    for file_path in python_files:
        detector.analyze_file(file_path)

    # Generate and output report
    report = detector.generate_report()

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Exit with error code if duplicates found
    duplicates = detector.find_duplicates()
    if duplicates:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
