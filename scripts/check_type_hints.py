#!/usr/bin/env python3
"""Check type hint coverage and modernization opportunities.

This script analyzes Python files for type hint coverage and identifies
opportunities for modernization to current Python type hint conventions.

Usage:
    python scripts/check_type_hints.py [path]

Examples:
    python scripts/check_type_hints.py src/Medical_KG_rev/
    python scripts/check_type_hints.py tests/

"""

import argparse
import ast
import sys
from pathlib import Path


class TypeHintChecker:
    """Checks type hint coverage and modernization opportunities."""

    def __init__(self) -> None:
        self.files_analyzed = 0
        self.functions_without_return_types = []
        self.functions_without_param_types = []
        self.deprecated_optional_usage = []
        self.deprecated_dict_list_usage = []
        self.errors = []
        self.domain_stats: dict[str, dict[str, int]] = {}

    def analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for type hint issues."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            domain_type = self._get_domain_type(file_path)

            # Initialize domain stats if needed
            if domain_type not in self.domain_stats:
                self.domain_stats[domain_type] = {
                    "files": 0,
                    "functions": 0,
                    "missing_return_types": 0,
                    "missing_param_types": 0,
                    "deprecated_optional": 0,
                    "deprecated_dict_list": 0,
                }

            self.domain_stats[domain_type]["files"] += 1
            self._visit_node(tree, file_path, domain_type)
            self.files_analyzed += 1

        except SyntaxError as e:
            self.errors.append(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            self.errors.append(f"Error analyzing {file_path}: {e}")

    def _get_domain_type(self, file_path: Path) -> str:
        """Determine the domain type of a file based on its path."""
        file_str = str(file_path)

        if "gateway" in file_str:
            return "Gateway"
        elif "services" in file_str:
            return "Service"
        elif "adapters" in file_str:
            return "Adapter"
        elif "orchestration" in file_str or "stages" in file_str:
            return "Orchestration"
        elif "kg" in file_str:
            return "Knowledge Graph"
        elif "storage" in file_str or "vector_store" in file_str:
            return "Storage"
        elif "validation" in file_str:
            return "Validation"
        elif "utils" in file_str:
            return "Utility"
        elif "test" in file_str:
            return "Test"
        else:
            return "Other"

    def _visit_node(self, node: ast.AST, file_path: Path, domain_type: str) -> None:
        """Visit AST nodes and check type hints."""
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            self._check_function(node, file_path, domain_type)

        # Recursively visit child nodes
        for child in ast.iter_child_nodes(node):
            self._visit_node(child, file_path, domain_type)

    def _check_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: Path, domain_type: str
    ) -> None:
        """Check type hints for a function."""
        self.domain_stats[domain_type]["functions"] += 1

        # Check return type annotation
        if node.returns is None:
            self.functions_without_return_types.append(
                (str(file_path), node.lineno, node.name, domain_type)
            )
            self.domain_stats[domain_type]["missing_return_types"] += 1

        # Check parameter type annotations
        for arg in node.args.args:
            if arg.annotation is None:
                self.functions_without_param_types.append(
                    (str(file_path), node.lineno, f"{node.name}({arg.arg})", domain_type)
                )
                self.domain_stats[domain_type]["missing_param_types"] += 1

        # Check for deprecated Optional usage
        deprecated_optional_count = self._check_deprecated_optional(node, file_path)
        self.domain_stats[domain_type]["deprecated_optional"] += deprecated_optional_count

        # Check for deprecated dict/list usage
        deprecated_dict_list_count = self._check_deprecated_dict_list(node, file_path)
        self.domain_stats[domain_type]["deprecated_dict_list"] += deprecated_dict_list_count

    def _check_deprecated_optional(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: Path
    ) -> int:
        """Check for deprecated Optional[T] usage."""
        # This is a simplified check - in practice, you'd need to parse the annotation AST
        # For now, we'll check the source code directly
        count = 0
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                if "Optional[" in line and "typing" in line:
                    self.deprecated_optional_usage.append((str(file_path), i, line.strip()))
                    count += 1
        except Exception:
            pass
        return count

    def _check_deprecated_dict_list(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: Path
    ) -> int:
        """Check for deprecated dict/list usage instead of Mapping/Sequence."""
        count = 0
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                if ("dict[" in line or "list[" in line) and "typing" in line:
                    self.deprecated_dict_list_usage.append((str(file_path), i, line.strip()))
                    count += 1
        except Exception:
            pass
        return count

    def generate_report(self) -> str:
        """Generate a comprehensive type hint analysis report."""
        report = []
        report.append("=" * 60)
        report.append("TYPE HINT ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        if self.errors:
            report.append("ERRORS:")
            for error in self.errors:
                report.append(f"  ❌ {error}")
            report.append("")

        report.append(f"Files analyzed: {self.files_analyzed}")
        report.append("")

        # Domain-specific statistics
        if self.domain_stats:
            report.append("DOMAIN-SPECIFIC STATISTICS:")
            for domain, stats in sorted(self.domain_stats.items()):
                if stats["functions"] > 0:
                    report.append(f"  {domain}:")
                    report.append(f"    Files: {stats['files']}")
                    report.append(f"    Functions: {stats['functions']}")
                    report.append(f"    Missing return types: {stats['missing_return_types']}")
                    report.append(f"    Missing param types: {stats['missing_param_types']}")
                    report.append(f"    Deprecated Optional: {stats['deprecated_optional']}")
                    report.append(f"    Deprecated dict/list: {stats['deprecated_dict_list']}")
                    report.append("")

        # Functions without return types
        report.append(
            f"Functions without return type annotations: {len(self.functions_without_return_types)}"
        )
        if self.functions_without_return_types:
            report.append("Top 10 functions missing return types:")
            for file_path, line_num, func_name, domain in self.functions_without_return_types[:10]:
                report.append(f"  📁 {file_path}:{line_num} - {func_name} ({domain})")
            if len(self.functions_without_return_types) > 10:
                report.append(f"  ... and {len(self.functions_without_return_types) - 10} more")
        report.append("")

        # Functions without parameter types
        report.append(
            f"Functions without parameter type annotations: {len(self.functions_without_param_types)}"
        )
        if self.functions_without_param_types:
            report.append("Top 10 functions missing parameter types:")
            for file_path, line_num, func_sig, domain in self.functions_without_param_types[:10]:
                report.append(f"  📁 {file_path}:{line_num} - {func_sig} ({domain})")
            if len(self.functions_without_param_types) > 10:
                report.append(f"  ... and {len(self.functions_without_param_types) - 10} more")
        report.append("")

        # Deprecated Optional usage
        report.append(f"Deprecated Optional[T] usage: {len(self.deprecated_optional_usage)}")
        if self.deprecated_optional_usage:
            report.append("Files using deprecated Optional syntax:")
            for file_path, line_num, line_content in self.deprecated_optional_usage[:10]:
                report.append(f"  📁 {file_path}:{line_num} - {line_content}")
            if len(self.deprecated_optional_usage) > 10:
                report.append(f"  ... and {len(self.deprecated_optional_usage) - 10} more")
        report.append("")

        # Deprecated dict/list usage
        report.append(f"Deprecated dict/list usage: {len(self.deprecated_dict_list_usage)}")
        if self.deprecated_dict_list_usage:
            report.append("Files using deprecated dict/list syntax:")
            for file_path, line_num, line_content in self.deprecated_dict_list_usage[:10]:
                report.append(f"  📁 {file_path}:{line_num} - {line_content}")
            if len(self.deprecated_dict_list_usage) > 10:
                report.append(f"  ... and {len(self.deprecated_dict_list_usage) - 10} more")
        report.append("")

        # Summary
        total_issues = (
            len(self.functions_without_return_types)
            + len(self.functions_without_param_types)
            + len(self.deprecated_optional_usage)
            + len(self.deprecated_dict_list_usage)
        )

        report.append("SUMMARY:")
        report.append(f"  Total type hint issues: {total_issues}")
        report.append(f"  Files analyzed: {self.files_analyzed}")
        report.append(
            f"  Average issues per file: {total_issues / max(self.files_analyzed, 1):.1f}"
        )

        return "\n".join(report)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check type hint coverage")
    parser.add_argument(
        "path",
        nargs="?",
        default="src/Medical_KG_rev/",
        help="Path to analyze (default: src/Medical_KG_rev/)",
    )
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path {path} does not exist")
        sys.exit(1)

    checker = TypeHintChecker()

    # Find all Python files
    python_files = []
    if path.is_file() and path.suffix == ".py":
        python_files = [path]
    else:
        python_files = list(path.rglob("*.py"))

    print(f"Analyzing {len(python_files)} Python files...")

    for file_path in python_files:
        checker.analyze_file(file_path)

    # Generate and output report
    report = checker.generate_report()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Exit with error code if issues found
    total_issues = (
        len(checker.functions_without_return_types)
        + len(checker.functions_without_param_types)
        + len(checker.deprecated_optional_usage)
        + len(checker.deprecated_dict_list_usage)
    )

    if total_issues > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
