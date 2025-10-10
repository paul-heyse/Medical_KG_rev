#!/usr/bin/env python3
"""Docstring coverage checker for Medical_KG_rev repository-wide components.

This script calculates docstring coverage for Python files and fails if
coverage falls below the minimum threshold. Provides detailed reporting
by module type and domain.

Usage:
    python scripts/check_docstring_coverage.py [--min-coverage 90] [--verbose] [path]
    python scripts/check_docstring_coverage.py --report-html [path]
"""

import argparse
import ast
import sys
from pathlib import Path

# Repository-wide paths to check
REPOSITORY_PATHS = [
    "src/Medical_KG_rev/gateway/",
    "src/Medical_KG_rev/services/",
    "src/Medical_KG_rev/adapters/",
    "src/Medical_KG_rev/orchestration/",
    "src/Medical_KG_rev/kg/",
    "src/Medical_KG_rev/storage/",
    "src/Medical_KG_rev/validation/",
    "src/Medical_KG_rev/utils/",
    "tests/",
]

# Files to exclude from checks
EXCLUDE_PATTERNS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "build",
    "dist",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "__init__.py",  # Often minimal, check separately
}


class DocstringCoverageChecker:
    """Check docstring coverage for Python files."""

    def __init__(self, verbose: bool = False, min_coverage: float = 90.0):
        self.verbose = verbose
        self.min_coverage = min_coverage
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.coverage_data: dict[str, dict[str, any]] = {}
        self.domain_coverage: dict[str, dict[str, any]] = {}

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose or level in ("ERROR", "WARNING"):
            print(f"[{level}] {message}")

    def get_domain_type(self, file_path: Path) -> str:
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

    def has_docstring(self, node: ast.AST) -> bool:
        """Check if an AST node has a docstring."""
        if not hasattr(node, "body") or not node.body:
            return False

        first_stmt = node.body[0]
        return (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
        )

    def should_check_node(self, node: ast.AST) -> bool:
        """Determine if a node should be checked for docstring coverage."""
        # Skip private methods/functions (starting with _)
        if hasattr(node, "name") and node.name.startswith("_"):
            return False

        # Skip test functions
        if hasattr(node, "name") and node.name.startswith("test_"):
            return False

        # Skip property setters/getters
        if isinstance(node, ast.FunctionDef) and node.name.startswith(("get_", "set_")):
            return False

        return True

    def check_file(self, file_path: Path) -> tuple[int, int, list[str]]:
        """Check docstring coverage for a single Python file."""
        self.log(f"Checking {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Cannot read {file_path}: {e}")
            return 0, 0, []

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {file_path}: {e}")
            return 0, 0, []

        total_items = 0
        documented_items = 0
        missing_docstrings = []

        # Check module docstring
        if tree.body:
            total_items += 1
            if self.has_docstring(tree):
                documented_items += 1
            else:
                missing_docstrings.append(f"{file_path}:1:module")

        # Walk through all nodes in the AST
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if self.should_check_node(node):
                    total_items += 1
                    if self.has_docstring(node):
                        documented_items += 1
                    else:
                        line_num = getattr(node, "lineno", "?")
                        node_type = "class" if isinstance(node, ast.ClassDef) else "function"
                        missing_docstrings.append(f"{file_path}:{line_num}:{node_type}:{node.name}")

        return total_items, documented_items, missing_docstrings

    def check_directory(self, directory: Path) -> bool:
        """Check docstring coverage for all Python files in a directory."""
        if not directory.exists():
            self.errors.append(f"Directory does not exist: {directory}")
            return False

        success = True
        python_files = list(directory.rglob("*.py"))

        total_items = 0
        total_documented = 0
        all_missing = []

        for file_path in python_files:
            # Skip excluded files
            if any(pattern in str(file_path) for pattern in EXCLUDE_PATTERNS):
                continue

            file_total, file_documented, file_missing = self.check_file(file_path)
            total_items += file_total
            total_documented += file_documented
            all_missing.extend(file_missing)

            # Calculate coverage for this file
            if file_total > 0:
                file_coverage = (file_documented / file_total) * 100
                domain_type = self.get_domain_type(file_path)

                self.coverage_data[str(file_path)] = {
                    "total": file_total,
                    "documented": file_documented,
                    "coverage": file_coverage,
                    "missing": file_missing,
                    "domain": domain_type,
                }

                # Track domain coverage
                if domain_type not in self.domain_coverage:
                    self.domain_coverage[domain_type] = {"total": 0, "documented": 0, "files": 0}

                self.domain_coverage[domain_type]["total"] += file_total
                self.domain_coverage[domain_type]["documented"] += file_documented
                self.domain_coverage[domain_type]["files"] += 1

                if file_coverage < self.min_coverage:
                    self.errors.append(
                        f"{file_path}: {file_coverage:.1f}% coverage "
                        f"({file_documented}/{file_total} items documented) "
                        f"below minimum {self.min_coverage}%"
                    )
                    success = False

        # Store overall coverage data
        if total_items > 0:
            overall_coverage = (total_documented / total_items) * 100
            self.coverage_data["OVERALL"] = {
                "total": total_items,
                "documented": total_documented,
                "coverage": overall_coverage,
                "missing": all_missing,
            }

            if overall_coverage < self.min_coverage:
                self.errors.append(
                    f"Overall coverage {overall_coverage:.1f}% "
                    f"({total_documented}/{total_items} items documented) "
                    f"below minimum {self.min_coverage}%"
                )
                success = False

        return success

    def generate_report(self) -> str:
        """Generate a summary report of docstring coverage."""
        report = []
        report.append("=" * 60)
        report.append("DOCSTRING COVERAGE REPORT")
        report.append("=" * 60)

        if self.coverage_data:
            # Show overall coverage
            if "OVERALL" in self.coverage_data:
                overall = self.coverage_data["OVERALL"]
                report.append(
                    f"\nOVERALL COVERAGE: {overall['coverage']:.1f}% "
                    f"({overall['documented']}/{overall['total']} items documented)"
                )

            # Show domain coverage
            if self.domain_coverage:
                report.append("\nDOMAIN COVERAGE:")
                for domain, data in sorted(self.domain_coverage.items()):
                    if data["total"] > 0:
                        domain_coverage = (data["documented"] / data["total"]) * 100
                        status = "✅" if domain_coverage >= self.min_coverage else "❌"
                        report.append(
                            f"  {status} {domain}: {domain_coverage:.1f}% "
                            f"({data['documented']}/{data['total']} items, {data['files']} files)"
                        )

            # Show per-file coverage
            report.append("\nPER-FILE COVERAGE:")
            for file_path, data in self.coverage_data.items():
                if file_path != "OVERALL":
                    status = "✅" if data["coverage"] >= self.min_coverage else "❌"
                    report.append(
                        f"  {status} {file_path}: {data['coverage']:.1f}% "
                        f"({data['documented']}/{data['total']} items documented)"
                    )

            # Show missing docstrings
            if "OVERALL" in self.coverage_data:
                missing = self.coverage_data["OVERALL"]["missing"]
                if missing:
                    report.append(f"\nMISSING DOCSTRINGS ({len(missing)} items):")
                    for item in missing[:20]:  # Show first 20
                        report.append(f"  ❌ {item}")
                    if len(missing) > 20:
                        report.append(f"  ... and {len(missing) - 20} more")

        if self.errors:
            report.append(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                report.append(f"  ❌ {error}")

        if self.warnings:
            report.append(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                report.append(f"  ⚠️  {warning}")

        report.append("\nSUMMARY:")
        report.append(f"  Minimum Coverage Required: {self.min_coverage}%")
        report.append(f"  Total Errors: {len(self.errors)}")
        report.append(f"  Total Warnings: {len(self.warnings)}")

        if len(self.errors) == 0:
            report.append("  Status: ✅ ALL CHECKS PASSED")
        else:
            report.append("  Status: ❌ COVERAGE BELOW THRESHOLD")

        return "\n".join(report)


def main():
    """Main entry point for the docstring coverage checker."""
    parser = argparse.ArgumentParser(description="Check docstring coverage")
    parser.add_argument(
        "path", nargs="?", default=".", help="Path to check (default: current directory)"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=90.0,
        help="Minimum coverage percentage (default: 90.0)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    checker = DocstringCoverageChecker(verbose=args.verbose, min_coverage=args.min_coverage)
    path = Path(args.path)

    if path.is_file():
        total, documented, missing = checker.check_file(path)
        if total > 0:
            coverage = (documented / total) * 100
            checker.coverage_data[str(path)] = {
                "total": total,
                "documented": documented,
                "coverage": coverage,
                "missing": missing,
            }
            success = coverage >= args.min_coverage
        else:
            success = True
    else:
        # Check all repository directories
        success = True
        for repo_path in REPOSITORY_PATHS:
            full_path = Path(repo_path)
            if full_path.exists():
                checker.log(f"Checking repository directory: {full_path}")
                if not checker.check_directory(full_path):
                    success = False
            else:
                checker.log(f"Repository directory not found: {full_path}", "WARNING")

    # Generate and print report
    report = checker.generate_report()
    print(report)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
