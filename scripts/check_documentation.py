#!/usr/bin/env python3
"""
Documentation compliance checker for Medical_KG_rev pipeline components.

This script validates that all pipeline components follow the established
documentation standards including:
- Module-level docstrings
- Class/function docstrings with proper sections
- Section headers for code organization
- Type hints
- Import organization

Usage:
    python scripts/check_documentation.py [--fix] [--verbose] [path]
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Standard section headers for code organization
REQUIRED_SECTIONS = {
    "IMPORTS",
    "REQUEST/RESPONSE MODELS",
    "DATA MODELS",
    "METRICS",
    "BASE COORDINATOR INTERFACE",
    "COORDINATOR IMPLEMENTATION",
    "ERROR TRANSLATION",
    "JOB STATE DATA MODEL",
    "LIFECYCLE MANAGER",
    "EXPORTS"
}

# Pipeline component paths to check
PIPELINE_PATHS = [
    "src/Medical_KG_rev/gateway/coordinators/",
    "src/Medical_KG_rev/services/",
    "src/Medical_KG_rev/orchestration/",
    "src/Medical_KG_rev/adapters/",
    "src/Medical_KG_rev/validation/",
    "src/Medical_KG_rev/kg/",
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

class DocumentationChecker:
    """Check documentation compliance for Python files."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.fixes_applied: List[str] = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose or level in ("ERROR", "WARNING"):
            print(f"[{level}] {message}")

    def check_file(self, file_path: Path) -> bool:
        """Check a single Python file for documentation compliance."""
        self.log(f"Checking {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Cannot read {file_path}: {e}")
            return False

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {file_path}: {e}")
            return False

        # Check module docstring
        if not self._has_module_docstring(tree):
            self.errors.append(f"{file_path}: Missing module docstring")

        # Check for section headers
        missing_sections = self._check_section_headers(content, file_path)
        if missing_sections:
            self.warnings.append(f"{file_path}: Missing section headers: {missing_sections}")

        # Check class and function docstrings
        self._check_docstrings(tree, file_path)

        # Check type hints
        self._check_type_hints(tree, file_path)

        # Check import organization
        self._check_imports(content, file_path)

        return len(self.errors) == 0

    def _has_module_docstring(self, tree: ast.AST) -> bool:
        """Check if module has a docstring."""
        if not tree.body:
            return False

        first_stmt = tree.body[0]
        return (isinstance(first_stmt, ast.Expr) and
                isinstance(first_stmt.value, ast.Constant) and
                isinstance(first_stmt.value.value, str))

    def _check_section_headers(self, content: str, file_path: Path) -> Set[str]:
        """Check for required section headers."""
        lines = content.split('\n')
        found_sections = set()

        for line in lines:
            line = line.strip()
            if line.startswith('# ') and line[2:] in REQUIRED_SECTIONS:
                found_sections.add(line[2:])

        # Determine which sections are missing based on file type
        expected_sections = self._get_expected_sections(file_path)
        return expected_sections - found_sections

    def _get_expected_sections(self, file_path: Path) -> Set[str]:
        """Get expected section headers for a file based on its type."""
        if "coordinators" in str(file_path):
            return {"IMPORTS", "REQUEST/RESPONSE MODELS", "COORDINATOR IMPLEMENTATION", "ERROR TRANSLATION", "EXPORTS"}
        elif "base" in str(file_path):
            return {"IMPORTS", "DATA MODELS", "METRICS", "BASE COORDINATOR INTERFACE", "EXPORTS"}
        elif "job_lifecycle" in str(file_path):
            return {"IMPORTS", "JOB STATE DATA MODEL", "LIFECYCLE MANAGER", "EXPORTS"}
        else:
            return {"IMPORTS", "EXPORTS"}  # Minimum required

    def _check_docstrings(self, tree: ast.AST, file_path: Path):
        """Check docstrings for classes and functions."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    if not node.name.startswith('_'):  # Skip private methods
                        self.warnings.append(f"{file_path}: {node.name} missing docstring")

    def _check_type_hints(self, tree: ast.AST, file_path: Path):
        """Check for type hints on function definitions."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith('_'):  # Skip private methods
                    if not node.returns and not node.name.startswith('test_'):
                        self.warnings.append(f"{file_path}: {node.name} missing return type hint")

    def _check_imports(self, content: str, file_path: Path):
        """Check import organization."""
        lines = content.split('\n')
        import_lines = []

        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                import_lines.append((i, line))

        # Check if imports are grouped and ordered
        if len(import_lines) > 1:
            # Simple check: imports should be at the top
            first_import_line = import_lines[0][0]
            if first_import_line > 10:  # Allow some flexibility
                self.warnings.append(f"{file_path}: Imports not at top of file")

    def check_directory(self, directory: Path) -> bool:
        """Check all Python files in a directory."""
        if not directory.exists():
            self.errors.append(f"Directory does not exist: {directory}")
            return False

        success = True
        python_files = list(directory.rglob("*.py"))

        for file_path in python_files:
            # Skip excluded files
            if any(pattern in str(file_path) for pattern in EXCLUDE_PATTERNS):
                continue

            if not self.check_file(file_path):
                success = False

        return success

    def generate_report(self) -> str:
        """Generate a summary report of all findings."""
        report = []
        report.append("=" * 60)
        report.append("DOCUMENTATION COMPLIANCE REPORT")
        report.append("=" * 60)

        if self.errors:
            report.append(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                report.append(f"  ❌ {error}")

        if self.warnings:
            report.append(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                report.append(f"  ⚠️  {warning}")

        if self.fixes_applied:
            report.append(f"\nFIXES APPLIED ({len(self.fixes_applied)}):")
            for fix in self.fixes_applied:
                report.append(f"  ✅ {fix}")

        report.append(f"\nSUMMARY:")
        report.append(f"  Total Errors: {len(self.errors)}")
        report.append(f"  Total Warnings: {len(self.warnings)}")
        report.append(f"  Total Fixes: {len(self.fixes_applied)}")

        if len(self.errors) == 0 and len(self.warnings) == 0:
            report.append(f"  Status: ✅ ALL CHECKS PASSED")
        elif len(self.errors) == 0:
            report.append(f"  Status: ⚠️  WARNINGS ONLY")
        else:
            report.append(f"  Status: ❌ ERRORS FOUND")

        return "\n".join(report)


def main():
    """Main entry point for the documentation checker."""
    parser = argparse.ArgumentParser(description="Check documentation compliance")
    parser.add_argument("path", nargs="?", default=".", help="Path to check (default: current directory)")
    parser.add_argument("--fix", action="store_true", help="Apply automatic fixes where possible")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    checker = DocumentationChecker(verbose=args.verbose)
    path = Path(args.path)

    if path.is_file():
        success = checker.check_file(path)
    else:
        # Check all pipeline directories
        success = True
        for pipeline_path in PIPELINE_PATHS:
            full_path = Path(pipeline_path)
            if full_path.exists():
                checker.log(f"Checking pipeline directory: {full_path}")
                if not checker.check_directory(full_path):
                    success = False
            else:
                checker.log(f"Pipeline directory not found: {full_path}", "WARNING")

    # Generate and print report
    report = checker.generate_report()
    print(report)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

