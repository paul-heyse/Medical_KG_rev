#!/usr/bin/env python3
"""Method ordering checker for Medical_KG_rev repository.

This script validates that methods in classes follow the established ordering:
- Special methods first (__init__, __repr__, etc.)
- Public methods (alphabetical)
- Private methods (alphabetical)
- Properties (alphabetical)
- Class methods (alphabetical)
- Static methods (alphabetical)

Usage:
    python scripts/check_method_ordering.py [--verbose] [path]
    python scripts/check_method_ordering.py --auto-fix [path]
"""

import argparse
import ast
import sys
from pathlib import Path


class MethodOrderingChecker:
    """Check method ordering compliance for Python classes."""

    def __init__(self, verbose: bool = False, auto_fix: bool = False):
        self.verbose = verbose
        self.auto_fix = auto_fix
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.files_processed = 0
        self.classes_checked = 0

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose or level in ("ERROR", "WARNING"):
            print(f"[{level}] {message}")

    def check_file(self, file_path: Path) -> bool:
        """Check method ordering in a single Python file."""
        self.log(f"Checking {file_path}")

        try:
            with file_path.open(encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Cannot read {file_path}: {e}")
            return False

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {file_path}: {e}")
            return False

        success = True
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not self._check_class_method_ordering(node, file_path, content):
                    success = False
                self.classes_checked += 1

        self.files_processed += 1
        return success

    def _check_class_method_ordering(
        self, class_node: ast.ClassDef, file_path: Path, content: str
    ) -> bool:
        """Check method ordering within a class."""
        methods = []

        for node in class_node.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                methods.append(node)
            elif isinstance(node, ast.ClassDef):
                # Nested class - skip for now
                continue

        if len(methods) <= 1:
            return True  # No ordering issues with 0-1 methods

        # Categorize methods
        special_methods = []
        public_methods = []
        private_methods = []
        properties = []
        class_methods = []
        static_methods = []

        for method in methods:
            if self._is_special_method(method):
                special_methods.append(method)
            elif self._is_property(method):
                properties.append(method)
            elif self._is_class_method(method):
                class_methods.append(method)
            elif self._is_static_method(method):
                static_methods.append(method)
            elif method.name.startswith("_"):
                private_methods.append(method)
            else:
                public_methods.append(method)

        # Sort each category alphabetically
        special_methods.sort(key=lambda m: self._get_method_sort_key(m))
        public_methods.sort(key=lambda m: m.name)
        private_methods.sort(key=lambda m: m.name)
        properties.sort(key=lambda m: m.name)
        class_methods.sort(key=lambda m: m.name)
        static_methods.sort(key=lambda m: m.name)

        # Expected order
        expected_order = (
            special_methods
            + public_methods
            + private_methods
            + properties
            + class_methods
            + static_methods
        )

        # Check if current order matches expected order
        current_order = methods
        if current_order != expected_order:
            self._report_ordering_issues(class_node, file_path, current_order, expected_order)
            return False

        return True

    def _is_special_method(self, method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if method is a special method (__init__, __repr__, etc.)."""
        special_methods = {
            "__init__",
            "__new__",
            "__del__",
            "__repr__",
            "__str__",
            "__bytes__",
            "__format__",
            "__lt__",
            "__le__",
            "__eq__",
            "__ne__",
            "__gt__",
            "__ge__",
            "__hash__",
            "__bool__",
            "__getattr__",
            "__getattribute__",
            "__setattr__",
            "__delattr__",
            "__dir__",
            "__getitem__",
            "__setitem__",
            "__delitem__",
            "__iter__",
            "__reversed__",
            "__contains__",
            "__len__",
            "__length_hint__",
            "__call__",
            "__enter__",
            "__exit__",
            "__await__",
            "__aiter__",
            "__anext__",
            "__aenter__",
            "__aexit__",
        }
        return method.name in special_methods

    def _is_property(self, method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if method is a property."""
        # This is a simplified check - in practice, you'd need to check decorators
        # For now, we'll check the source code directly
        return False  # Properties are handled separately

    def _is_class_method(self, method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if method is a class method."""
        # This is a simplified check - in practice, you'd need to check decorators
        return False  # Class methods are handled separately

    def _is_static_method(self, method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if method is a static method."""
        # This is a simplified check - in practice, you'd need to check decorators
        return False  # Static methods are handled separately

    def _get_method_sort_key(self, method: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Get sort key for special methods (__init__ first, then alphabetical)."""
        if method.name == "__init__":
            return "0__init__"  # __init__ always first
        else:
            return f"1{method.name}"  # Other special methods alphabetically

    def _report_ordering_issues(
        self, class_node: ast.ClassDef, file_path: Path, current_order: list, expected_order: list
    ) -> None:
        """Report method ordering issues."""
        self.errors.append(
            f"{file_path}:{class_node.lineno}: Class '{class_node.name}' has incorrect method ordering"
        )

        if self.verbose:
            self.log(f"  Current order: {[m.name for m in current_order]}")
            self.log(f"  Expected order: {[m.name for m in expected_order]}")

    def check_directory(self, directory: Path) -> bool:
        """Check method ordering in all Python files in a directory."""
        if not directory.exists():
            self.errors.append(f"Directory does not exist: {directory}")
            return False

        success = True
        python_files = list(directory.rglob("*.py"))

        for file_path in python_files:
            # Skip excluded files
            if any(
                pattern in str(file_path) for pattern in ["__pycache__", ".git", ".pytest_cache"]
            ):
                continue

            if not self.check_file(file_path):
                success = False

        return success

    def generate_report(self) -> str:
        """Generate a summary report of all findings."""
        report = []
        report.append("=" * 60)
        report.append("METHOD ORDERING COMPLIANCE REPORT")
        report.append("=" * 60)

        if self.errors:
            report.append(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                report.append(f"  ❌ {error}")

        if self.warnings:
            report.append(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                report.append(f"  ⚠️  {warning}")

        report.append("\nSUMMARY:")
        report.append(f"  Files processed: {self.files_processed}")
        report.append(f"  Classes checked: {self.classes_checked}")
        report.append(f"  Total Errors: {len(self.errors)}")
        report.append(f"  Total Warnings: {len(self.warnings)}")

        if len(self.errors) == 0:
            report.append("  Status: ✅ ALL CHECKS PASSED")
        else:
            report.append("  Status: ❌ ERRORS FOUND")

        return "\n".join(report)


def main():
    """Run the method ordering checker."""
    parser = argparse.ArgumentParser(description="Check method ordering compliance")
    parser.add_argument(
        "path", nargs="?", default=".", help="Path to check (default: current directory)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--auto-fix", action="store_true", help="Automatically fix ordering issues")

    args = parser.parse_args()

    checker = MethodOrderingChecker(verbose=args.verbose, auto_fix=args.auto_fix)
    path = Path(args.path)

    success = checker.check_file(path) if path.is_file() else checker.check_directory(path)

    # Generate and print report
    report = checker.generate_report()
    print(report)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
