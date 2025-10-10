#!/usr/bin/env python3
"""Import organizer for Medical_KG_rev repository.

This script organizes imports according to the established standards:
- Group imports by category (stdlib, third-party, first-party, relative)
- Sort imports alphabetically within groups
- Remove duplicate imports
- Add proper spacing between groups

Usage:
    python scripts/organize_imports.py [--dry-run] [path]
    python scripts/organize_imports.py --fix [path]
"""

import argparse
import ast
import sys
from pathlib import Path


class ImportOrganizer:
    """Organizes imports according to repository standards."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.files_processed = 0
        self.files_modified = 0
        self.errors = []

    def organize_file(self, file_path: Path) -> bool:
        """Organize imports in a single Python file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse the file to understand its structure
            tree = ast.parse(content, filename=str(file_path))

            # Extract existing imports
            imports = self._extract_imports(content)

            # Organize imports
            organized_imports = self._organize_imports(imports)

            # Check if changes are needed
            if self._imports_changed(imports, organized_imports):
                if not self.dry_run:
                    new_content = self._replace_imports(content, imports, organized_imports)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    self.files_modified += 1
                else:
                    print(f"Would modify: {file_path}")
                self.files_processed += 1
                return True
            else:
                return False

        except SyntaxError as e:
            self.errors.append(f"Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error processing {file_path}: {e}")
            return False

    def _extract_imports(self, content: str) -> list[tuple[int, str, str]]:
        """Extract imports from file content."""
        imports = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith(("import ", "from ")):
                imports.append((i, line, "import"))
            elif line.startswith("#") and "import" in line:
                imports.append((i, line, "comment"))

        return imports

    def _organize_imports(self, imports: list[tuple[int, str, str]]) -> list[str]:
        """Organize imports into proper groups."""
        stdlib_imports = []
        third_party_imports = []
        first_party_imports = []
        relative_imports = []

        # Known standard library modules
        stdlib_modules = {
            "os",
            "sys",
            "pathlib",
            "typing",
            "collections",
            "collections.abc",
            "dataclasses",
            "enum",
            "functools",
            "itertools",
            "operator",
            "re",
            "json",
            "logging",
            "asyncio",
            "threading",
            "multiprocessing",
            "subprocess",
            "shutil",
            "tempfile",
            "io",
            "contextlib",
            "abc",
            "copy",
            "pickle",
            "time",
            "datetime",
            "calendar",
            "math",
            "random",
            "statistics",
            "decimal",
            "fractions",
            "urllib",
            "http",
            "email",
            "html",
            "xml",
            "csv",
            "sqlite3",
            "hashlib",
            "hmac",
            "secrets",
            "uuid",
            "base64",
            "binascii",
            "struct",
            "array",
            "ctypes",
            "weakref",
            "gc",
            "traceback",
            "warnings",
            "inspect",
            "ast",
            "dis",
            "pickletools",
            "keyword",
            "token",
            "tokenize",
            "symbol",
            "parser",
            "codeop",
            "py_compile",
            "compileall",
            "site",
            "sysconfig",
            "platform",
            "getopt",
            "argparse",
            "optparse",
            "getpass",
            "curses",
            "tty",
            "termios",
            "pwd",
            "grp",
            "crypt",
            "spwd",
            "nis",
            "syslog",
            "resource",
            "pipes",
            "signal",
            "mmap",
            "select",
            "selectors",
            "socket",
            "ssl",
            "socketserver",
            "xmlrpc",
            "http.server",
            "urllib.request",
            "urllib.parse",
            "urllib.error",
            "urllib.robotparser",
            "ftplib",
            "poplib",
            "imaplib",
            "nntplib",
            "smtplib",
            "smtpd",
            "telnetlib",
            "wsgiref",
            "http.cookies",
            "http.cookiejar",
            "xml.etree",
            "xml.dom",
            "xml.sax",
            "html.parser",
            "html.entities",
            "xmlrpc.client",
            "xmlrpc.server",
            "wsgiref.util",
            "wsgiref.headers",
            "wsgiref.simple_server",
            "wsgiref.validate",
            "wsgiref.handlers",
            "wsgiref.middleware",
            "wsgiref.types",
        }

        for line_num, line, line_type in imports:
            if line_type == "comment":
                continue

            # Determine import category
            if line.startswith("from .") or line.startswith("import ."):
                relative_imports.append(line)
            elif line.startswith("from Medical_KG_rev") or line.startswith("import Medical_KG_rev"):
                first_party_imports.append(line)
            else:
                # Extract module name
                if line.startswith("from "):
                    module = line.split()[1].split(".")[0]
                else:
                    module = line.split()[1].split(".")[0]

                if module in stdlib_modules:
                    stdlib_imports.append(line)
                else:
                    third_party_imports.append(line)

        # Sort imports within each group
        stdlib_imports.sort()
        third_party_imports.sort()
        first_party_imports.sort()
        relative_imports.sort()

        # Combine organized imports
        organized = []
        if stdlib_imports:
            organized.extend(stdlib_imports)
            organized.append("")  # Blank line after group

        if third_party_imports:
            organized.extend(third_party_imports)
            organized.append("")  # Blank line after group

        if first_party_imports:
            organized.extend(first_party_imports)
            organized.append("")  # Blank line after group

        if relative_imports:
            organized.extend(relative_imports)
            organized.append("")  # Blank line after group

        return organized

    def _imports_changed(self, original: list[tuple[int, str, str]], organized: list[str]) -> bool:
        """Check if imports have changed."""
        original_lines = [line for _, line, _ in original if not line.startswith("#")]
        return original_lines != organized

    def _replace_imports(
        self,
        content: str,
        original_imports: list[tuple[int, str, str]],
        organized_imports: list[str],
    ) -> str:
        """Replace imports in file content."""
        lines = content.split("\n")

        # Find the range of import lines
        if not original_imports:
            return content

        start_line = min(line_num for line_num, _, _ in original_imports)
        end_line = max(line_num for line_num, _, _ in original_imports)

        # Replace the import section
        new_lines = lines[:start_line]
        new_lines.extend(organized_imports)
        new_lines.extend(lines[end_line + 1 :])

        return "\n".join(new_lines)

    def organize_directory(self, directory: Path) -> int:
        """Organize imports in all Python files in a directory."""
        if not directory.exists():
            self.errors.append(f"Directory does not exist: {directory}")
            return 0

        python_files = list(directory.rglob("*.py"))
        modified_count = 0

        for file_path in python_files:
            # Skip certain files
            if any(
                pattern in str(file_path) for pattern in ["__pycache__", ".git", ".pytest_cache"]
            ):
                continue

            if self.organize_file(file_path):
                modified_count += 1

        return modified_count

    def generate_report(self) -> str:
        """Generate a summary report."""
        report = []
        report.append("=" * 60)
        report.append("IMPORT ORGANIZATION REPORT")
        report.append("=" * 60)

        if self.errors:
            report.append(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                report.append(f"  ❌ {error}")

        report.append("\nSUMMARY:")
        report.append(f"  Files processed: {self.files_processed}")
        report.append(f"  Files modified: {self.files_modified}")
        report.append(f"  Errors: {len(self.errors)}")

        if self.dry_run:
            report.append("  Mode: DRY RUN (no files modified)")
        else:
            report.append("  Mode: LIVE (files modified)")

        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Organize imports in Python files")
    parser.add_argument(
        "path",
        nargs="?",
        default="src/Medical_KG_rev/",
        help="Path to organize (default: src/Medical_KG_rev/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Fix import organization (opposite of --dry-run)"
    )

    args = parser.parse_args()

    # Determine mode
    dry_run = args.dry_run and not args.fix

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path {path} does not exist")
        sys.exit(1)

    organizer = ImportOrganizer(dry_run=dry_run)

    if path.is_file() and path.suffix == ".py":
        organizer.organize_file(path)
    else:
        organizer.organize_directory(path)

    # Generate and print report
    report = organizer.generate_report()
    print(report)

    # Exit with error code if errors found
    sys.exit(1 if organizer.errors else 0)


if __name__ == "__main__":
    main()
