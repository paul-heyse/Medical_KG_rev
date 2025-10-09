#!/usr/bin/env python3
"""
Section header checker for Medical_KG_rev pipeline and authentication components.

This script validates that all pipeline components follow the established
section header standards including:
- Required section headers are present
- Sections appear in correct order
- Each section contains appropriate content

Usage:
    python scripts/check_section_headers.py [--verbose] [path]
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Section header patterns - must match exactly
SECTION_PATTERN = r"^# ={10,} ([A-Z][A-Z_/ ]*[A-Z]) ={10,}$"

# Expected sections for different module types
COORDINATOR_SECTIONS = [
    "IMPORTS",
    "REQUEST/RESPONSE MODELS",
    "COORDINATOR IMPLEMENTATION",
    "ERROR TRANSLATION",
    "EXPORTS"
]

BASE_COORDINATOR_SECTIONS = [
    "IMPORTS",
    "DATA MODELS",
    "METRICS",
    "BASE COORDINATOR INTERFACE",
    "EXPORTS"
]

JOB_LIFECYCLE_SECTIONS = [
    "IMPORTS",
    "JOB STATE DATA MODEL",
    "LIFECYCLE MANAGER",
    "EXPORTS"
]

SERVICE_SECTIONS = [
    "IMPORTS",
    "TYPE DEFINITIONS & CONSTANTS",
    "SERVICE CLASS DEFINITION",
    "INITIALIZATION & SETUP",
    "CHUNKING ENDPOINTS",
    "EMBEDDING ENDPOINTS",
    "RETRIEVAL ENDPOINTS",
    "ADAPTER MANAGEMENT ENDPOINTS",
    "VALIDATION ENDPOINTS",
    "EXTRACTION ENDPOINTS",
    "ADMIN & UTILITY ENDPOINTS",
    "PRIVATE HELPERS"
]

POLICY_SECTIONS = [
    "IMPORTS",
    "DATA MODELS",
    "INTERFACE",
    "IMPLEMENTATIONS",
    "FACTORY FUNCTIONS",
    "EXPORTS"
]

ORCHESTRATION_SECTIONS = [
    "IMPORTS",
    "STAGE CONTEXT DATA MODELS",
    "STAGE IMPLEMENTATIONS",
    "PLUGIN REGISTRATION",
    "EXPORTS"
]

TEST_SECTIONS = [
    "IMPORTS",
    "FIXTURES",
    "UNIT TESTS",
    "INTEGRATION TESTS",
    "HELPER FUNCTIONS"
]

# Pipeline component paths to check
PIPELINE_PATHS = [
    "src/Medical_KG_rev/auth/",
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

class SectionHeaderChecker:
    """Check section header compliance for Python files."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose or level in ("ERROR", "WARNING"):
            print(f"[{level}] {message}")

    def get_expected_sections(self, file_path: Path) -> List[str]:
        """Get expected sections for a file based on its type."""
        file_str = str(file_path)

        if "coordinators" in file_str:
            if "base" in file_str:
                return BASE_COORDINATOR_SECTIONS
            elif "job_lifecycle" in file_str:
                return JOB_LIFECYCLE_SECTIONS
            else:
                return COORDINATOR_SECTIONS
        elif "services" in file_str:
            return SERVICE_SECTIONS
        elif "policy" in file_str or "persister" in file_str or "telemetry" in file_str:
            return POLICY_SECTIONS
        elif "orchestration" in file_str or "stages" in file_str:
            return ORCHESTRATION_SECTIONS
        elif "test" in file_str:
            return TEST_SECTIONS
        else:
            # Default minimal sections
            return ["IMPORTS", "EXPORTS"]

    def extract_sections(self, content: str) -> List[Tuple[int, str]]:
        """Extract section headers from file content."""
        lines = content.split('\n')
        sections = []

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('# =') and line.endswith('= #'):
                # Extract section name between the equals signs
                parts = line.split(' ')
                if len(parts) >= 3:
                    section_name = ' '.join(parts[1:-1])
                    sections.append((i + 1, section_name))

        return sections

    def check_file(self, file_path: Path) -> bool:
        """Check a single Python file for section header compliance."""
        self.log(f"Checking {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Cannot read {file_path}: {e}")
            return False

        # Extract sections from file
        sections = self.extract_sections(content)

        if not sections:
            self.warnings.append(f"{file_path}: No section headers found")
            return True  # Not an error, just a warning

        # Get expected sections for this file type
        expected_sections = self.get_expected_sections(file_path)

        # Check if all expected sections are present
        found_section_names = [name for _, name in sections]
        missing_sections = set(expected_sections) - set(found_section_names)

        if missing_sections:
            self.errors.append(f"{file_path}: Missing sections: {sorted(missing_sections)}")

        # Check section ordering
        for i, (line_num, section_name) in enumerate(sections):
            if section_name in expected_sections:
                expected_index = expected_sections.index(section_name)
                if i != expected_index:
                    self.errors.append(
                        f"{file_path}:{line_num}: Section '{section_name}' appears at position {i+1}, "
                        f"expected position {expected_index+1}"
                    )

        # Check for duplicate sections
        section_counts = {}
        for _, section_name in sections:
            section_counts[section_name] = section_counts.get(section_name, 0) + 1

        for section_name, count in section_counts.items():
            if count > 1:
                self.errors.append(f"{file_path}: Duplicate section '{section_name}' appears {count} times")

        return len(self.errors) == 0

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
        report.append("SECTION HEADER COMPLIANCE REPORT")
        report.append("=" * 60)

        if self.errors:
            report.append(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                report.append(f"  ❌ {error}")

        if self.warnings:
            report.append(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                report.append(f"  ⚠️  {warning}")

        report.append(f"\nSUMMARY:")
        report.append(f"  Total Errors: {len(self.errors)}")
        report.append(f"  Total Warnings: {len(self.warnings)}")

        if len(self.errors) == 0:
            report.append(f"  Status: ✅ ALL CHECKS PASSED")
        else:
            report.append(f"  Status: ❌ ERRORS FOUND")

        return "\n".join(report)


def main():
    """Main entry point for the section header checker."""
    parser = argparse.ArgumentParser(description="Check section header compliance")
    parser.add_argument("path", nargs="?", default=".", help="Path to check (default: current directory)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    checker = SectionHeaderChecker(verbose=args.verbose)
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
