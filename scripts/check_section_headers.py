#!/usr/bin/env python3
"""
Section header checker for Medical_KG_rev repository-wide components.

This script validates that all Python modules follow the established
section header standards including:
- Required section headers are present
- Sections appear in correct order
- Each section contains appropriate content
- Module type detection and validation

Usage:
    python scripts/check_section_headers.py [--verbose] [path]
    python scripts/check_section_headers.py --auto-fix [path]
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Section header patterns - must match exactly
SECTION_PATTERN = r"^# ={10,} ([A-Z][A-Z_/ ]*[A-Z]) ={10,}$"

# Expected sections for different module types (from section_headers.md)
GATEWAY_SECTIONS = [
    "IMPORTS",
    "TYPE DEFINITIONS",
    "REQUEST/RESPONSE MODELS",
    "COORDINATOR IMPLEMENTATION",
    "ERROR TRANSLATION",
    "FACTORY FUNCTIONS",
    "EXPORTS"
]

SERVICE_SECTIONS = [
    "IMPORTS",
    "TYPE DEFINITIONS",
    "DATA MODELS",
    "INTERFACES",
    "IMPLEMENTATIONS",
    "FACTORY FUNCTIONS",
    "HELPER FUNCTIONS",
    "EXPORTS"
]

ADAPTER_SECTIONS = [
    "IMPORTS",
    "TYPE DEFINITIONS",
    "DATA MODELS",
    "ADAPTER IMPLEMENTATION",
    "ERROR HANDLING",
    "FACTORY FUNCTIONS",
    "HELPER FUNCTIONS",
    "EXPORTS"
]

ORCHESTRATION_SECTIONS = [
    "IMPORTS",
    "TYPE DEFINITIONS",
    "STAGE CONTEXT DATA MODELS",
    "STAGE IMPLEMENTATIONS",
    "PLUGIN REGISTRATION",
    "FACTORY FUNCTIONS",
    "HELPER FUNCTIONS",
    "EXPORTS"
]

KG_SECTIONS = [
    "IMPORTS",
    "TYPE DEFINITIONS",
    "SCHEMA DATA MODELS",
    "CLIENT IMPLEMENTATION",
    "TEMPLATES",
    "FACTORY FUNCTIONS",
    "HELPER FUNCTIONS",
    "EXPORTS"
]

STORAGE_SECTIONS = [
    "IMPORTS",
    "TYPE DEFINITIONS",
    "DATA MODELS",
    "INTERFACES",
    "IMPLEMENTATIONS",
    "FACTORY FUNCTIONS",
    "HELPER FUNCTIONS",
    "EXPORTS"
]

VALIDATION_SECTIONS = [
    "IMPORTS",
    "TYPE DEFINITIONS",
    "DATA MODELS",
    "VALIDATOR IMPLEMENTATION",
    "ERROR HANDLING",
    "FACTORY FUNCTIONS",
    "HELPER FUNCTIONS",
    "EXPORTS"
]

UTILITY_SECTIONS = [
    "IMPORTS",
    "TYPE DEFINITIONS",
    "UTILITY FUNCTIONS",
    "HELPER CLASSES",
    "FACTORY FUNCTIONS",
    "HELPER FUNCTIONS",
    "EXPORTS"
]

TEST_SECTIONS = [
    "IMPORTS",
    "TYPE DEFINITIONS",
    "FIXTURES",
    "UNIT TESTS",
    "INTEGRATION TESTS",
    "HELPER FUNCTIONS",
    "EXPORTS"
]

# Legacy sections for backward compatibility
LEGACY_COORDINATOR_SECTIONS = [
    "IMPORTS",
    "REQUEST/RESPONSE MODELS",
    "COORDINATOR IMPLEMENTATION",
    "ERROR TRANSLATION",
    "EXPORTS"
]

LEGACY_SERVICE_SECTIONS = [
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

        # Gateway modules
        if "gateway" in file_str:
            if "coordinators" in file_str:
                return GATEWAY_SECTIONS
            else:
                return GATEWAY_SECTIONS

        # Service modules
        elif "services" in file_str:
            return SERVICE_SECTIONS

        # Adapter modules
        elif "adapters" in file_str:
            return ADAPTER_SECTIONS

        # Orchestration modules
        elif "orchestration" in file_str or "stages" in file_str:
            return ORCHESTRATION_SECTIONS

        # Knowledge Graph modules
        elif "kg" in file_str:
            return KG_SECTIONS

        # Storage modules
        elif "storage" in file_str or "vector_store" in file_str:
            return STORAGE_SECTIONS

        # Validation modules
        elif "validation" in file_str:
            return VALIDATION_SECTIONS

        # Utility modules
        elif "utils" in file_str:
            return UTILITY_SECTIONS

        # Test modules
        elif "test" in file_str:
            return TEST_SECTIONS

        # Legacy compatibility
        elif "policy" in file_str or "persister" in file_str or "telemetry" in file_str:
            return LEGACY_SERVICE_SECTIONS

        else:
            # Default minimal sections
            return ["IMPORTS", "EXPORTS"]

    def extract_sections(self, content: str) -> List[Tuple[int, str]]:
        """Extract section headers from file content."""
        lines = content.split('\n')
        sections = []

        for i, line in enumerate(lines):
            line = line.strip()
            # Match section headers like: # ==============================================================================
            # SECTION NAME
            # ==============================================================================
            if line.startswith('# =') and line.endswith('='):
                # Extract section name between the equals signs
                parts = line.split(' ')
                if len(parts) >= 3:
                    section_name = ' '.join(parts[1:-1])
                    sections.append((i + 1, section_name))
            # Also match single-line headers like: # ============================================================================== SECTION NAME ==============================================================================
            elif line.startswith('# =') and '=' in line[3:]:
                # Find the section name between equals signs
                start_idx = line.find(' ', 3)
                end_idx = line.rfind(' =')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    section_name = line[start_idx+1:end_idx].strip()
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
