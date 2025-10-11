#!/usr/bin/env python3
"""CI torch isolation check script.

This script validates that the torch isolation architecture is maintained
and prevents torch imports from being added to the main gateway.
"""

import sys
from pathlib import Path
from typing import Any

from tests.validation.test_torch_isolation_completeness import TorchIsolationValidator


class TorchIsolationQualityGate:
    """Quality gate for torch isolation validation."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.validator = TorchIsolationValidator(project_root)
        self.torch_keywords = {"torch", "pytorch", "transformers", "accelerate", "safetensors"}

    def check_torch_imports(self) -> dict[str, Any]:
        """Check for torch imports in main gateway code."""
        self.validator.scan_for_torch_imports()

        return {
            "passed": len(self.validator.torch_imports) == 0,
            "torch_imports": list(self.validator.torch_imports),
            "message": f"Found {len(self.validator.torch_imports)} torch imports in main gateway",
        }

    def check_torch_usage(self) -> dict[str, Any]:
        """Check for torch usage in main gateway code."""
        self.validator.scan_for_torch_usage()

        return {
            "passed": len(self.validator.torch_usage) == 0,
            "torch_usage": self.validator.torch_usage,
            "message": f"Found {len(self.validator.torch_usage)} torch usage instances in main gateway",
        }

    def check_grpc_clients(self) -> dict[str, Any]:
        """Check that gRPC clients exist for torch functionality."""
        grpc_clients_exist = self.validator.validate_grpc_clients_exist()

        return {
            "passed": grpc_clients_exist,
            "message": (
                "gRPC clients exist for torch functionality"
                if grpc_clients_exist
                else "gRPC clients missing for torch functionality"
            ),
        }

    def check_proto_definitions(self) -> dict[str, Any]:
        """Check that gRPC proto definitions exist."""
        proto_definitions_exist = self.validator.validate_proto_definitions_exist()

        return {
            "passed": proto_definitions_exist,
            "message": (
                "gRPC proto definitions exist"
                if proto_definitions_exist
                else "gRPC proto definitions missing"
            ),
        }

    def check_docker_configurations(self) -> dict[str, Any]:
        """Check that Docker configurations exist."""
        docker_configurations_exist = self.validator.validate_docker_configurations_exist()

        return {
            "passed": docker_configurations_exist,
            "message": (
                "Docker configurations exist"
                if docker_configurations_exist
                else "Docker configurations missing"
            ),
        }

    def check_requirements_torch_free(self) -> dict[str, Any]:
        """Check that requirements.txt is torch-free."""
        requirements_torch_free = self.validator.validate_requirements_torch_free()

        return {
            "passed": requirements_torch_free,
            "message": (
                "requirements.txt is torch-free"
                if requirements_torch_free
                else "requirements.txt contains torch dependencies"
            ),
        }

    def run_all_checks(self) -> dict[str, Any]:
        """Run all quality gate checks."""
        checks = {
            "torch_imports": self.check_torch_imports(),
            "torch_usage": self.check_torch_usage(),
            "grpc_clients": self.check_grpc_clients(),
            "proto_definitions": self.check_proto_definitions(),
            "docker_configurations": self.check_docker_configurations(),
            "requirements_torch_free": self.check_requirements_torch_free(),
        }

        all_passed = all(check["passed"] for check in checks.values())

        return {
            "all_passed": all_passed,
            "checks": checks,
            "summary": {
                "total_checks": len(checks),
                "passed_checks": sum(1 for check in checks.values() if check["passed"]),
                "failed_checks": sum(1 for check in checks.values() if not check["passed"]),
            },
        }


def main() -> None:
    """Main entry point for CI torch isolation check."""
    project_root = Path(__file__).parent.parent

    print("🔍 Running Torch Isolation Quality Gate...")
    print("=" * 60)

    quality_gate = TorchIsolationQualityGate(project_root)
    result = quality_gate.run_all_checks()

    print("\n📊 Quality Gate Results:")
    print(f"   Total Checks: {result['summary']['total_checks']}")
    print(f"   Passed: {result['summary']['passed_checks']}")
    print(f"   Failed: {result['summary']['failed_checks']}")

    print("\n🔍 Individual Check Results:")
    for check_name, check_result in result["checks"].items():
        status = "✅ PASS" if check_result["passed"] else "❌ FAIL"
        print(f"   {check_name}: {status}")
        if not check_result["passed"]:
            print(f"      {check_result['message']}")

    if result["checks"]["torch_imports"]["torch_imports"]:
        print("\n🚨 Torch Imports Found:")
        for import_line in result["checks"]["torch_imports"]["torch_imports"]:
            print(f"   - {import_line}")

    if result["checks"]["torch_usage"]["torch_usage"]:
        print("\n🚨 Torch Usage Found:")
        for file_path, line_num, line_content in result["checks"]["torch_usage"]["torch_usage"]:
            print(f"   - {file_path}:{line_num}: {line_content}")

    print("\n" + "=" * 60)

    if result["all_passed"]:
        print("✅ TORCH ISOLATION QUALITY GATE PASSED!")
        print("\nAll quality gate checks have passed successfully.")
        print("The torch isolation architecture is maintained.")
        sys.exit(0)
    else:
        print("❌ TORCH ISOLATION QUALITY GATE FAILED!")
        print("\nPlease address the issues above before proceeding.")
        print("The torch isolation architecture is not maintained.")
        sys.exit(1)


if __name__ == "__main__":
    main()
