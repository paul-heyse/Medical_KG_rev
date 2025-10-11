#!/usr/bin/env python3
"""Comprehensive torch isolation validation script.

This script runs all validation tests to ensure that the torch isolation
architecture has been properly implemented and that no torch dependencies
remain in the main gateway.
"""

import sys
from pathlib import Path
from typing import Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.validation.test_docker_image_validation import DockerImageValidator
from tests.validation.test_production_deployment_validation import ProductionDeploymentValidator
from tests.validation.test_service_api_compatibility import ServiceAPICompatibilityValidator
from tests.validation.test_torch_isolation_completeness import TorchIsolationValidator


def run_torch_isolation_validation() -> dict[str, Any]:
    """Run comprehensive torch isolation validation."""
    print("🔍 Running Torch Isolation Validation...")
    print("=" * 60)

    # Initialize validators
    torch_validator = TorchIsolationValidator(project_root)
    api_validator = ServiceAPICompatibilityValidator()
    docker_validator = DockerImageValidator(project_root)
    production_validator = ProductionDeploymentValidator(project_root)

    # Run validations
    print("\n1. Torch Isolation Completeness Validation")
    print("-" * 50)
    torch_report = torch_validator.get_validation_report()

    print(f"   Torch imports found: {torch_report['torch_imports_found']}")
    print(f"   Torch usage found: {torch_report['torch_usage_found']}")
    print(f"   gRPC clients exist: {torch_report['grpc_clients_exist']}")
    print(f"   Proto definitions exist: {torch_report['proto_definitions_exist']}")
    print(f"   Docker configurations exist: {torch_report['docker_configurations_exist']}")
    print(f"   Requirements torch-free: {torch_report['requirements_torch_free']}")
    print(f"   ✅ Isolation complete: {torch_report['isolation_complete']}")

    if torch_report["torch_imports"]:
        print("\n   Torch imports found:")
        for import_line in torch_report["torch_imports"]:
            print(f"     - {import_line}")

    if torch_report["torch_usage"]:
        print("\n   Torch usage found:")
        for file_path, line_num, line_content in torch_report["torch_usage"]:
            print(f"     - {file_path}:{line_num}: {line_content}")

    print("\n2. Service API Compatibility Validation")
    print("-" * 50)
    api_report = api_validator.get_compatibility_report()

    print(
        f"   GPU Service API Complete: {all(method['valid'] for method in api_report['gpu_service'].values())}"
    )
    print(
        f"   Embedding Service API Complete: {all(method['valid'] for method in api_report['embedding_service'].values())}"
    )
    print(
        f"   Reranking Service API Complete: {all(method['valid'] for method in api_report['reranking_service'].values())}"
    )
    print(
        f"   Docling VLM Service API Complete: {all(method['valid'] for method in api_report['docling_vlm_service'].values())}"
    )
    print(f"   ✅ Overall Compatible: {api_report['overall_compatible']}")

    if not api_report["overall_compatible"]:
        print("\n   API Compatibility Issues:")
        for service_name, service_api in api_report.items():
            if service_name != "overall_compatible":
                for method_name, result in service_api.items():
                    if not result["valid"]:
                        print(f"     - {service_name}.{method_name}: Missing")

    print("\n3. Docker Image Validation")
    print("-" * 50)
    docker_report = docker_validator.get_validation_report()

    print(f"   Gateway Dockerfile Torch-Free: {docker_report['gateway_dockerfile']['torch_free']}")
    print(
        f"   GPU Service Dockerfiles Exist: {docker_report['gpu_service_dockerfiles']['all_exist']}"
    )
    print(
        f"   GPU Service Dockerfiles Configured: {docker_report['gpu_service_dockerfiles']['all_properly_configured']}"
    )
    print(f"   Docker Compose Exists: {docker_report['docker_compose']['exists']}")
    print(f"   Docker Compose Configured: {docker_report['docker_compose']['properly_configured']}")
    print(
        f"   Kubernetes Deployments Exist: {docker_report['kubernetes_deployments']['all_exist']}"
    )
    print(
        f"   Kubernetes Deployments Configured: {docker_report['kubernetes_deployments']['all_properly_configured']}"
    )
    print(f"   ✅ Overall Valid: {docker_report['overall_valid']}")

    if not docker_report["overall_valid"]:
        print("\n   Docker Validation Issues:")

        if not docker_report["gateway_dockerfile"]["torch_free"]:
            print(f"     - Gateway Dockerfile: {docker_report['gateway_dockerfile']['issues']}")

        if (
            not docker_report["gpu_service_dockerfiles"]["all_exist"]
            or not docker_report["gpu_service_dockerfiles"]["all_properly_configured"]
        ):
            print(
                f"     - GPU Service Dockerfiles: {docker_report['gpu_service_dockerfiles']['services']}"
            )

        if not docker_report["docker_compose"]["properly_configured"]:
            print(f"     - Docker Compose: {docker_report['docker_compose']['issues']}")

        if (
            not docker_report["kubernetes_deployments"]["all_exist"]
            or not docker_report["kubernetes_deployments"]["all_properly_configured"]
        ):
            print(
                f"     - Kubernetes Deployments: {docker_report['kubernetes_deployments']['deployments']}"
            )

    print("\n4. Production Deployment Validation")
    print("-" * 50)
    production_report = production_validator.get_validation_report()

    print(f"   Requirements.txt Torch-Free: {production_report['requirements_txt']['torch_free']}")
    print(f"   Docker Images Exist: {production_report['docker_images']['all_exist']}")
    print(f"   Docker Images Torch-Free: {production_report['docker_images']['all_torch_free']}")
    print(
        f"   Kubernetes Deployments Exist: {production_report['kubernetes_deployments']['all_exist']}"
    )
    print(
        f"   Kubernetes Deployments Configured: {production_report['kubernetes_deployments']['all_properly_configured']}"
    )
    print(
        f"   CI/CD Pipeline Configured: {production_report['ci_cd_pipeline']['properly_configured']}"
    )
    print(
        f"   Monitoring Configuration Exists: {production_report['monitoring_configuration']['all_exist']}"
    )
    print(
        f"   Monitoring Configuration Configured: {production_report['monitoring_configuration']['all_properly_configured']}"
    )
    print(f"   ✅ Overall Valid: {production_report['overall_valid']}")

    if not production_report["overall_valid"]:
        print("\n   Production Deployment Validation Issues:")

        if not production_report["requirements_txt"]["torch_free"]:
            print(f"     - Requirements.txt: {production_report['requirements_txt']['issues']}")

        if (
            not production_report["docker_images"]["all_exist"]
            or not production_report["docker_images"]["all_torch_free"]
        ):
            print(f"     - Docker Images: {production_report['docker_images']['images']}")

        if (
            not production_report["kubernetes_deployments"]["all_exist"]
            or not production_report["kubernetes_deployments"]["all_properly_configured"]
        ):
            print(
                f"     - Kubernetes Deployments: {production_report['kubernetes_deployments']['deployments']}"
            )

        if not production_report["ci_cd_pipeline"]["properly_configured"]:
            print(f"     - CI/CD Pipeline: {production_report['ci_cd_pipeline']['issues']}")

        if (
            not production_report["monitoring_configuration"]["all_exist"]
            or not production_report["monitoring_configuration"]["all_properly_configured"]
        ):
            print(
                f"     - Monitoring Configuration: {production_report['monitoring_configuration']['dashboards']}"
            )

    # Overall validation result
    overall_success = (
        torch_report["isolation_complete"]
        and api_report["overall_compatible"]
        and docker_report["overall_valid"]
        and production_report["overall_valid"]
    )

    print("\n" + "=" * 60)
    print("🎯 OVERALL VALIDATION RESULT")
    print("=" * 60)

    if overall_success:
        print("✅ TORCH ISOLATION VALIDATION PASSED!")
        print("\nAll validation checks have passed successfully:")
        print("  ✓ No torch imports in main gateway")
        print("  ✓ No torch usage in main gateway")
        print("  ✓ gRPC clients provide equivalent functionality")
        print("  ✓ Docker images properly configured")
        print("  ✓ Production deployment ready")
        print("\nThe torch isolation architecture is complete and ready for production!")
    else:
        print("❌ TORCH ISOLATION VALIDATION FAILED!")
        print("\nPlease address the issues above before proceeding to production.")
        print("The torch isolation architecture is not complete.")

    return {
        "overall_success": overall_success,
        "torch_isolation": torch_report,
        "api_compatibility": api_report,
        "docker_validation": docker_report,
        "production_deployment": production_report,
    }


def main() -> None:
    """Main entry point for the validation script."""
    try:
        result = run_torch_isolation_validation()

        if result["overall_success"]:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Validation script failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
