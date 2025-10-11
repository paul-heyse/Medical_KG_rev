#!/usr/bin/env python3
"""Validate torch-free deployment script.

This script validates that deployments are torch-free and properly configured
for the torch isolation architecture.
"""

import json
import sys
from pathlib import Path
from typing import Any

from tests.validation.test_docker_image_validation import DockerImageValidator
from tests.validation.test_production_deployment_validation import ProductionDeploymentValidator


class TorchFreeDeploymentValidator:
    """Validator for torch-free deployments."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docker_validator = DockerImageValidator(project_root)
        self.production_validator = ProductionDeploymentValidator(project_root)

    def validate_docker_images(self) -> dict[str, Any]:
        """Validate Docker images for torch isolation."""
        docker_report = self.docker_validator.get_validation_report()

        return {
            "passed": docker_report["overall_valid"],
            "gateway_torch_free": docker_report["gateway_dockerfile"]["torch_free"],
            "gpu_services_exist": docker_report["gpu_service_dockerfiles"]["all_exist"],
            "gpu_services_configured": docker_report["gpu_service_dockerfiles"][
                "all_properly_configured"
            ],
            "docker_compose_exists": docker_report["docker_compose"]["exists"],
            "docker_compose_configured": docker_report["docker_compose"]["properly_configured"],
            "kubernetes_deployments_exist": docker_report["kubernetes_deployments"]["all_exist"],
            "kubernetes_deployments_configured": docker_report["kubernetes_deployments"][
                "all_properly_configured"
            ],
            "issues": self._extract_docker_issues(docker_report),
        }

    def validate_production_deployment(self) -> dict[str, Any]:
        """Validate production deployment configuration."""
        production_report = self.production_validator.get_validation_report()

        return {
            "passed": production_report["overall_valid"],
            "requirements_torch_free": production_report["requirements_txt"]["torch_free"],
            "docker_images_exist": production_report["docker_images"]["all_exist"],
            "docker_images_torch_free": production_report["docker_images"]["all_torch_free"],
            "kubernetes_deployments_exist": production_report["kubernetes_deployments"][
                "all_exist"
            ],
            "kubernetes_deployments_configured": production_report["kubernetes_deployments"][
                "all_properly_configured"
            ],
            "ci_cd_pipeline_configured": production_report["ci_cd_pipeline"]["properly_configured"],
            "monitoring_configuration_exists": production_report["monitoring_configuration"][
                "all_exist"
            ],
            "monitoring_configuration_configured": production_report["monitoring_configuration"][
                "all_properly_configured"
            ],
            "issues": self._extract_production_issues(production_report),
        }

    def validate_service_integration(self) -> dict[str, Any]:
        """Validate service integration for torch isolation."""
        # Check if docker-compose file exists and can be parsed
        compose_path = self.project_root / "ops/docker/docker-compose.torch-isolation.yml"

        if not compose_path.exists():
            return {
                "passed": False,
                "message": "docker-compose.torch-isolation.yml does not exist",
                "issues": ["Missing docker-compose.torch-isolation.yml"],
            }

        try:
            with open(compose_path, encoding="utf-8") as f:
                content = f.read()

            # Check for required services
            required_services = [
                "gpu-management-service",
                "embedding-service",
                "reranking-service",
                "docling-vlm-service",
                "gateway",
            ]

            missing_services = []
            for service in required_services:
                if service not in content:
                    missing_services.append(service)

            # Check for gRPC port configurations
            grpc_ports = ["50051", "50052", "50053", "50054"]
            missing_ports = []
            for port in grpc_ports:
                if port not in content:
                    missing_ports.append(port)

            issues = []
            if missing_services:
                issues.append(f"Missing services: {missing_services}")
            if missing_ports:
                issues.append(f"Missing gRPC ports: {missing_ports}")

            return {
                "passed": len(issues) == 0,
                "message": (
                    "Service integration properly configured"
                    if len(issues) == 0
                    else f"Service integration issues: {issues}"
                ),
                "issues": issues,
            }

        except (UnicodeDecodeError, OSError) as e:
            return {
                "passed": False,
                "message": f"Error reading docker-compose file: {e}",
                "issues": [f"Error reading docker-compose file: {e}"],
            }

    def validate_ci_cd_pipeline(self) -> dict[str, Any]:
        """Validate CI/CD pipeline for torch isolation."""
        ci_file = self.project_root / ".github/workflows/ci.yml"

        if not ci_file.exists():
            return {
                "passed": False,
                "message": "CI/CD pipeline file does not exist",
                "issues": ["Missing .github/workflows/ci.yml"],
            }

        try:
            with open(ci_file, encoding="utf-8") as f:
                content = f.read()

            issues = []

            # Check for Docker image building
            if "docker build" not in content.lower():
                issues.append("Missing Docker image building")

            # Check for service integration tests
            if "service-integration-tests" not in content.lower():
                issues.append("Missing service integration tests")

            # Check for torch isolation tests
            if "torch-isolation" not in content.lower():
                issues.append("Missing torch isolation tests")

            # Check for deployment automation
            if "deploy_torch_isolation" not in content.lower():
                issues.append("Missing deployment automation")

            # Check for validation steps
            if "validate_torch_isolation_deployment" not in content.lower():
                issues.append("Missing deployment validation")

            return {
                "passed": len(issues) == 0,
                "message": (
                    "CI/CD pipeline properly configured"
                    if len(issues) == 0
                    else f"CI/CD pipeline issues: {issues}"
                ),
                "issues": issues,
            }

        except (UnicodeDecodeError, OSError) as e:
            return {
                "passed": False,
                "message": f"Error reading CI/CD file: {e}",
                "issues": [f"Error reading CI/CD file: {e}"],
            }

    def validate_monitoring_configuration(self) -> dict[str, Any]:
        """Validate monitoring configuration for torch isolation."""
        monitoring_files = [
            "ops/monitoring/service-architecture-dashboard.json",
            "ops/monitoring/gpu-services-dashboard.json",
            "ops/monitoring/torch-isolation-dashboard.json",
        ]

        results = {}
        all_exist = True
        all_properly_configured = True

        for monitoring_file in monitoring_files:
            file_path = self.project_root / monitoring_file
            dashboard_name = monitoring_file.split("/")[-1].replace(".json", "")

            if not file_path.exists():
                results[dashboard_name] = {
                    "exists": False,
                    "properly_configured": False,
                    "issues": ["Monitoring dashboard file does not exist"],
                }
                all_exist = False
                all_properly_configured = False
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                issues = []

                # Check for valid JSON
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    issues.append("Invalid JSON format")

                # Check for dashboard structure
                if "dashboard" not in content.lower():
                    issues.append("Missing dashboard structure")

                if "panels" not in content.lower():
                    issues.append("Missing panels configuration")

                # Check for service-specific metrics
                if "gpu" in dashboard_name.lower():
                    if "gpu_utilization" not in content.lower():
                        issues.append("Missing GPU utilization metrics")

                if "service" in dashboard_name.lower():
                    if "grpc" not in content.lower():
                        issues.append("Missing gRPC metrics")

                results[dashboard_name] = {
                    "exists": True,
                    "properly_configured": len(issues) == 0,
                    "issues": issues,
                }

                if len(issues) > 0:
                    all_properly_configured = False

            except (UnicodeDecodeError, OSError) as e:
                results[dashboard_name] = {
                    "exists": True,
                    "properly_configured": False,
                    "issues": [f"Error reading monitoring file: {e}"],
                }
                all_properly_configured = False

        return {
            "passed": all_exist and all_properly_configured,
            "message": (
                "Monitoring configuration properly configured"
                if all_exist and all_properly_configured
                else "Monitoring configuration issues"
            ),
            "issues": self._extract_monitoring_issues(results),
            "dashboards": results,
        }

    def _extract_docker_issues(self, docker_report: dict[str, Any]) -> list[str]:
        """Extract Docker validation issues."""
        issues = []

        if not docker_report["gateway_dockerfile"]["torch_free"]:
            issues.extend(docker_report["gateway_dockerfile"]["issues"])

        if (
            not docker_report["gpu_service_dockerfiles"]["all_exist"]
            or not docker_report["gpu_service_dockerfiles"]["all_properly_configured"]
        ):
            for service_name, service_result in docker_report["gpu_service_dockerfiles"][
                "services"
            ].items():
                if not service_result["exists"] or not service_result["properly_configured"]:
                    issues.extend(service_result["issues"])

        if not docker_report["docker_compose"]["properly_configured"]:
            issues.extend(docker_report["docker_compose"]["issues"])

        if (
            not docker_report["kubernetes_deployments"]["all_exist"]
            or not docker_report["kubernetes_deployments"]["all_properly_configured"]
        ):
            for deployment_name, deployment_result in docker_report["kubernetes_deployments"][
                "deployments"
            ].items():
                if not deployment_result["exists"] or not deployment_result["properly_configured"]:
                    issues.extend(deployment_result["issues"])

        return issues

    def _extract_production_issues(self, production_report: dict[str, Any]) -> list[str]:
        """Extract production deployment validation issues."""
        issues = []

        if not production_report["requirements_txt"]["torch_free"]:
            issues.extend(production_report["requirements_txt"]["issues"])

        if (
            not production_report["docker_images"]["all_exist"]
            or not production_report["docker_images"]["all_torch_free"]
        ):
            for image_name, image_result in production_report["docker_images"]["images"].items():
                if not image_result["exists"] or not image_result["torch_free"]:
                    issues.extend(image_result["issues"])

        if (
            not production_report["kubernetes_deployments"]["all_exist"]
            or not production_report["kubernetes_deployments"]["all_properly_configured"]
        ):
            for deployment_name, deployment_result in production_report["kubernetes_deployments"][
                "deployments"
            ].items():
                if not deployment_result["exists"] or not deployment_result["properly_configured"]:
                    issues.extend(deployment_result["issues"])

        if not production_report["ci_cd_pipeline"]["properly_configured"]:
            issues.extend(production_report["ci_cd_pipeline"]["issues"])

        if (
            not production_report["monitoring_configuration"]["all_exist"]
            or not production_report["monitoring_configuration"]["all_properly_configured"]
        ):
            for dashboard_name, dashboard_result in production_report["monitoring_configuration"][
                "dashboards"
            ].items():
                if not dashboard_result["exists"] or not dashboard_result["properly_configured"]:
                    issues.extend(dashboard_result["issues"])

        return issues

    def _extract_monitoring_issues(self, monitoring_results: dict[str, Any]) -> list[str]:
        """Extract monitoring configuration issues."""
        issues = []

        for dashboard_name, dashboard_result in monitoring_results.items():
            if not dashboard_result["exists"] or not dashboard_result["properly_configured"]:
                issues.extend(dashboard_result["issues"])

        return issues

    def run_all_validations(self) -> dict[str, Any]:
        """Run all deployment validations."""
        validations = {
            "docker_images": self.validate_docker_images(),
            "production_deployment": self.validate_production_deployment(),
            "service_integration": self.validate_service_integration(),
            "ci_cd_pipeline": self.validate_ci_cd_pipeline(),
            "monitoring_configuration": self.validate_monitoring_configuration(),
        }

        all_passed = all(validation["passed"] for validation in validations.values())

        return {
            "all_passed": all_passed,
            "validations": validations,
            "summary": {
                "total_validations": len(validations),
                "passed_validations": sum(
                    1 for validation in validations.values() if validation["passed"]
                ),
                "failed_validations": sum(
                    1 for validation in validations.values() if not validation["passed"]
                ),
            },
        }


def main() -> None:
    """Main entry point for torch-free deployment validation."""
    project_root = Path(__file__).parent.parent

    print("🔍 Running Torch-Free Deployment Validation...")
    print("=" * 60)

    validator = TorchFreeDeploymentValidator(project_root)
    result = validator.run_all_validations()

    print("\n📊 Deployment Validation Results:")
    print(f"   Total Validations: {result['summary']['total_validations']}")
    print(f"   Passed: {result['summary']['passed_validations']}")
    print(f"   Failed: {result['summary']['failed_validations']}")

    print("\n🔍 Individual Validation Results:")
    for validation_name, validation_result in result["validations"].items():
        status = "✅ PASS" if validation_result["passed"] else "❌ FAIL"
        print(f"   {validation_name}: {status}")
        if not validation_result["passed"]:
            print(f"      {validation_result['message']}")
            if validation_result.get("issues"):
                for issue in validation_result["issues"]:
                    print(f"         - {issue}")

    print("\n" + "=" * 60)

    if result["all_passed"]:
        print("✅ TORCH-FREE DEPLOYMENT VALIDATION PASSED!")
        print("\nAll deployment validations have passed successfully.")
        print("The deployment is properly configured for torch isolation.")
        sys.exit(0)
    else:
        print("❌ TORCH-FREE DEPLOYMENT VALIDATION FAILED!")
        print("\nPlease address the issues above before proceeding.")
        print("The deployment is not properly configured for torch isolation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
