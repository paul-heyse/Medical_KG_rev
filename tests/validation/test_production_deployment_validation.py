"""Production deployment validation tests.

This module validates that production deployments are torch-free and
properly configured for the torch isolation architecture.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest


class ProductionDeploymentValidator:
    """Validates production deployment configurations for torch isolation."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.torch_packages = {
            "torch",
            "pytorch",
            "transformers",
            "accelerate",
            "safetensors",
            "tokenizers",
            "regex",
            "sentencepiece",
            "tiktoken",
            "onnxruntime",
            "onnx",
            "fast-pytorch-kmeans",
            "torchvision",
            "torchaudio",
        }

    def validate_requirements_txt(self) -> dict[str, Any]:
        """Validate that requirements.txt is torch-free."""
        requirements_path = self.project_root / "requirements.txt"

        if not requirements_path.exists():
            return {
                "exists": False,
                "torch_free": False,
                "issues": ["requirements.txt does not exist"],
            }

        try:
            with open(requirements_path, encoding="utf-8") as f:
                content = f.read()

            issues = []
            torch_packages_found = []

            # Check for torch packages
            for package in self.torch_packages:
                if package in content.lower():
                    torch_packages_found.append(package)

            if torch_packages_found:
                issues.append(f"Found torch packages: {torch_packages_found}")

            # Check for gRPC dependencies
            grpc_packages = ["grpcio", "grpcio-tools", "grpcio-health-checking"]
            missing_grpc = []
            for package in grpc_packages:
                if package not in content.lower():
                    missing_grpc.append(package)

            if missing_grpc:
                issues.append(f"Missing gRPC packages: {missing_grpc}")

            return {
                "exists": True,
                "torch_free": len(torch_packages_found) == 0,
                "grpc_packages_present": len(missing_grpc) == 0,
                "issues": issues,
                "torch_packages_found": torch_packages_found,
                "missing_grpc_packages": missing_grpc,
            }

        except (UnicodeDecodeError, OSError) as e:
            return {
                "exists": True,
                "torch_free": False,
                "issues": [f"Error reading requirements.txt: {e}"],
            }

    def validate_docker_images(self) -> dict[str, Any]:
        """Validate Docker images for production deployment."""
        docker_images = [
            "gateway",
            "gpu-management-service",
            "embedding-service",
            "reranking-service",
            "docling-vlm-service",
        ]

        results = {}
        all_exist = True
        all_torch_free = True

        for image in docker_images:
            dockerfile_path = self.project_root / f"ops/docker/{image}/Dockerfile"

            if not dockerfile_path.exists():
                results[image] = {
                    "exists": False,
                    "torch_free": False,
                    "issues": ["Dockerfile does not exist"],
                }
                all_exist = False
                all_torch_free = False
                continue

            try:
                with open(dockerfile_path, encoding="utf-8") as f:
                    content = f.read()

                issues = []
                torch_packages_found = []

                # Check for torch packages
                for package in self.torch_packages:
                    if package in content.lower():
                        torch_packages_found.append(package)

                # Gateway should be torch-free, GPU services should have torch
                if image == "gateway":
                    if torch_packages_found:
                        issues.append(
                            f"Gateway should be torch-free but contains: {torch_packages_found}"
                        )
                        all_torch_free = False
                else:
                    if not torch_packages_found:
                        issues.append("GPU service should contain torch packages")
                        all_torch_free = False

                # Check for gRPC dependencies
                if "grpcio" not in content.lower():
                    issues.append("Missing grpcio dependency")

                if "grpcio-tools" not in content.lower():
                    issues.append("Missing grpcio-tools dependency")

                results[image] = {
                    "exists": True,
                    "torch_free": len(issues) == 0,
                    "issues": issues,
                    "torch_packages_found": torch_packages_found,
                }

                if len(issues) > 0:
                    all_torch_free = False

            except (UnicodeDecodeError, OSError) as e:
                results[image] = {
                    "exists": True,
                    "torch_free": False,
                    "issues": [f"Error reading Dockerfile: {e}"],
                }
                all_torch_free = False

        return {"all_exist": all_exist, "all_torch_free": all_torch_free, "images": results}

    def validate_kubernetes_deployments(self) -> dict[str, Any]:
        """Validate Kubernetes deployment configurations."""
        k8s_files = [
            "ops/k8s/gateway-deployment-torch-free.yaml",
            "ops/k8s/gpu-services-deployment.yaml",
            "ops/k8s/embedding-services-deployment.yaml",
            "ops/k8s/reranking-services-deployment.yaml",
            "ops/k8s/docling-vlm-service-deployment.yaml",
        ]

        results = {}
        all_exist = True
        all_properly_configured = True

        for k8s_file in k8s_files:
            file_path = self.project_root / k8s_file
            deployment_name = k8s_file.split("/")[-1].replace(".yaml", "")

            if not file_path.exists():
                results[deployment_name] = {
                    "exists": False,
                    "properly_configured": False,
                    "issues": ["Kubernetes deployment file does not exist"],
                }
                all_exist = False
                all_properly_configured = False
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                issues = []

                # Check for basic Kubernetes structure
                if "apiVersion" not in content:
                    issues.append("Missing apiVersion")

                if "kind" not in content:
                    issues.append("Missing kind")

                if "metadata" not in content:
                    issues.append("Missing metadata")

                if "spec" not in content:
                    issues.append("Missing spec")

                # Check for container specifications
                if "containers:" not in content:
                    issues.append("Missing containers specification")

                if "image:" not in content:
                    issues.append("Missing image specification")

                # Check for resource limits
                if "resources:" not in content:
                    issues.append("Missing resources specification")

                # Check for health checks
                if "livenessProbe" not in content:
                    issues.append("Missing livenessProbe")

                if "readinessProbe" not in content:
                    issues.append("Missing readinessProbe")

                # Check for service mesh configuration
                if "serviceMesh" not in content and "istio" not in content.lower():
                    issues.append("Missing service mesh configuration")

                results[deployment_name] = {
                    "exists": True,
                    "properly_configured": len(issues) == 0,
                    "issues": issues,
                }

                if len(issues) > 0:
                    all_properly_configured = False

            except (UnicodeDecodeError, OSError) as e:
                results[deployment_name] = {
                    "exists": True,
                    "properly_configured": False,
                    "issues": [f"Error reading Kubernetes file: {e}"],
                }
                all_properly_configured = False

        return {
            "all_exist": all_exist,
            "all_properly_configured": all_properly_configured,
            "deployments": results,
        }

    def validate_ci_cd_pipeline(self) -> dict[str, Any]:
        """Validate CI/CD pipeline configuration."""
        ci_file = self.project_root / ".github/workflows/ci.yml"

        if not ci_file.exists():
            return {
                "exists": False,
                "properly_configured": False,
                "issues": ["CI/CD pipeline file does not exist"],
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

            return {"exists": True, "properly_configured": len(issues) == 0, "issues": issues}

        except (UnicodeDecodeError, OSError) as e:
            return {
                "exists": True,
                "properly_configured": False,
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
            "all_exist": all_exist,
            "all_properly_configured": all_properly_configured,
            "dashboards": results,
        }

    def get_validation_report(self) -> dict[str, Any]:
        """Generate a comprehensive production deployment validation report."""
        return {
            "requirements_txt": self.validate_requirements_txt(),
            "docker_images": self.validate_docker_images(),
            "kubernetes_deployments": self.validate_kubernetes_deployments(),
            "ci_cd_pipeline": self.validate_ci_cd_pipeline(),
            "monitoring_configuration": self.validate_monitoring_configuration(),
            "overall_valid": (
                self.validate_requirements_txt()["torch_free"]
                and self.validate_docker_images()["all_exist"]
                and self.validate_docker_images()["all_torch_free"]
                and self.validate_kubernetes_deployments()["all_exist"]
                and self.validate_kubernetes_deployments()["all_properly_configured"]
                and self.validate_ci_cd_pipeline()["properly_configured"]
                and self.validate_monitoring_configuration()["all_exist"]
                and self.validate_monitoring_configuration()["all_properly_configured"]
            ),
        }


class TestProductionDeploymentValidation:
    """Test suite for production deployment validation."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def validator(self, project_root: Path) -> ProductionDeploymentValidator:
        """Create a production deployment validator."""
        return ProductionDeploymentValidator(project_root)

    def test_requirements_txt_torch_free(self, validator: ProductionDeploymentValidator) -> None:
        """Test that requirements.txt is torch-free."""
        result = validator.validate_requirements_txt()

        assert result["exists"], "requirements.txt does not exist"
        assert result[
            "torch_free"
        ], f"requirements.txt contains torch dependencies: {result['issues']}"
        assert result[
            "grpc_packages_present"
        ], f"requirements.txt missing gRPC packages: {result['issues']}"

    def test_docker_images_exist(self, validator: ProductionDeploymentValidator) -> None:
        """Test that all Docker images exist."""
        result = validator.validate_docker_images()

        assert result["all_exist"], "Some Docker images are missing"

    def test_docker_images_torch_free(self, validator: ProductionDeploymentValidator) -> None:
        """Test that Docker images are properly configured for torch isolation."""
        result = validator.validate_docker_images()

        assert result[
            "all_torch_free"
        ], f"Some Docker images are not properly configured: {result['images']}"

    def test_kubernetes_deployments_exist(self, validator: ProductionDeploymentValidator) -> None:
        """Test that Kubernetes deployment files exist."""
        result = validator.validate_kubernetes_deployments()

        assert result["all_exist"], "Some Kubernetes deployment files are missing"

    def test_kubernetes_deployments_configured(
        self, validator: ProductionDeploymentValidator
    ) -> None:
        """Test that Kubernetes deployment files are properly configured."""
        result = validator.validate_kubernetes_deployments()

        assert result[
            "all_properly_configured"
        ], f"Some Kubernetes deployment files are not properly configured: {result['deployments']}"

    def test_ci_cd_pipeline_configured(self, validator: ProductionDeploymentValidator) -> None:
        """Test that CI/CD pipeline is properly configured."""
        result = validator.validate_ci_cd_pipeline()

        assert result["exists"], "CI/CD pipeline file does not exist"
        assert result[
            "properly_configured"
        ], f"CI/CD pipeline is not properly configured: {result['issues']}"

    def test_monitoring_configuration_exists(
        self, validator: ProductionDeploymentValidator
    ) -> None:
        """Test that monitoring configuration exists."""
        result = validator.validate_monitoring_configuration()

        assert result["all_exist"], "Some monitoring configuration files are missing"

    def test_monitoring_configuration_configured(
        self, validator: ProductionDeploymentValidator
    ) -> None:
        """Test that monitoring configuration is properly configured."""
        result = validator.validate_monitoring_configuration()

        assert result[
            "all_properly_configured"
        ], f"Some monitoring configuration files are not properly configured: {result['dashboards']}"

    def test_overall_production_validation(self, validator: ProductionDeploymentValidator) -> None:
        """Test overall production deployment validation."""
        report = validator.get_validation_report()

        assert report["overall_valid"], f"Production deployment validation failed: {report}"


class TestProductionDeploymentContent:
    """Test suite for production deployment content validation."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent

    def test_requirements_txt_content(self, project_root: Path) -> None:
        """Test requirements.txt content."""
        requirements_path = project_root / "requirements.txt"

        if requirements_path.exists():
            with open(requirements_path, encoding="utf-8") as f:
                content = f.read()

            # Should not contain torch packages
            torch_packages = ["torch", "pytorch", "transformers", "accelerate"]
            for package in torch_packages:
                assert package not in content.lower(), f"requirements.txt contains {package}"

            # Should contain gRPC packages
            grpc_packages = ["grpcio", "grpcio-tools", "grpcio-health-checking"]
            for package in grpc_packages:
                assert package in content.lower(), f"requirements.txt missing {package}"

    def test_gateway_dockerfile_content(self, project_root: Path) -> None:
        """Test gateway Dockerfile content."""
        dockerfile_path = project_root / "ops/docker/gateway/Dockerfile"

        if dockerfile_path.exists():
            with open(dockerfile_path, encoding="utf-8") as f:
                content = f.read()

            # Should not contain torch
            assert "torch" not in content.lower(), "Gateway Dockerfile contains torch"
            assert "pytorch" not in content.lower(), "Gateway Dockerfile contains pytorch"

            # Should contain gRPC
            assert "grpcio" in content.lower(), "Gateway Dockerfile missing grpcio"
            assert "grpcio-tools" in content.lower(), "Gateway Dockerfile missing grpcio-tools"

    def test_gpu_service_dockerfile_content(self, project_root: Path) -> None:
        """Test GPU service Dockerfile content."""
        dockerfiles = [
            "ops/docker/gpu-management/Dockerfile",
            "ops/docker/embedding-service/Dockerfile",
            "ops/docker/reranking-service/Dockerfile",
            "ops/docker/docling-vlm-service/Dockerfile",
        ]

        for dockerfile in dockerfiles:
            dockerfile_path = project_root / dockerfile

            if dockerfile_path.exists():
                with open(dockerfile_path, encoding="utf-8") as f:
                    content = f.read()

                # Should contain torch
                assert "torch" in content.lower(), f"{dockerfile} missing torch"

                # Should contain gRPC
                assert "grpcio" in content.lower(), f"{dockerfile} missing grpcio"
                assert "grpcio-tools" in content.lower(), f"{dockerfile} missing grpcio-tools"

    def test_ci_cd_pipeline_content(self, project_root: Path) -> None:
        """Test CI/CD pipeline content."""
        ci_file = project_root / ".github/workflows/ci.yml"

        if ci_file.exists():
            with open(ci_file, encoding="utf-8") as f:
                content = f.read()

            # Should contain Docker image building
            assert "docker build" in content.lower(), "CI/CD missing Docker image building"

            # Should contain service integration tests
            assert (
                "service-integration-tests" in content.lower()
            ), "CI/CD missing service integration tests"

            # Should contain torch isolation tests
            assert "torch-isolation" in content.lower(), "CI/CD missing torch isolation tests"

            # Should contain deployment automation
            assert (
                "deploy_torch_isolation" in content.lower()
            ), "CI/CD missing deployment automation"

    def test_kubernetes_deployment_content(self, project_root: Path) -> None:
        """Test Kubernetes deployment content."""
        k8s_files = [
            "ops/k8s/gateway-deployment-torch-free.yaml",
            "ops/k8s/gpu-services-deployment.yaml",
            "ops/k8s/embedding-services-deployment.yaml",
            "ops/k8s/reranking-services-deployment.yaml",
        ]

        for k8s_file in k8s_files:
            file_path = project_root / k8s_file

            if file_path.exists():
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Should contain basic Kubernetes structure
                assert "apiVersion" in content, f"{k8s_file} missing apiVersion"
                assert "kind" in content, f"{k8s_file} missing kind"
                assert "metadata" in content, f"{k8s_file} missing metadata"
                assert "spec" in content, f"{k8s_file} missing spec"

                # Should contain container specifications
                assert "containers:" in content, f"{k8s_file} missing containers"
                assert "image:" in content, f"{k8s_file} missing image specification"

                # Should contain resource specifications
                assert "resources:" in content, f"{k8s_file} missing resources"

                # Should contain health checks
                assert "livenessProbe" in content, f"{k8s_file} missing livenessProbe"
                assert "readinessProbe" in content, f"{k8s_file} missing readinessProbe"


if __name__ == "__main__":
    # Run production deployment validation when script is executed directly
    project_root = Path(__file__).parent.parent.parent
    validator = ProductionDeploymentValidator(project_root)

    report = validator.get_validation_report()

    print("=== Production Deployment Validation Report ===")
    print(f"Requirements.txt Torch-Free: {report['requirements_txt']['torch_free']}")
    print(f"Docker Images Exist: {report['docker_images']['all_exist']}")
    print(f"Docker Images Torch-Free: {report['docker_images']['all_torch_free']}")
    print(f"Kubernetes Deployments Exist: {report['kubernetes_deployments']['all_exist']}")
    print(
        f"Kubernetes Deployments Configured: {report['kubernetes_deployments']['all_properly_configured']}"
    )
    print(f"CI/CD Pipeline Configured: {report['ci_cd_pipeline']['properly_configured']}")
    print(f"Monitoring Configuration Exists: {report['monitoring_configuration']['all_exist']}")
    print(
        f"Monitoring Configuration Configured: {report['monitoring_configuration']['all_properly_configured']}"
    )
    print(f"Overall Valid: {report['overall_valid']}")

    if not report["overall_valid"]:
        print("\nProduction Deployment Validation Issues:")

        if not report["requirements_txt"]["torch_free"]:
            print(f"  - Requirements.txt: {report['requirements_txt']['issues']}")

        if (
            not report["docker_images"]["all_exist"]
            or not report["docker_images"]["all_torch_free"]
        ):
            print(f"  - Docker Images: {report['docker_images']['images']}")

        if (
            not report["kubernetes_deployments"]["all_exist"]
            or not report["kubernetes_deployments"]["all_properly_configured"]
        ):
            print(f"  - Kubernetes Deployments: {report['kubernetes_deployments']['deployments']}")

        if not report["ci_cd_pipeline"]["properly_configured"]:
            print(f"  - CI/CD Pipeline: {report['ci_cd_pipeline']['issues']}")

        if (
            not report["monitoring_configuration"]["all_exist"]
            or not report["monitoring_configuration"]["all_properly_configured"]
        ):
            print(
                f"  - Monitoring Configuration: {report['monitoring_configuration']['dashboards']}"
            )

    if report["overall_valid"]:
        print("\n✅ Production deployment validation passed!")
    else:
        print("\n❌ Production deployment validation failed!")
        sys.exit(1)
