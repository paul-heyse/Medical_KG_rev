"""Docker image validation tests.

This module validates that Docker images are properly configured for
torch isolation and that production images are torch-free.
"""

from pathlib import Path
from typing import Any

import pytest


class DockerImageValidator:
    """Validates Docker image configurations for torch isolation."""

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

    def validate_gateway_dockerfile(self) -> dict[str, Any]:
        """Validate that gateway Dockerfile is torch-free."""
        dockerfile_path = self.project_root / "ops/docker/gateway/Dockerfile"

        if not dockerfile_path.exists():
            return {
                "exists": False,
                "torch_free": False,
                "issues": ["Gateway Dockerfile does not exist"],
            }

        try:
            with open(dockerfile_path, encoding="utf-8") as f:
                content = f.read()

            issues = []
            torch_packages_found = []

            # Check for torch package installations
            for package in self.torch_packages:
                if package in content.lower():
                    torch_packages_found.append(package)

            if torch_packages_found:
                issues.append(f"Found torch packages: {torch_packages_found}")

            # Check for torch-related environment variables
            torch_env_vars = ["TORCH", "PYTORCH", "CUDA"]
            for env_var in torch_env_vars:
                if env_var in content:
                    issues.append(f"Found torch environment variable: {env_var}")

            return {
                "exists": True,
                "torch_free": len(issues) == 0,
                "issues": issues,
                "torch_packages_found": torch_packages_found,
            }

        except (UnicodeDecodeError, OSError) as e:
            return {
                "exists": True,
                "torch_free": False,
                "issues": [f"Error reading Dockerfile: {e}"],
            }

    def validate_gpu_service_dockerfiles(self) -> dict[str, Any]:
        """Validate that GPU service Dockerfiles exist and are properly configured."""
        dockerfiles = [
            "ops/docker/gpu-management/Dockerfile",
            "ops/docker/embedding-service/Dockerfile",
            "ops/docker/reranking-service/Dockerfile",
            "ops/docker/docling-vlm-service/Dockerfile",
        ]

        results = {}
        all_exist = True
        all_properly_configured = True

        for dockerfile in dockerfiles:
            dockerfile_path = self.project_root / dockerfile
            service_name = dockerfile.split("/")[-2]

            if not dockerfile_path.exists():
                results[service_name] = {
                    "exists": False,
                    "properly_configured": False,
                    "issues": ["Dockerfile does not exist"],
                }
                all_exist = False
                all_properly_configured = False
                continue

            try:
                with open(dockerfile_path, encoding="utf-8") as f:
                    content = f.read()

                issues = []

                # Check for basic Dockerfile structure
                if "FROM" not in content:
                    issues.append("Missing FROM instruction")

                if "RUN" not in content:
                    issues.append("Missing RUN instruction")

                # Check for torch installation (should be present in GPU services)
                torch_found = any(package in content.lower() for package in self.torch_packages)
                if not torch_found:
                    issues.append("Missing torch installation (required for GPU services)")

                # Check for gRPC health check
                if "grpc_health_probe" not in content:
                    issues.append("Missing gRPC health check")

                # Check for proper port exposure
                if "EXPOSE" not in content:
                    issues.append("Missing EXPOSE instruction")

                results[service_name] = {
                    "exists": True,
                    "properly_configured": len(issues) == 0,
                    "issues": issues,
                    "torch_installed": torch_found,
                }

                if len(issues) > 0:
                    all_properly_configured = False

            except (UnicodeDecodeError, OSError) as e:
                results[service_name] = {
                    "exists": True,
                    "properly_configured": False,
                    "issues": [f"Error reading Dockerfile: {e}"],
                }
                all_properly_configured = False

        return {
            "all_exist": all_exist,
            "all_properly_configured": all_properly_configured,
            "services": results,
        }

    def validate_docker_compose_configuration(self) -> dict[str, Any]:
        """Validate docker-compose.torch-isolation.yml configuration."""
        compose_path = self.project_root / "ops/docker/docker-compose.torch-isolation.yml"

        if not compose_path.exists():
            return {
                "exists": False,
                "properly_configured": False,
                "issues": ["docker-compose.torch-isolation.yml does not exist"],
            }

        try:
            with open(compose_path, encoding="utf-8") as f:
                content = f.read()

            issues = []
            services_found = []

            # Check for required services
            required_services = [
                "gpu-management-service",
                "embedding-service",
                "reranking-service",
                "docling-vlm-service",
                "gateway",
            ]

            for service in required_services:
                if service in content:
                    services_found.append(service)
                else:
                    issues.append(f"Missing required service: {service}")

            # Check for gRPC port configurations
            grpc_ports = ["50051", "50052", "50053", "50054"]
            for port in grpc_ports:
                if port not in content:
                    issues.append(f"Missing gRPC port configuration: {port}")

            # Check for volume mounts
            if "volumes:" not in content:
                issues.append("Missing volumes configuration")

            # Check for environment variables
            if "environment:" not in content:
                issues.append("Missing environment variables configuration")

            return {
                "exists": True,
                "properly_configured": len(issues) == 0,
                "issues": issues,
                "services_found": services_found,
                "required_services": required_services,
            }

        except (UnicodeDecodeError, OSError) as e:
            return {
                "exists": True,
                "properly_configured": False,
                "issues": [f"Error reading docker-compose file: {e}"],
            }

    def validate_kubernetes_deployments(self) -> dict[str, Any]:
        """Validate Kubernetes deployment configurations."""
        k8s_files = [
            "ops/k8s/gateway-deployment-torch-free.yaml",
            "ops/k8s/gpu-services-deployment.yaml",
            "ops/k8s/embedding-services-deployment.yaml",
            "ops/k8s/reranking-services-deployment.yaml",
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

                # Check for resource limits
                if "resources:" not in content:
                    issues.append("Missing resources specification")

                # Check for health checks
                if "livenessProbe" not in content:
                    issues.append("Missing livenessProbe")

                if "readinessProbe" not in content:
                    issues.append("Missing readinessProbe")

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

    def get_validation_report(self) -> dict[str, Any]:
        """Generate a comprehensive Docker validation report."""
        return {
            "gateway_dockerfile": self.validate_gateway_dockerfile(),
            "gpu_service_dockerfiles": self.validate_gpu_service_dockerfiles(),
            "docker_compose": self.validate_docker_compose_configuration(),
            "kubernetes_deployments": self.validate_kubernetes_deployments(),
            "overall_valid": (
                self.validate_gateway_dockerfile()["torch_free"]
                and self.validate_gpu_service_dockerfiles()["all_exist"]
                and self.validate_gpu_service_dockerfiles()["all_properly_configured"]
                and self.validate_docker_compose_configuration()["properly_configured"]
                and self.validate_kubernetes_deployments()["all_exist"]
                and self.validate_kubernetes_deployments()["all_properly_configured"]
            ),
        }


class TestDockerImageValidation:
    """Test suite for Docker image validation."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def validator(self, project_root: Path) -> DockerImageValidator:
        """Create a Docker image validator."""
        return DockerImageValidator(project_root)

    def test_gateway_dockerfile_torch_free(self, validator: DockerImageValidator) -> None:
        """Test that gateway Dockerfile is torch-free."""
        result = validator.validate_gateway_dockerfile()

        assert result["exists"], "Gateway Dockerfile does not exist"
        assert result[
            "torch_free"
        ], f"Gateway Dockerfile contains torch dependencies: {result['issues']}"

    def test_gpu_service_dockerfiles_exist(self, validator: DockerImageValidator) -> None:
        """Test that GPU service Dockerfiles exist."""
        result = validator.validate_gpu_service_dockerfiles()

        assert result["all_exist"], "Some GPU service Dockerfiles are missing"

    def test_gpu_service_dockerfiles_configured(self, validator: DockerImageValidator) -> None:
        """Test that GPU service Dockerfiles are properly configured."""
        result = validator.validate_gpu_service_dockerfiles()

        assert result[
            "all_properly_configured"
        ], f"Some GPU service Dockerfiles are not properly configured: {result['services']}"

    def test_docker_compose_exists(self, validator: DockerImageValidator) -> None:
        """Test that docker-compose.torch-isolation.yml exists."""
        result = validator.validate_docker_compose_configuration()

        assert result["exists"], "docker-compose.torch-isolation.yml does not exist"

    def test_docker_compose_configured(self, validator: DockerImageValidator) -> None:
        """Test that docker-compose.torch-isolation.yml is properly configured."""
        result = validator.validate_docker_compose_configuration()

        assert result[
            "properly_configured"
        ], f"docker-compose.torch-isolation.yml is not properly configured: {result['issues']}"

    def test_kubernetes_deployments_exist(self, validator: DockerImageValidator) -> None:
        """Test that Kubernetes deployment files exist."""
        result = validator.validate_kubernetes_deployments()

        assert result["all_exist"], "Some Kubernetes deployment files are missing"

    def test_kubernetes_deployments_configured(self, validator: DockerImageValidator) -> None:
        """Test that Kubernetes deployment files are properly configured."""
        result = validator.validate_kubernetes_deployments()

        assert result[
            "all_properly_configured"
        ], f"Some Kubernetes deployment files are not properly configured: {result['deployments']}"

    def test_overall_docker_validation(self, validator: DockerImageValidator) -> None:
        """Test overall Docker validation."""
        report = validator.get_validation_report()

        assert report["overall_valid"], f"Docker validation failed: {report}"


class TestDockerImageContent:
    """Test suite for Docker image content validation."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent

    def test_gateway_dockerfile_content(self, project_root: Path) -> None:
        """Test gateway Dockerfile content."""
        dockerfile_path = project_root / "ops/docker/gateway/Dockerfile"

        if dockerfile_path.exists():
            with open(dockerfile_path, encoding="utf-8") as f:
                content = f.read()

            # Should not contain torch installations
            assert "torch" not in content.lower(), "Gateway Dockerfile contains torch"
            assert "pytorch" not in content.lower(), "Gateway Dockerfile contains pytorch"
            assert "transformers" not in content.lower(), "Gateway Dockerfile contains transformers"

            # Should contain gRPC dependencies
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

                # Should contain torch installations
                assert "torch" in content.lower(), f"{dockerfile} missing torch installation"

                # Should contain gRPC dependencies
                assert "grpcio" in content.lower(), f"{dockerfile} missing grpcio"
                assert "grpcio-tools" in content.lower(), f"{dockerfile} missing grpcio-tools"

                # Should contain health check
                assert "health" in content.lower(), f"{dockerfile} missing health check"

    def test_docker_compose_service_definitions(self, project_root: Path) -> None:
        """Test docker-compose service definitions."""
        compose_path = project_root / "ops/docker/docker-compose.torch-isolation.yml"

        if compose_path.exists():
            with open(compose_path, encoding="utf-8") as f:
                content = f.read()

            # Should contain all required services
            required_services = [
                "gpu-management-service",
                "embedding-service",
                "reranking-service",
                "docling-vlm-service",
                "gateway",
            ]

            for service in required_services:
                assert service in content, f"docker-compose missing service: {service}"

            # Should contain gRPC port mappings
            grpc_ports = ["50051", "50052", "50053", "50054"]
            for port in grpc_ports:
                assert port in content, f"docker-compose missing gRPC port: {port}"

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
    # Run Docker validation when script is executed directly
    project_root = Path(__file__).parent.parent.parent
    validator = DockerImageValidator(project_root)

    report = validator.get_validation_report()

    print("=== Docker Image Validation Report ===")
    print(f"Gateway Dockerfile Torch-Free: {report['gateway_dockerfile']['torch_free']}")
    print(f"GPU Service Dockerfiles Exist: {report['gpu_service_dockerfiles']['all_exist']}")
    print(
        f"GPU Service Dockerfiles Configured: {report['gpu_service_dockerfiles']['all_properly_configured']}"
    )
    print(f"Docker Compose Exists: {report['docker_compose']['exists']}")
    print(f"Docker Compose Configured: {report['docker_compose']['properly_configured']}")
    print(f"Kubernetes Deployments Exist: {report['kubernetes_deployments']['all_exist']}")
    print(
        f"Kubernetes Deployments Configured: {report['kubernetes_deployments']['all_properly_configured']}"
    )
    print(f"Overall Valid: {report['overall_valid']}")

    if not report["overall_valid"]:
        print("\nDocker Validation Issues:")

        if not report["gateway_dockerfile"]["torch_free"]:
            print(f"  - Gateway Dockerfile: {report['gateway_dockerfile']['issues']}")

        if (
            not report["gpu_service_dockerfiles"]["all_exist"]
            or not report["gpu_service_dockerfiles"]["all_properly_configured"]
        ):
            print(f"  - GPU Service Dockerfiles: {report['gpu_service_dockerfiles']['services']}")

        if not report["docker_compose"]["properly_configured"]:
            print(f"  - Docker Compose: {report['docker_compose']['issues']}")

        if (
            not report["kubernetes_deployments"]["all_exist"]
            or not report["kubernetes_deployments"]["all_properly_configured"]
        ):
            print(f"  - Kubernetes Deployments: {report['kubernetes_deployments']['deployments']}")

    if report["overall_valid"]:
        print("\n✅ Docker image validation passed!")
    else:
        print("\n❌ Docker image validation failed!")
        exit(1)
