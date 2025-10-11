#!/usr/bin/env python3
"""Validation script for torch isolation deployment.

This script validates that the torch isolation deployment is working correctly,
including torch-free main gateway operation, GPU service functionality,
service communication, and error handling.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Medical_KG_rev.services.clients.embedding_client import EmbeddingClientManager
from Medical_KG_rev.services.clients.gpu_client import GPUClientManager
from Medical_KG_rev.services.clients.reranking_client import RerankingClientManager
from Medical_KG_rev.services.parsing.docling_vlm_client import DoclingVLMClientManager

app = typer.Typer(help="Torch Isolation Deployment Validation")
console = Console()


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, name: str, success: bool, message: str = "", details: dict[str, Any] = None):
        self.name = name
        self.success = success
        self.message = message
        self.details = details or {}


class TorchIsolationValidator:
    """Validator for torch isolation deployment."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or "ops/config/torch_isolation_deployment.yaml"
        self.config = self._load_config()
        self.client_managers = {}
        self.results: list[ValidationResult] = []

    def _load_config(self) -> dict[str, Any]:
        """Load validation configuration."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                "validation": {"timeout": 300, "retries": 3, "health_check_interval": 30},
                "services": {
                    "gpu_management": {"port": 50051},
                    "embedding_service": {"port": 50052},
                    "reranking_service": {"port": 50053},
                    "docling_vlm_service": {"port": 50054},
                    "gateway": {"port": 8000},
                },
            }

    async def initialize_clients(self) -> bool:
        """Initialize gRPC client managers."""
        try:
            console.print("🔌 Initializing gRPC client managers...", style="blue")

            # Initialize client managers
            self.client_managers["gpu"] = GPUClientManager()
            self.client_managers["embedding"] = EmbeddingClientManager()
            self.client_managers["reranking"] = RerankingClientManager()
            self.client_managers["docling_vlm"] = DoclingVLMClientManager()

            # Initialize clients
            for name, manager in self.client_managers.items():
                try:
                    await manager.initialize()
                    console.print(f"✅ {name} client initialized", style="green")
                except Exception as e:
                    console.print(f"❌ {name} client initialization failed: {e}", style="red")
                    return False

            return True

        except Exception as e:
            console.print(f"❌ Client initialization failed: {e}", style="red")
            return False

    async def validate_torch_free_gateway(self) -> ValidationResult:
        """Validate that the main gateway is torch-free."""
        try:
            console.print("🔍 Validating torch-free gateway...", style="blue")

            # Check for torch imports in gateway code
            gateway_path = Path("src/Medical_KG_rev/gateway")
            torch_imports = []

            for py_file in gateway_path.rglob("*.py"):
                try:
                    content = py_file.read_text()
                    if "import torch" in content or "from torch" in content:
                        torch_imports.append(str(py_file))
                except Exception:
                    continue

            if torch_imports:
                return ValidationResult(
                    "torch_free_gateway",
                    False,
                    f"Found torch imports in gateway: {torch_imports}",
                    {"torch_imports": torch_imports},
                )

            # Check gateway health endpoint
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:8000/health", timeout=10.0)
                    if response.status_code == 200:
                        health_data = response.json()
                        if health_data.get("status") == "healthy":
                            return ValidationResult(
                                "torch_free_gateway",
                                True,
                                "Gateway is torch-free and healthy",
                                {"health_status": health_data},
                            )
                        else:
                            return ValidationResult(
                                "torch_free_gateway",
                                False,
                                f"Gateway health check failed: {health_data}",
                                {"health_status": health_data},
                            )
                    else:
                        return ValidationResult(
                            "torch_free_gateway",
                            False,
                            f"Gateway health endpoint returned {response.status_code}",
                            {"status_code": response.status_code},
                        )
            except Exception as e:
                return ValidationResult(
                    "torch_free_gateway",
                    False,
                    f"Gateway health check failed: {e}",
                    {"error": str(e)},
                )

        except Exception as e:
            return ValidationResult(
                "torch_free_gateway",
                False,
                f"Torch-free gateway validation failed: {e}",
                {"error": str(e)},
            )

    async def validate_gpu_service_functionality(self) -> ValidationResult:
        """Validate GPU service functionality."""
        try:
            console.print("🔍 Validating GPU service functionality...", style="blue")

            if "gpu" not in self.client_managers:
                return ValidationResult(
                    "gpu_service_functionality", False, "GPU client manager not initialized"
                )

            gpu_manager = self.client_managers["gpu"]

            # Test GPU status
            try:
                status = await gpu_manager.get_status()
                if not status:
                    return ValidationResult(
                        "gpu_service_functionality", False, "GPU service status check failed"
                    )
            except Exception as e:
                return ValidationResult(
                    "gpu_service_functionality",
                    False,
                    f"GPU service status check failed: {e}",
                    {"error": str(e)},
                )

            # Test GPU device listing
            try:
                devices = await gpu_manager.list_devices()
                if not devices:
                    return ValidationResult(
                        "gpu_service_functionality", False, "GPU service device listing failed"
                    )
            except Exception as e:
                return ValidationResult(
                    "gpu_service_functionality",
                    False,
                    f"GPU service device listing failed: {e}",
                    {"error": str(e)},
                )

            return ValidationResult(
                "gpu_service_functionality",
                True,
                "GPU service functionality validated",
                {"status": status, "devices": devices},
            )

        except Exception as e:
            return ValidationResult(
                "gpu_service_functionality",
                False,
                f"GPU service functionality validation failed: {e}",
                {"error": str(e)},
            )

    async def validate_embedding_service_functionality(self) -> ValidationResult:
        """Validate embedding service functionality."""
        try:
            console.print("🔍 Validating embedding service functionality...", style="blue")

            if "embedding" not in self.client_managers:
                return ValidationResult(
                    "embedding_service_functionality",
                    False,
                    "Embedding client manager not initialized",
                )

            embedding_manager = self.client_managers["embedding"]

            # Test embedding generation
            try:
                test_text = "This is a test document for embedding generation."
                embedding = await embedding_manager.generate_embedding(test_text, "test-model")

                if not embedding:
                    return ValidationResult(
                        "embedding_service_functionality", False, "Embedding generation failed"
                    )

                # Test batch embedding generation
                batch_texts = [test_text, "Another test document", "Third test document"]
                batch_embeddings = await embedding_manager.generate_embeddings_batch(
                    batch_texts, "test-model"
                )

                if not batch_embeddings or len(batch_embeddings) != len(batch_texts):
                    return ValidationResult(
                        "embedding_service_functionality",
                        False,
                        "Batch embedding generation failed",
                    )

            except Exception as e:
                return ValidationResult(
                    "embedding_service_functionality",
                    False,
                    f"Embedding service functionality failed: {e}",
                    {"error": str(e)},
                )

            return ValidationResult(
                "embedding_service_functionality",
                True,
                "Embedding service functionality validated",
                {"single_embedding": bool(embedding), "batch_embeddings": len(batch_embeddings)},
            )

        except Exception as e:
            return ValidationResult(
                "embedding_service_functionality",
                False,
                f"Embedding service functionality validation failed: {e}",
                {"error": str(e)},
            )

    async def validate_reranking_service_functionality(self) -> ValidationResult:
        """Validate reranking service functionality."""
        try:
            console.print("🔍 Validating reranking service functionality...", style="blue")

            if "reranking" not in self.client_managers:
                return ValidationResult(
                    "reranking_service_functionality",
                    False,
                    "Reranking client manager not initialized",
                )

            reranking_manager = self.client_managers["reranking"]

            # Test reranking
            try:
                test_query = "medical research study"
                test_documents = [
                    "This is a medical research study about heart disease.",
                    "This document discusses cooking recipes.",
                    "Another medical study about diabetes treatment.",
                ]

                reranked = await reranking_manager.rerank_batch(
                    test_query, test_documents, "test-model"
                )

                if not reranked:
                    return ValidationResult(
                        "reranking_service_functionality", False, "Reranking failed"
                    )

            except Exception as e:
                return ValidationResult(
                    "reranking_service_functionality",
                    False,
                    f"Reranking service functionality failed: {e}",
                    {"error": str(e)},
                )

            return ValidationResult(
                "reranking_service_functionality",
                True,
                "Reranking service functionality validated",
                {"reranked": bool(reranked)},
            )

        except Exception as e:
            return ValidationResult(
                "reranking_service_functionality",
                False,
                f"Reranking service functionality validation failed: {e}",
                {"error": str(e)},
            )

    async def validate_docling_vlm_service_functionality(self) -> ValidationResult:
        """Validate Docling VLM service functionality."""
        try:
            console.print("🔍 Validating Docling VLM service functionality...", style="blue")

            if "docling_vlm" not in self.client_managers:
                return ValidationResult(
                    "docling_vlm_service_functionality",
                    False,
                    "Docling VLM client manager not initialized",
                )

            docling_manager = self.client_managers["docling_vlm"]

            # Test VLM processing (with mock PDF content)
            try:
                # Create a simple test PDF content
                test_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test PDF) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"

                result = await docling_manager.process_pdf(test_pdf_content)

                if not result:
                    return ValidationResult(
                        "docling_vlm_service_functionality", False, "Docling VLM processing failed"
                    )

            except Exception as e:
                return ValidationResult(
                    "docling_vlm_service_functionality",
                    False,
                    f"Docling VLM service functionality failed: {e}",
                    {"error": str(e)},
                )

            return ValidationResult(
                "docling_vlm_service_functionality",
                True,
                "Docling VLM service functionality validated",
                {"processed": bool(result)},
            )

        except Exception as e:
            return ValidationResult(
                "docling_vlm_service_functionality",
                False,
                f"Docling VLM service functionality validation failed: {e}",
                {"error": str(e)},
            )

    async def validate_service_communication(self) -> ValidationResult:
        """Validate service communication and error handling."""
        try:
            console.print("🔍 Validating service communication...", style="blue")

            # Test service health checks
            health_results = {}
            for name, manager in self.client_managers.items():
                try:
                    health_status = await manager.health_check()
                    health_results[name] = health_status
                except Exception:
                    health_results[name] = False

            # Check if all services are healthy
            all_healthy = all(health_results.values())

            if not all_healthy:
                return ValidationResult(
                    "service_communication",
                    False,
                    f"Some services are not healthy: {health_results}",
                    {"health_results": health_results},
                )

            # Test service stats
            stats_results = {}
            for name, manager in self.client_managers.items():
                try:
                    stats = await manager.get_stats()
                    stats_results[name] = stats
                except Exception:
                    stats_results[name] = None

            return ValidationResult(
                "service_communication",
                True,
                "Service communication validated",
                {"health_results": health_results, "stats_results": stats_results},
            )

        except Exception as e:
            return ValidationResult(
                "service_communication",
                False,
                f"Service communication validation failed: {e}",
                {"error": str(e)},
            )

    async def validate_deployment_health(self) -> ValidationResult:
        """Validate overall deployment health."""
        try:
            console.print("🔍 Validating deployment health...", style="blue")

            # Check Kubernetes deployments
            try:
                result = subprocess.run(
                    ["kubectl", "get", "deployments", "-o", "json"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                deployments = json.loads(result.stdout)

                # Check if all required deployments are ready
                required_deployments = [
                    "gpu-management-service",
                    "embedding-service",
                    "reranking-service",
                    "docling-vlm-service",
                    "gateway",
                ]

                deployment_status = {}
                for deployment in deployments.get("items", []):
                    name = deployment["metadata"]["name"]
                    if name in required_deployments:
                        ready_replicas = deployment["status"].get("readyReplicas", 0)
                        desired_replicas = deployment["spec"].get("replicas", 0)
                        deployment_status[name] = {
                            "ready": ready_replicas,
                            "desired": desired_replicas,
                            "healthy": ready_replicas == desired_replicas,
                        }

                all_deployments_healthy = all(
                    status["healthy"] for status in deployment_status.values()
                )

                if not all_deployments_healthy:
                    return ValidationResult(
                        "deployment_health",
                        False,
                        f"Some deployments are not healthy: {deployment_status}",
                        {"deployment_status": deployment_status},
                    )

            except subprocess.CalledProcessError as e:
                return ValidationResult(
                    "deployment_health",
                    False,
                    f"Failed to check Kubernetes deployments: {e}",
                    {"error": str(e)},
                )

            return ValidationResult(
                "deployment_health",
                True,
                "Deployment health validated",
                {"deployment_status": deployment_status},
            )

        except Exception as e:
            return ValidationResult(
                "deployment_health",
                False,
                f"Deployment health validation failed: {e}",
                {"error": str(e)},
            )

    async def run_all_validations(self) -> list[ValidationResult]:
        """Run all validation checks."""
        console.print("🚀 Starting torch isolation deployment validation...", style="blue")

        # Initialize clients
        if not await self.initialize_clients():
            self.results.append(
                ValidationResult(
                    "client_initialization", False, "Failed to initialize gRPC client managers"
                )
            )
            return self.results

        # Run validation checks
        validation_checks = [
            self.validate_torch_free_gateway,
            self.validate_gpu_service_functionality,
            self.validate_embedding_service_functionality,
            self.validate_reranking_service_functionality,
            self.validate_docling_vlm_service_functionality,
            self.validate_service_communication,
            self.validate_deployment_health,
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for check in validation_checks:
                task = progress.add_task(f"Running {check.__name__}...", total=100)

                try:
                    result = await check()
                    self.results.append(result)

                    if result.success:
                        progress.update(task, advance=100, description=f"✅ {result.name}")
                    else:
                        progress.update(task, advance=100, description=f"❌ {result.name}")

                except Exception as e:
                    error_result = ValidationResult(
                        check.__name__, False, f"Validation check failed: {e}", {"error": str(e)}
                    )
                    self.results.append(error_result)
                    progress.update(task, advance=100, description=f"❌ {check.__name__}")

        return self.results

    def display_results(self) -> None:
        """Display validation results."""
        console.print("\n📊 Validation Results", style="bold blue")

        # Summary table
        summary_table = Table(title="Validation Summary")
        summary_table.add_column("Check", style="cyan")
        summary_table.add_column("Status", style="green")
        summary_table.add_column("Message", style="white")

        for result in self.results:
            if result.success:
                status = "✅ PASS"
                style = "green"
            else:
                status = "❌ FAIL"
                style = "red"

            summary_table.add_row(result.name, status, result.message)

        console.print(summary_table)

        # Overall result
        all_passed = all(result.success for result in self.results)
        if all_passed:
            console.print(
                "\n✅ All validations passed! Torch isolation deployment is healthy.", style="green"
            )
        else:
            failed_count = sum(1 for result in self.results if not result.success)
            console.print(
                f"\n❌ {failed_count} validation(s) failed. Please check the details above.",
                style="red",
            )

    async def close(self) -> None:
        """Close resources."""
        for manager in self.client_managers.values():
            if hasattr(manager, "close"):
                await manager.close()


@app.command()
def validate(
    config_path: str | None = typer.Option(None, help="Path to validation configuration"),
    timeout: int = typer.Option(300, help="Validation timeout in seconds"),
):
    """Validate torch isolation deployment."""

    async def _validate():
        validator = TorchIsolationValidator(config_path)

        try:
            results = await validator.run_all_validations()
            validator.display_results()

            # Exit with appropriate code
            all_passed = all(result.success for result in results)
            sys.exit(0 if all_passed else 1)

        except Exception as e:
            console.print(f"❌ Validation error: {e}", style="red")
            sys.exit(1)
        finally:
            await validator.close()

    asyncio.run(_validate())


@app.command()
def check_torch_free(
    config_path: str | None = typer.Option(None, help="Path to validation configuration"),
):
    """Check if the main gateway is torch-free."""

    async def _check_torch_free():
        validator = TorchIsolationValidator(config_path)

        try:
            result = await validator.validate_torch_free_gateway()

            if result.success:
                console.print("✅ Main gateway is torch-free", style="green")
            else:
                console.print(f"❌ Main gateway is not torch-free: {result.message}", style="red")

            sys.exit(0 if result.success else 1)

        except Exception as e:
            console.print(f"❌ Torch-free check error: {e}", style="red")
            sys.exit(1)
        finally:
            await validator.close()

    asyncio.run(_check_torch_free())


@app.command()
def check_service_health(
    config_path: str | None = typer.Option(None, help="Path to validation configuration"),
):
    """Check service health."""

    async def _check_service_health():
        validator = TorchIsolationValidator(config_path)

        try:
            if not await validator.initialize_clients():
                console.print("❌ Failed to initialize clients", style="red")
                sys.exit(1)

            result = await validator.validate_service_communication()

            if result.success:
                console.print("✅ All services are healthy", style="green")
            else:
                console.print(f"❌ Service health check failed: {result.message}", style="red")

            sys.exit(0 if result.success else 1)

        except Exception as e:
            console.print(f"❌ Service health check error: {e}", style="red")
            sys.exit(1)
        finally:
            await validator.close()

    asyncio.run(_check_service_health())


if __name__ == "__main__":
    app()
