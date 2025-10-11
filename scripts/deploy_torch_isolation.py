#!/usr/bin/env python3
"""Deployment automation script for torch isolation architecture.

This script automates the deployment of the torch-isolated service architecture,
including blue-green deployment, service discovery, health checks, and rollback procedures.
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
from rich.panel import Panel
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

app = typer.Typer(help="Torch Isolation Deployment Automation")
console = Console()


class DeploymentConfig:
    """Configuration for torch isolation deployment."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or "ops/config/torch_isolation_deployment.yaml"
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load deployment configuration."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default configuration
            return {
                "deployment": {
                    "strategy": "blue_green",
                    "environment": "production",
                    "namespace": "medical-kg",
                    "timeout": 1800,
                    "health_check_interval": 30,
                    "max_retries": 3,
                },
                "services": {
                    "gpu_management": {
                        "replicas": 2,
                        "resources": {"cpu": "1000m", "memory": "2Gi", "gpu": "1"},
                        "port": 50051,
                    },
                    "embedding_service": {
                        "replicas": 3,
                        "resources": {"cpu": "2000m", "memory": "8Gi", "gpu": "1"},
                        "port": 50052,
                    },
                    "reranking_service": {
                        "replicas": 2,
                        "resources": {"cpu": "1000m", "memory": "4Gi", "gpu": "1"},
                        "port": 50053,
                    },
                    "docling_vlm_service": {
                        "replicas": 2,
                        "resources": {"cpu": "2000m", "memory": "24Gi", "gpu": "1"},
                        "port": 50054,
                    },
                },
                "gateway": {
                    "replicas": 3,
                    "resources": {"cpu": "1000m", "memory": "2Gi"},
                    "port": 8000,
                },
                "monitoring": {
                    "enabled": True,
                    "metrics_endpoint": "/metrics",
                    "health_endpoint": "/health",
                },
            }

    def get_service_config(self, service_name: str) -> dict[str, Any]:
        """Get configuration for a specific service."""
        return self.config.get("services", {}).get(service_name, {})

    def get_deployment_config(self) -> dict[str, Any]:
        """Get deployment configuration."""
        return self.config.get("deployment", {})


class ServiceHealthChecker:
    """Service health checking utilities."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.client_managers = {}

    async def initialize_clients(self) -> None:
        """Initialize gRPC client managers."""
        try:
            # Initialize client managers
            self.client_managers["gpu"] = GPUClientManager()
            self.client_managers["embedding"] = EmbeddingClientManager()
            self.client_managers["reranking"] = RerankingClientManager()
            self.client_managers["docling_vlm"] = DoclingVLMClientManager()

            # Initialize clients
            for manager in self.client_managers.values():
                await manager.initialize()

        except Exception as e:
            console.print(f"❌ Failed to initialize client managers: {e}", style="red")
            raise

    async def check_service_health(self, service_name: str) -> dict[str, Any]:
        """Check health of a specific service."""
        if service_name not in self.client_managers:
            return {"status": "unknown", "error": "Service not found"}

        try:
            manager = self.client_managers[service_name]

            # Check health endpoint
            health_status = await manager.health_check()

            # Get service stats
            stats = await manager.get_stats()

            return {
                "status": "healthy" if health_status else "unhealthy",
                "stats": stats,
                "timestamp": asyncio.get_event_loop().time(),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time(),
            }

    async def check_all_services(self) -> dict[str, dict[str, Any]]:
        """Check health of all services."""
        results = {}

        for service_name in self.client_managers.keys():
            results[service_name] = await self.check_service_health(service_name)

        return results

    async def wait_for_services_healthy(self, timeout: int = 300) -> bool:
        """Wait for all services to be healthy."""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            health_results = await self.check_all_services()

            # Check if all services are healthy
            all_healthy = all(result["status"] == "healthy" for result in health_results.values())

            if all_healthy:
                return True

            # Wait before next check
            await asyncio.sleep(
                self.config.get_deployment_config().get("health_check_interval", 30)
            )

        return False

    async def close(self) -> None:
        """Close all client managers."""
        for manager in self.client_managers.values():
            if hasattr(manager, "close"):
                await manager.close()


class KubernetesDeployer:
    """Kubernetes deployment utilities."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.namespace = config.get_deployment_config().get("namespace", "medical-kg")

    def apply_manifest(self, manifest_path: str) -> bool:
        """Apply Kubernetes manifest."""
        try:
            cmd = ["kubectl", "apply", "-f", manifest_path, "-n", self.namespace]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            console.print(f"✅ Applied manifest: {manifest_path}", style="green")
            return True

        except subprocess.CalledProcessError as e:
            console.print(f"❌ Failed to apply manifest {manifest_path}: {e}", style="red")
            console.print(f"Error output: {e.stderr}", style="red")
            return False

    def delete_manifest(self, manifest_path: str) -> bool:
        """Delete Kubernetes manifest."""
        try:
            cmd = ["kubectl", "delete", "-f", manifest_path, "-n", self.namespace]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            console.print(f"✅ Deleted manifest: {manifest_path}", style="green")
            return True

        except subprocess.CalledProcessError as e:
            console.print(f"❌ Failed to delete manifest {manifest_path}: {e}", style="red")
            console.print(f"Error output: {e.stderr}", style="red")
            return False

    def get_deployment_status(self, deployment_name: str) -> dict[str, Any]:
        """Get deployment status."""
        try:
            cmd = [
                "kubectl",
                "get",
                "deployment",
                deployment_name,
                "-n",
                self.namespace,
                "-o",
                "json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            return json.loads(result.stdout)

        except subprocess.CalledProcessError as e:
            return {"error": str(e)}

    def wait_for_deployment_ready(self, deployment_name: str, timeout: int = 600) -> bool:
        """Wait for deployment to be ready."""
        try:
            cmd = [
                "kubectl",
                "wait",
                "--for=condition=available",
                f"deployment/{deployment_name}",
                "-n",
                self.namespace,
                f"--timeout={timeout}s",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            return True

        except subprocess.CalledProcessError:
            return False

    def get_pod_logs(self, deployment_name: str, lines: int = 100) -> str:
        """Get pod logs for a deployment."""
        try:
            # Get pod name
            cmd = [
                "kubectl",
                "get",
                "pods",
                "-n",
                self.namespace,
                "-l",
                f"app={deployment_name}",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            pod_name = result.stdout.strip()

            if not pod_name:
                return "No pods found"

            # Get logs
            cmd = ["kubectl", "logs", pod_name, "-n", self.namespace, f"--tail={lines}"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            return result.stdout

        except subprocess.CalledProcessError as e:
            return f"Error getting logs: {e}"


class TorchIsolationDeployer:
    """Main deployment orchestrator for torch isolation."""

    def __init__(self, config_path: str | None = None):
        self.config = DeploymentConfig(config_path)
        self.health_checker = ServiceHealthChecker(self.config)
        self.k8s_deployer = KubernetesDeployer(self.config)
        self.manifest_paths = {
            "gpu_management": "ops/k8s/gpu-management-service.yaml",
            "embedding_service": "ops/k8s/embedding-service.yaml",
            "reranking_service": "ops/k8s/reranking-service.yaml",
            "docling_vlm_service": "ops/k8s/docling-vlm-service.yaml",
            "gateway": "ops/k8s/gateway-deployment-torch-free.yaml",
            "hpa": "ops/k8s/hpa-gpu-services.yaml",
            "monitoring": "ops/k8s/gpu-metrics-exporter.yaml",
        }

    async def deploy_services(self, services: list[str]) -> bool:
        """Deploy specified services."""
        success = True

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for service in services:
                if service not in self.manifest_paths:
                    console.print(f"⚠️  Unknown service: {service}", style="yellow")
                    continue

                manifest_path = self.manifest_paths[service]
                task = progress.add_task(f"Deploying {service}...", total=100)

                try:
                    # Apply manifest
                    progress.update(task, advance=30, description=f"Applying {service} manifest...")

                    if not self.k8s_deployer.apply_manifest(manifest_path):
                        success = False
                        progress.update(
                            task, advance=100, description=f"❌ Failed to deploy {service}"
                        )
                        continue

                    # Wait for deployment
                    progress.update(
                        task, advance=50, description=f"Waiting for {service} to be ready..."
                    )

                    if not self.k8s_deployer.wait_for_deployment_ready(service):
                        success = False
                        progress.update(
                            task, advance=100, description=f"❌ {service} deployment timeout"
                        )
                        continue

                    progress.update(
                        task, advance=100, description=f"✅ {service} deployed successfully"
                    )

                except Exception as e:
                    success = False
                    progress.update(
                        task, advance=100, description=f"❌ {service} deployment failed: {e}"
                    )

        return success

    async def validate_deployment(self) -> bool:
        """Validate the deployment."""
        console.print("\n🔍 Validating deployment...", style="blue")

        try:
            # Initialize health checker
            await self.health_checker.initialize_clients()

            # Wait for services to be healthy
            timeout = self.config.get_deployment_config().get("timeout", 1800)
            healthy = await self.health_checker.wait_for_services_healthy(timeout)

            if healthy:
                console.print("✅ All services are healthy", style="green")
                return True
            else:
                console.print("❌ Some services are not healthy", style="red")

                # Show health status
                health_results = await self.health_checker.check_all_services()
                self._display_health_status(health_results)

                return False

        except Exception as e:
            console.print(f"❌ Validation failed: {e}", style="red")
            return False

    def _display_health_status(self, health_results: dict[str, dict[str, Any]]) -> None:
        """Display health status of services."""
        table = Table(title="Service Health Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")

        for service, result in health_results.items():
            status = result.get("status", "unknown")
            details = result.get("error", "OK")

            if status == "healthy":
                status_text = "✅ Healthy"
                style = "green"
            elif status == "unhealthy":
                status_text = "❌ Unhealthy"
                style = "red"
            else:
                status_text = "⚠️  Unknown"
                style = "yellow"

            table.add_row(service, status_text, details)

        console.print(table)

    async def rollback_deployment(self, services: list[str]) -> bool:
        """Rollback deployment for specified services."""
        console.print(f"\n🔄 Rolling back services: {', '.join(services)}", style="yellow")

        success = True

        for service in services:
            if service not in self.manifest_paths:
                console.print(f"⚠️  Unknown service: {service}", style="yellow")
                continue

            manifest_path = self.manifest_paths[service]

            if not self.k8s_deployer.delete_manifest(manifest_path):
                success = False

        return success

    async def get_deployment_status(self) -> dict[str, Any]:
        """Get overall deployment status."""
        status = {}

        for service in self.manifest_paths.keys():
            deployment_status = self.k8s_deployer.get_deployment_status(service)
            status[service] = deployment_status

        return status

    async def close(self) -> None:
        """Close resources."""
        await self.health_checker.close()


@app.command()
def deploy(
    services: str = typer.Option("all", help="Comma-separated list of services to deploy"),
    config_path: str | None = typer.Option(None, help="Path to deployment configuration"),
    validate: bool = typer.Option(True, help="Validate deployment after completion"),
    timeout: int = typer.Option(1800, help="Deployment timeout in seconds"),
):
    """Deploy torch isolation services."""

    async def _deploy():
        deployer = TorchIsolationDeployer(config_path)

        try:
            # Parse services
            if services == "all":
                service_list = list(deployer.manifest_paths.keys())
            else:
                service_list = [s.strip() for s in services.split(",")]

            console.print(
                f"🚀 Deploying torch isolation services: {', '.join(service_list)}", style="blue"
            )

            # Deploy services
            success = await deployer.deploy_services(service_list)

            if not success:
                console.print("❌ Deployment failed", style="red")
                return

            # Validate deployment
            if validate:
                if await deployer.validate_deployment():
                    console.print("✅ Deployment completed successfully", style="green")
                else:
                    console.print("❌ Deployment validation failed", style="red")
            else:
                console.print("✅ Deployment completed (validation skipped)", style="green")

        except Exception as e:
            console.print(f"❌ Deployment error: {e}", style="red")
        finally:
            await deployer.close()

    asyncio.run(_deploy())


@app.command()
def validate(
    config_path: str | None = typer.Option(None, help="Path to deployment configuration"),
    timeout: int = typer.Option(300, help="Validation timeout in seconds"),
):
    """Validate torch isolation deployment."""

    async def _validate():
        deployer = TorchIsolationDeployer(config_path)

        try:
            console.print("🔍 Validating torch isolation deployment...", style="blue")

            if await deployer.validate_deployment():
                console.print("✅ Deployment validation passed", style="green")
            else:
                console.print("❌ Deployment validation failed", style="red")

        except Exception as e:
            console.print(f"❌ Validation error: {e}", style="red")
        finally:
            await deployer.close()

    asyncio.run(_validate())


@app.command()
def rollback(
    services: str = typer.Option("all", help="Comma-separated list of services to rollback"),
    config_path: str | None = typer.Option(None, help="Path to deployment configuration"),
):
    """Rollback torch isolation deployment."""

    async def _rollback():
        deployer = TorchIsolationDeployer(config_path)

        try:
            # Parse services
            if services == "all":
                service_list = list(deployer.manifest_paths.keys())
            else:
                service_list = [s.strip() for s in services.split(",")]

            console.print(f"🔄 Rolling back services: {', '.join(service_list)}", style="yellow")

            if await deployer.rollback_deployment(service_list):
                console.print("✅ Rollback completed successfully", style="green")
            else:
                console.print("❌ Rollback failed", style="red")

        except Exception as e:
            console.print(f"❌ Rollback error: {e}", style="red")
        finally:
            await deployer.close()

    asyncio.run(_rollback())


@app.command()
def status(config_path: str | None = typer.Option(None, help="Path to deployment configuration")):
    """Get deployment status."""

    async def _status():
        deployer = TorchIsolationDeployer(config_path)

        try:
            console.print("📊 Getting deployment status...", style="blue")

            status = await deployer.get_deployment_status()

            # Display status table
            table = Table(title="Deployment Status")
            table.add_column("Service", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Replicas", style="white")
            table.add_column("Ready", style="white")

            for service, deployment_status in status.items():
                if "error" in deployment_status:
                    table.add_row(service, "❌ Error", "N/A", "N/A")
                else:
                    spec = deployment_status.get("spec", {})
                    status_info = deployment_status.get("status", {})

                    replicas = spec.get("replicas", 0)
                    ready_replicas = status_info.get("readyReplicas", 0)

                    if ready_replicas == replicas:
                        status_text = "✅ Ready"
                    else:
                        status_text = "⚠️  Not Ready"

                    table.add_row(service, status_text, str(replicas), str(ready_replicas))

            console.print(table)

        except Exception as e:
            console.print(f"❌ Status error: {e}", style="red")
        finally:
            await deployer.close()

    asyncio.run(_status())


@app.command()
def logs(
    service: str = typer.Argument(..., help="Service name to get logs for"),
    lines: int = typer.Option(100, help="Number of log lines to retrieve"),
    config_path: str | None = typer.Option(None, help="Path to deployment configuration"),
):
    """Get service logs."""
    deployer = TorchIsolationDeployer(config_path)

    try:
        console.print(f"📋 Getting logs for {service}...", style="blue")

        logs = deployer.k8s_deployer.get_pod_logs(service, lines)

        if logs:
            console.print(Panel(logs, title=f"{service} Logs", border_style="blue"))
        else:
            console.print(f"❌ No logs found for {service}", style="red")

    except Exception as e:
        console.print(f"❌ Logs error: {e}", style="red")


if __name__ == "__main__":
    app()
