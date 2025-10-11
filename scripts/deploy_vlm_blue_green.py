#!/usr/bin/env python3
"""Blue-green deployment script for Docling VLM service.

This script implements a blue-green deployment strategy for the Docling VLM service,
allowing for zero-downtime deployments with rollback capabilities.
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer()


class DeploymentColor(Enum):
    """Deployment color for blue-green strategy."""

    BLUE = "blue"
    GREEN = "green"


class DeploymentStatus(Enum):
    """Deployment status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"


@dataclass
class DeploymentConfig:
    """Configuration for VLM deployment."""

    namespace: str = "medical-kg"
    service_name: str = "docling-vlm-service"
    image_tag: str = "latest"
    replicas: int = 2
    health_check_timeout: int = 300
    rollout_timeout: int = 600
    validation_timeout: int = 300
    rollback_timeout: int = 300


@dataclass
class DeploymentState:
    """Current deployment state."""

    active_color: DeploymentColor
    blue_deployment: Optional[str] = None
    green_deployment: Optional[str] = None
    blue_status: DeploymentStatus = DeploymentStatus.PENDING
    green_status: DeploymentStatus = DeploymentStatus.PENDING
    last_deployment_time: Optional[float] = None


class VLMBlueGreenDeployer:
    """Blue-green deployment manager for Docling VLM service."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.state_file = Path(f"/tmp/vlm-deployment-state-{config.namespace}.json")
        self.state = self._load_state()

    def _load_state(self) -> DeploymentState:
        """Load deployment state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    return DeploymentState(
                        active_color=DeploymentColor(data["active_color"]),
                        blue_deployment=data.get("blue_deployment"),
                        green_deployment=data.get("green_deployment"),
                        blue_status=DeploymentStatus(data.get("blue_status", "pending")),
                        green_status=DeploymentStatus(data.get("green_status", "pending")),
                        last_deployment_time=data.get("last_deployment_time"),
                    )
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

        return DeploymentState(active_color=DeploymentColor.BLUE)

    def _save_state(self) -> None:
        """Save deployment state to file."""
        data = {
            "active_color": self.state.active_color.value,
            "blue_deployment": self.state.blue_deployment,
            "green_deployment": self.state.green_deployment,
            "blue_status": self.state.blue_status.value,
            "green_status": self.state.green_status.value,
            "last_deployment_time": self.state.last_deployment_time,
        }

        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def _run_kubectl(
        self, command: list[str], capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run kubectl command."""
        full_command = ["kubectl"] + command
        logger.info(f"Running: {' '.join(full_command)}")

        return subprocess.run(full_command, capture_output=capture_output, text=True, check=False)

    def _get_deployment_name(self, color: DeploymentColor) -> str:
        """Get deployment name for color."""
        return f"{self.config.service_name}-{color.value}"

    def _get_service_name(self, color: DeploymentColor) -> str:
        """Get service name for color."""
        return f"{self.config.service_name}-{color.value}"

    def _get_ingress_name(self, color: DeploymentColor) -> str:
        """Get ingress name for color."""
        return f"{self.config.service_name}-{color.value}"

    async def _check_deployment_health(self, color: DeploymentColor) -> bool:
        """Check if deployment is healthy."""
        deployment_name = self._get_deployment_name(color)

        # Check if deployment is ready
        result = self._run_kubectl(
            [
                "get",
                "deployment",
                deployment_name,
                "-n",
                self.config.namespace,
                "-o",
                "jsonpath={.status.readyReplicas}",
            ]
        )

        if result.returncode != 0:
            return False

        try:
            ready_replicas = int(result.stdout.strip())
            return ready_replicas >= self.config.replicas
        except ValueError:
            return False

    async def _check_service_health(self, color: DeploymentColor) -> bool:
        """Check if service is healthy via health check."""
        service_name = self._get_service_name(color)

        # Get service endpoint
        result = self._run_kubectl(
            [
                "get",
                "service",
                service_name,
                "-n",
                self.config.namespace,
                "-o",
                "jsonpath={.spec.clusterIP}",
            ]
        )

        if result.returncode != 0:
            return False

        cluster_ip = result.stdout.strip()
        if not cluster_ip:
            return False

        # Perform health check via port-forward
        try:
            # Create port-forward
            port_forward = subprocess.Popen(
                [
                    "kubectl",
                    "port-forward",
                    f"service/{service_name}",
                    "50054:50054",
                    "-n",
                    self.config.namespace,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for port-forward to be ready
            await asyncio.sleep(2)

            # Perform health check
            health_check = subprocess.run(
                [
                    "python",
                    "-c",
                    """
import grpc
from grpc_health.v1 import health_pb2_grpc, health_pb2
try:
    channel = grpc.insecure_channel('localhost:50054')
    stub = health_pb2_grpc.HealthStub(channel)
    response = stub.Check(health_pb2.HealthCheckRequest(service='docling_vlm'))
    exit(0 if response.status == health_pb2.HealthCheckResponse.SERVING else 1)
except Exception:
    exit(1)
""",
                ],
                capture_output=True,
                timeout=10,
            )

            # Clean up port-forward
            port_forward.terminate()
            port_forward.wait()

            return health_check.returncode == 0

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def _wait_for_deployment(self, color: DeploymentColor, timeout: int) -> bool:
        """Wait for deployment to be ready."""
        deployment_name = self._get_deployment_name(color)

        with Progress(
            SpinnerColumn(), TextColumn(f"Waiting for {color.value} deployment..."), console=console
        ) as progress:
            task = progress.add_task("Deploying", total=None)

            start_time = time.time()
            while time.time() - start_time < timeout:
                if await self._check_deployment_health(color):
                    progress.update(task, description=f"{color.value} deployment ready")
                    return True

                await asyncio.sleep(5)
                progress.advance(task)

            progress.update(task, description=f"{color.value} deployment timeout")
            return False

    async def _validate_deployment(self, color: DeploymentColor) -> bool:
        """Validate deployment with test requests."""
        service_name = self._get_service_name(color)

        with Progress(
            SpinnerColumn(), TextColumn(f"Validating {color.value} deployment..."), console=console
        ) as progress:
            task = progress.add_task("Validating", total=None)

            # Test basic functionality
            try:
                # Create port-forward for testing
                port_forward = subprocess.Popen(
                    [
                        "kubectl",
                        "port-forward",
                        f"service/{service_name}",
                        "50054:50054",
                        "-n",
                        self.config.namespace,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                await asyncio.sleep(2)

                # Test health endpoint
                health_test = subprocess.run(
                    [
                        "python",
                        "-c",
                        """
import grpc
from grpc_health.v1 import health_pb2_grpc, health_pb2
try:
    channel = grpc.insecure_channel('localhost:50054')
    stub = health_pb2_grpc.HealthStub(channel)
    response = stub.Check(health_pb2.HealthCheckRequest(service='docling_vlm'))
    print(f"Health status: {response.status}")
    exit(0 if response.status == health_pb2.HealthCheckResponse.SERVING else 1)
except Exception as e:
    print(f"Health check failed: {e}")
    exit(1)
""",
                    ],
                    capture_output=True,
                    timeout=30,
                )

                # Clean up port-forward
                port_forward.terminate()
                port_forward.wait()

                if health_test.returncode == 0:
                    progress.update(task, description=f"{color.value} deployment validated")
                    return True
                else:
                    progress.update(task, description=f"{color.value} deployment validation failed")
                    return False

            except Exception as e:
                logger.error(f"Validation failed: {e}")
                progress.update(task, description=f"{color.value} deployment validation error")
                return False

    def _create_deployment_manifest(self, color: DeploymentColor) -> str:
        """Create deployment manifest for color."""
        deployment_name = self._get_deployment_name(color)
        service_name = self._get_service_name(color)

        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
    name: {deployment_name}
    namespace: {self.config.namespace}
    labels:
        app: {self.config.service_name}
        color: {color.value}
        version: {self.config.image_tag}
spec:
    replicas: {self.config.replicas}
    selector:
        matchLabels:
            app: {self.config.service_name}
            color: {color.value}
    template:
        metadata:
            labels:
                app: {self.config.service_name}
                color: {color.value}
                version: {self.config.image_tag}
        spec:
            containers:
                - name: docling-vlm-service
                  image: medical-kg/docling-vlm-service:{self.config.image_tag}
                  ports:
                      - containerPort: 50054
                        name: grpc
                      - containerPort: 8080
                        name: metrics
                  env:
                      - name: CUDA_VISIBLE_DEVICES
                        value: "0"
                      - name: GPU_MEMORY_FRACTION
                        value: "0.9"
                      - name: MODEL_NAME
                        value: "gemma3-12b"
                      - name: BATCH_SIZE
                        value: "4"
                  resources:
                      requests:
                          memory: "16Gi"
                          cpu: "2000m"
                          nvidia.com/gpu: 1
                      limits:
                          memory: "32Gi"
                          cpu: "4000m"
                          nvidia.com/gpu: 1
                  livenessProbe:
                      exec:
                          command:
                              - python
                              - -c
                              - "import grpc; from grpc_health.v1 import health_pb2_grpc, health_pb2; channel = grpc.insecure_channel('localhost:50054'); stub = health_pb2_grpc.HealthStub(channel); response = stub.Check(health_pb2.HealthCheckRequest(service='docling_vlm')); assert response.status == health_pb2.HealthCheckResponse.SERVING"
                      initialDelaySeconds: 120
                      periodSeconds: 30
                  readinessProbe:
                      exec:
                          command:
                              - python
                              - -c
                              - "import grpc; from grpc_health.v1 import health_pb2_grpc, health_pb2; channel = grpc.insecure_channel('localhost:50054'); stub = health_pb2_grpc.HealthStub(channel); response = stub.Check(health_pb2.HealthCheckRequest(service='docling_vlm')); assert response.status == health_pb2.HealthCheckResponse.SERVING"
                      initialDelaySeconds: 60
                      periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
    name: {service_name}
    namespace: {self.config.namespace}
    labels:
        app: {self.config.service_name}
        color: {color.value}
spec:
    selector:
        app: {self.config.service_name}
        color: {color.value}
    ports:
        - port: 50054
          targetPort: 50054
          name: grpc
        - port: 8080
          targetPort: 8080
          name: metrics
    type: ClusterIP
"""
        return manifest

    async def deploy(self, image_tag: str) -> bool:
        """Deploy new version using blue-green strategy."""
        self.config.image_tag = image_tag

        # Determine target color (opposite of current active)
        target_color = (
            DeploymentColor.GREEN
            if self.state.active_color == DeploymentColor.BLUE
            else DeploymentColor.BLUE
        )

        console.print("[bold blue]Starting blue-green deployment[/bold blue]")
        console.print(f"Active color: [bold]{self.state.active_color.value}[/bold]")
        console.print(f"Target color: [bold]{target_color.value}[/bold]")
        console.print(f"Image tag: [bold]{image_tag}[/bold]")

        try:
            # Step 1: Deploy to target color
            console.print(f"\n[bold]Step 1: Deploying to {target_color.value}[/bold]")

            manifest = self._create_deployment_manifest(target_color)
            manifest_file = Path(f"/tmp/vlm-{target_color.value}-deployment.yaml")

            with open(manifest_file, "w") as f:
                f.write(manifest)

            # Apply deployment
            result = self._run_kubectl(["apply", "-f", str(manifest_file)])
            if result.returncode != 0:
                console.print(
                    f"[bold red]Failed to apply {target_color.value} deployment[/bold red]"
                )
                return False

            # Step 2: Wait for deployment to be ready
            console.print(f"\n[bold]Step 2: Waiting for {target_color.value} deployment[/bold]")

            if not await self._wait_for_deployment(target_color, self.config.rollout_timeout):
                console.print(
                    f"[bold red]{target_color.value} deployment failed to become ready[/bold red]"
                )
                await self._cleanup_deployment(target_color)
                return False

            # Step 3: Validate deployment
            console.print(f"\n[bold]Step 3: Validating {target_color.value} deployment[/bold]")

            if not await self._validate_deployment(target_color):
                console.print(
                    f"[bold red]{target_color.value} deployment validation failed[/bold red]"
                )
                await self._cleanup_deployment(target_color)
                return False

            # Step 4: Switch traffic to new deployment
            console.print(f"\n[bold]Step 4: Switching traffic to {target_color.value}[/bold]")

            if not await self._switch_traffic(target_color):
                console.print(
                    f"[bold red]Failed to switch traffic to {target_color.value}[/bold red]"
                )
                return False

            # Step 5: Update state
            self.state.active_color = target_color
            self.state.last_deployment_time = time.time()
            self._save_state()

            # Step 6: Cleanup old deployment
            console.print("\n[bold]Step 6: Cleaning up old deployment[/bold]")

            old_color = (
                DeploymentColor.GREEN
                if target_color == DeploymentColor.BLUE
                else DeploymentColor.BLUE
            )
            await self._cleanup_deployment(old_color)

            console.print("[bold green]Blue-green deployment completed successfully![/bold green]")
            console.print(f"Active color: [bold]{self.state.active_color.value}[/bold]")

            return True

        except Exception as e:
            console.print(f"[bold red]Deployment failed: {e}[/bold red]")
            logger.error(f"Deployment error: {e}")
            return False

    async def _switch_traffic(self, target_color: DeploymentColor) -> bool:
        """Switch traffic to target color deployment."""
        # Update the main service to point to the target color
        service_name = self._get_service_name(target_color)

        # Patch the main service selector
        result = self._run_kubectl(
            [
                "patch",
                "service",
                self.config.service_name,
                "-n",
                self.config.namespace,
                "-p",
                f'{{"spec": {{"selector": {{"app": "{self.config.service_name}", "color": "{target_color.value}"}}}}}}',
            ]
        )

        return result.returncode == 0

    async def _cleanup_deployment(self, color: DeploymentColor) -> None:
        """Clean up deployment for color."""
        deployment_name = self._get_deployment_name(color)
        service_name = self._get_service_name(color)

        # Delete deployment
        self._run_kubectl(
            ["delete", "deployment", deployment_name, "-n", self.config.namespace],
            capture_output=False,
        )

        # Delete service
        self._run_kubectl(
            ["delete", "service", service_name, "-n", self.config.namespace], capture_output=False
        )

    async def rollback(self) -> bool:
        """Rollback to previous deployment."""
        console.print("[bold yellow]Starting rollback[/bold yellow]")

        # Determine target color (opposite of current active)
        target_color = (
            DeploymentColor.GREEN
            if self.state.active_color == DeploymentColor.BLUE
            else DeploymentColor.BLUE
        )

        console.print(f"Current active: [bold]{self.state.active_color.value}[/bold]")
        console.print(f"Rollback target: [bold]{target_color.value}[/bold]")

        try:
            # Switch traffic back
            if not await self._switch_traffic(target_color):
                console.print(
                    f"[bold red]Failed to switch traffic back to {target_color.value}[/bold red]"
                )
                return False

            # Update state
            self.state.active_color = target_color
            self._save_state()

            console.print("[bold green]Rollback completed successfully![/bold green]")
            console.print(f"Active color: [bold]{self.state.active_color.value}[/bold]")

            return True

        except Exception as e:
            console.print(f"[bold red]Rollback failed: {e}[/bold red]")
            logger.error(f"Rollback error: {e}")
            return False

    def status(self) -> None:
        """Show deployment status."""
        table = Table(title="VLM Blue-Green Deployment Status")
        table.add_column("Color", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Deployment", style="green")
        table.add_column("Last Updated", style="yellow")

        # Blue status
        blue_status = "Active" if self.state.active_color == DeploymentColor.BLUE else "Inactive"
        table.add_row(
            "Blue",
            blue_status,
            self.state.blue_deployment or "Not deployed",
            str(self.state.last_deployment_time or "Never"),
        )

        # Green status
        green_status = "Active" if self.state.active_color == DeploymentColor.GREEN else "Inactive"
        table.add_row(
            "Green",
            green_status,
            self.state.green_deployment or "Not deployed",
            str(self.state.last_deployment_time or "Never"),
        )

        console.print(table)


@app.command()
def deploy(
    image_tag: str = typer.Argument(..., help="Docker image tag to deploy"),
    namespace: str = typer.Option("medical-kg", help="Kubernetes namespace"),
    replicas: int = typer.Option(2, help="Number of replicas"),
    timeout: int = typer.Option(600, help="Deployment timeout in seconds"),
) -> None:
    """Deploy VLM service using blue-green strategy."""
    config = DeploymentConfig(namespace=namespace, replicas=replicas, rollout_timeout=timeout)

    deployer = VLMBlueGreenDeployer(config)

    async def _deploy():
        success = await deployer.deploy(image_tag)
        if not success:
            typer.exit(1)

    asyncio.run(_deploy())


@app.command()
def rollback(namespace: str = typer.Option("medical-kg", help="Kubernetes namespace")) -> None:
    """Rollback to previous deployment."""
    config = DeploymentConfig(namespace=namespace)
    deployer = VLMBlueGreenDeployer(config)

    async def _rollback():
        success = await deployer.rollback()
        if not success:
            typer.exit(1)

    asyncio.run(_rollback())


@app.command()
def status(namespace: str = typer.Option("medical-kg", help="Kubernetes namespace")) -> None:
    """Show deployment status."""
    config = DeploymentConfig(namespace=namespace)
    deployer = VLMBlueGreenDeployer(config)
    deployer.status()


if __name__ == "__main__":
    app()
