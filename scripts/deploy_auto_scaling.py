#!/usr/bin/env python3
"""Deploy Auto-Scaling Infrastructure for GPU Services.

This script deploys the auto-scaling infrastructure including
HPA configurations, custom metrics adapters, and monitoring.
"""

import argparse
import logging
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


class AutoScalingDeployer:
    """Deploys auto-scaling infrastructure for GPU services."""

    def __init__(self, namespace: str = "medical-kg"):
        """Initialize auto-scaling deployer.

        Args:
            namespace: Kubernetes namespace

        """
        self.namespace = namespace
        self.kubectl_cmd = "kubectl"

    def run_command(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a command and return the result.

        Args:
            cmd: Command to run
            check: Whether to check for errors

        Returns:
            Completed process

        """
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)

        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")
        if result.stderr:
            logger.warning(f"STDERR: {result.stderr}")

        return result

    def create_namespace(self) -> None:
        """Create Kubernetes namespace if it doesn't exist."""
        try:
            self.run_command([self.kubectl_cmd, "create", "namespace", self.namespace], check=False)
            logger.info(f"Namespace {self.namespace} created or already exists")
        except subprocess.CalledProcessError as e:
            if e.returncode != 1:  # Namespace already exists
                raise

    def deploy_gpu_metrics_exporter(self) -> None:
        """Deploy GPU metrics exporter."""
        logger.info("Deploying GPU metrics exporter...")

        # Apply GPU metrics exporter deployment
        self.run_command([self.kubectl_cmd, "apply", "-f", "ops/k8s/gpu-metrics-exporter.yaml"])

        # Wait for deployment to be ready
        self.run_command(
            [
                self.kubectl_cmd,
                "wait",
                "--for=condition=available",
                "deployment/gpu-metrics-exporter",
                "-n",
                self.namespace,
                "--timeout=300s",
            ]
        )

        logger.info("GPU metrics exporter deployed successfully")

    def deploy_custom_metrics_adapter(self) -> None:
        """Deploy custom metrics adapter."""
        logger.info("Deploying custom metrics adapter...")

        # Apply custom metrics adapter deployment
        self.run_command([self.kubectl_cmd, "apply", "-f", "ops/k8s/custom-metrics-adapter.yaml"])

        # Wait for deployment to be ready
        self.run_command(
            [
                self.kubectl_cmd,
                "wait",
                "--for=condition=available",
                "deployment/custom-metrics-adapter",
                "-n",
                self.namespace,
                "--timeout=300s",
            ]
        )

        logger.info("Custom metrics adapter deployed successfully")

    def deploy_hpa_configurations(self) -> None:
        """Deploy HPA configurations for GPU services."""
        logger.info("Deploying HPA configurations...")

        # Apply HPA configurations
        self.run_command([self.kubectl_cmd, "apply", "-f", "ops/k8s/hpa-gpu-services.yaml"])

        logger.info("HPA configurations deployed successfully")

    def verify_deployment(self) -> None:
        """Verify that the deployment is working correctly."""
        logger.info("Verifying deployment...")

        # Check GPU metrics exporter
        try:
            result = self.run_command(
                [
                    self.kubectl_cmd,
                    "get",
                    "pods",
                    "-l",
                    "app=gpu-metrics-exporter",
                    "-n",
                    self.namespace,
                    "-o",
                    "jsonpath={.items[0].status.phase}",
                ]
            )
            if result.stdout.strip() != "Running":
                raise Exception("GPU metrics exporter is not running")
            logger.info("GPU metrics exporter is running")
        except Exception as e:
            logger.error(f"GPU metrics exporter verification failed: {e}")
            raise

        # Check custom metrics adapter
        try:
            result = self.run_command(
                [
                    self.kubectl_cmd,
                    "get",
                    "pods",
                    "-l",
                    "app=custom-metrics-adapter",
                    "-n",
                    self.namespace,
                    "-o",
                    "jsonpath={.items[0].status.phase}",
                ]
            )
            if result.stdout.strip() != "Running":
                raise Exception("Custom metrics adapter is not running")
            logger.info("Custom metrics adapter is running")
        except Exception as e:
            logger.error(f"Custom metrics adapter verification failed: {e}")
            raise

        # Check HPA configurations
        try:
            result = self.run_command([self.kubectl_cmd, "get", "hpa", "-n", self.namespace])
            if "gpu-management-hpa" not in result.stdout:
                raise Exception("HPA configurations not found")
            logger.info("HPA configurations are active")
        except Exception as e:
            logger.error(f"HPA verification failed: {e}")
            raise

        logger.info("Deployment verification completed successfully")

    def get_deployment_status(self) -> dict[str, Any]:
        """Get status of the auto-scaling deployment.

        Returns:
            Dictionary with deployment status

        """
        status: dict[str, Any] = {
            "namespace": self.namespace,
            "gpu_metrics_exporter": {"status": "unknown"},
            "custom_metrics_adapter": {"status": "unknown"},
            "hpa_configurations": {"status": "unknown"},
        }

        try:
            # Check GPU metrics exporter
            result = self.run_command(
                [
                    self.kubectl_cmd,
                    "get",
                    "pods",
                    "-l",
                    "app=gpu-metrics-exporter",
                    "-n",
                    self.namespace,
                    "-o",
                    "jsonpath={.items[0].status.phase}",
                ],
                check=False,
            )
            status["gpu_metrics_exporter"]["status"] = result.stdout.strip()

            # Check custom metrics adapter
            result = self.run_command(
                [
                    self.kubectl_cmd,
                    "get",
                    "pods",
                    "-l",
                    "app=custom-metrics-adapter",
                    "-n",
                    self.namespace,
                    "-o",
                    "jsonpath={.items[0].status.phase}",
                ],
                check=False,
            )
            status["custom_metrics_adapter"]["status"] = result.stdout.strip()

            # Check HPA configurations
            result = self.run_command(
                [
                    self.kubectl_cmd,
                    "get",
                    "hpa",
                    "-n",
                    self.namespace,
                    "-o",
                    "jsonpath={.items[*].metadata.name}",
                ],
                check=False,
            )
            hpa_names = result.stdout.strip().split()
            status["hpa_configurations"]["status"] = "active" if hpa_names else "inactive"
            status["hpa_configurations"]["hpa_names"] = hpa_names

        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")

        return status

    def deploy(self) -> None:
        """Deploy the complete auto-scaling infrastructure."""
        logger.info("Starting auto-scaling infrastructure deployment...")

        try:
            # Create namespace
            self.create_namespace()

            # Deploy components
            self.deploy_gpu_metrics_exporter()
            self.deploy_custom_metrics_adapter()
            self.deploy_hpa_configurations()

            # Verify deployment
            self.verify_deployment()

            logger.info("Auto-scaling infrastructure deployed successfully")

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise

    def undeploy(self) -> None:
        """Undeploy the auto-scaling infrastructure."""
        logger.info("Starting auto-scaling infrastructure undeployment...")

        try:
            # Remove HPA configurations
            self.run_command(
                [self.kubectl_cmd, "delete", "-f", "ops/k8s/hpa-gpu-services.yaml"], check=False
            )

            # Remove custom metrics adapter
            self.run_command(
                [self.kubectl_cmd, "delete", "-f", "ops/k8s/custom-metrics-adapter.yaml"],
                check=False,
            )

            # Remove GPU metrics exporter
            self.run_command(
                [self.kubectl_cmd, "delete", "-f", "ops/k8s/gpu-metrics-exporter.yaml"], check=False
            )

            logger.info("Auto-scaling infrastructure undeployed successfully")

        except Exception as e:
            logger.error(f"Undeployment failed: {e}")
            raise


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Deploy Auto-Scaling Infrastructure")
    parser.add_argument("--namespace", default="medical-kg", help="Kubernetes namespace")
    parser.add_argument(
        "--action",
        choices=["deploy", "undeploy", "status"],
        default="deploy",
        help="Action to perform",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create deployer
    deployer = AutoScalingDeployer(args.namespace)

    try:
        if args.action == "deploy":
            deployer.deploy()
        elif args.action == "undeploy":
            deployer.undeploy()
        elif args.action == "status":
            status = deployer.get_deployment_status()
            print("Deployment Status:")
            print(f"  Namespace: {status['namespace']}")
            print(f"  GPU Metrics Exporter: {status['gpu_metrics_exporter']['status']}")
            print(f"  Custom Metrics Adapter: {status['custom_metrics_adapter']['status']}")
            print(f"  HPA Configurations: {status['hpa_configurations']['status']}")
            if status["hpa_configurations"]["hpa_names"]:
                print(f"  HPA Names: {', '.join(status['hpa_configurations']['hpa_names'])}")

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
