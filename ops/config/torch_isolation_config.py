#!/usr/bin/env python3
"""Deployment configuration management for torch isolation.

This module provides configuration management for torch isolation deployment,
including environment-specific settings, service endpoints, validation rules,
and rollback procedures.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, validator


class ServiceConfig(BaseModel):
    """Configuration for a single service."""

    name: str = Field(..., description="Service name")
    replicas: int = Field(default=1, ge=1, le=10, description="Number of replicas")
    resources: dict[str, str] = Field(default_factory=dict, description="Resource requirements")
    port: int = Field(default=8000, ge=1, le=65535, description="Service port")
    health_check: dict[str, Any] = Field(
        default_factory=dict, description="Health check configuration"
    )
    scaling: dict[str, Any] = Field(default_factory=dict, description="Auto-scaling configuration")
    model_cache: dict[str, Any] | None = Field(
        default=None, description="Model cache configuration"
    )

    @validator("resources")
    def validate_resources(cls, v):
        """Validate resource requirements."""
        if v:
            # Validate CPU format (e.g., "1000m", "1")
            if "cpu" in v:
                cpu = v["cpu"]
                if not (cpu.endswith("m") and cpu[:-1].isdigit()) and not cpu.isdigit():
                    raise ValueError(f"Invalid CPU format: {cpu}")

            # Validate memory format (e.g., "2Gi", "2048Mi")
            if "memory" in v:
                memory = v["memory"]
                if not (memory.endswith(("Gi", "Mi", "G", "M")) and memory[:-2].isdigit()):
                    raise ValueError(f"Invalid memory format: {memory}")

        return v


class GatewayConfig(BaseModel):
    """Configuration for the main gateway."""

    replicas: int = Field(default=3, ge=1, le=10, description="Number of replicas")
    resources: dict[str, str] = Field(default_factory=dict, description="Resource requirements")
    port: int = Field(default=8000, ge=1, le=65535, description="Gateway port")
    health_check: dict[str, Any] = Field(
        default_factory=dict, description="Health check configuration"
    )
    scaling: dict[str, Any] = Field(default_factory=dict, description="Auto-scaling configuration")


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and observability."""

    enabled: bool = Field(default=True, description="Enable monitoring")
    metrics_endpoint: str = Field(default="/metrics", description="Metrics endpoint")
    health_endpoint: str = Field(default="/health", description="Health endpoint")
    prometheus: dict[str, Any] = Field(default_factory=dict, description="Prometheus configuration")
    grafana: dict[str, Any] = Field(default_factory=dict, description="Grafana configuration")
    alerting: dict[str, Any] = Field(default_factory=dict, description="Alerting configuration")


class SecurityConfig(BaseModel):
    """Configuration for security settings."""

    mtls: dict[str, Any] = Field(default_factory=dict, description="mTLS configuration")
    network_policies: dict[str, Any] = Field(default_factory=dict, description="Network policies")
    pod_security_policy: dict[str, Any] = Field(
        default_factory=dict, description="Pod security policy"
    )


class StorageConfig(BaseModel):
    """Configuration for storage settings."""

    persistent_volumes: dict[str, Any] = Field(
        default_factory=dict, description="Persistent volume configuration"
    )
    model_cache: dict[str, Any] = Field(
        default_factory=dict, description="Model cache configuration"
    )
    logs: dict[str, Any] = Field(default_factory=dict, description="Log storage configuration")


class NetworkConfig(BaseModel):
    """Configuration for network settings."""

    service_mesh: dict[str, Any] = Field(
        default_factory=dict, description="Service mesh configuration"
    )
    load_balancer: dict[str, Any] = Field(
        default_factory=dict, description="Load balancer configuration"
    )
    ingress: dict[str, Any] = Field(default_factory=dict, description="Ingress configuration")


class DeploymentConfig(BaseModel):
    """Configuration for deployment settings."""

    strategy: str = Field(default="blue_green", description="Deployment strategy")
    environment: str = Field(default="production", description="Target environment")
    namespace: str = Field(default="medical-kg", description="Kubernetes namespace")
    timeout: int = Field(default=1800, ge=60, le=3600, description="Deployment timeout in seconds")
    health_check_interval: int = Field(
        default=30, ge=5, le=300, description="Health check interval in seconds"
    )
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    rollback_on_failure: bool = Field(
        default=True, description="Enable automatic rollback on failure"
    )
    validation_required: bool = Field(
        default=True, description="Require validation before deployment"
    )

    @validator("strategy")
    def validate_strategy(cls, v):
        """Validate deployment strategy."""
        valid_strategies = ["blue_green", "rolling", "recreate"]
        if v not in valid_strategies:
            raise ValueError(f"Invalid deployment strategy: {v}. Must be one of {valid_strategies}")
        return v

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment."""
        valid_environments = ["development", "staging", "production"]
        if v not in valid_environments:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_environments}")
        return v


class TorchIsolationConfig(BaseModel):
    """Main configuration for torch isolation deployment."""

    deployment: DeploymentConfig = Field(
        default_factory=DeploymentConfig, description="Deployment configuration"
    )
    services: dict[str, ServiceConfig] = Field(
        default_factory=dict, description="Service configurations"
    )
    gateway: GatewayConfig = Field(
        default_factory=GatewayConfig, description="Gateway configuration"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring configuration"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig, description="Security configuration"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig, description="Storage configuration"
    )
    network: NetworkConfig = Field(
        default_factory=NetworkConfig, description="Network configuration"
    )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TorchIsolationConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_env(cls) -> "TorchIsolationConfig":
        """Load configuration from environment variables."""
        # Load base configuration
        config_path = os.getenv(
            "TORCH_ISOLATION_CONFIG_PATH", "ops/config/torch_isolation_deployment.yaml"
        )

        try:
            config = cls.from_yaml(config_path)
        except FileNotFoundError:
            # Create default configuration if file doesn't exist
            config = cls()

        # Override with environment variables
        if os.getenv("TORCH_ISOLATION_ENVIRONMENT"):
            config.deployment.environment = os.getenv("TORCH_ISOLATION_ENVIRONMENT")

        if os.getenv("TORCH_ISOLATION_NAMESPACE"):
            config.deployment.namespace = os.getenv("TORCH_ISOLATION_NAMESPACE")

        if os.getenv("TORCH_ISOLATION_TIMEOUT"):
            config.deployment.timeout = int(os.getenv("TORCH_ISOLATION_TIMEOUT"))

        if os.getenv("TORCH_ISOLATION_STRATEGY"):
            config.deployment.strategy = os.getenv("TORCH_ISOLATION_STRATEGY")

        return config

    def to_yaml(self, output_path: str | Path) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)

    def get_service_config(self, service_name: str) -> ServiceConfig | None:
        """Get configuration for a specific service."""
        return self.services.get(service_name)

    def get_environment_config(self) -> dict[str, Any]:
        """Get environment-specific configuration overrides."""
        env = self.deployment.environment

        # Default configurations for each environment
        env_configs = {
            "development": {
                "services": {
                    "gpu_management": {"replicas": 1},
                    "embedding_service": {"replicas": 1},
                    "reranking_service": {"replicas": 1},
                    "docling_vlm_service": {"replicas": 1},
                },
                "gateway": {"replicas": 1},
                "monitoring": {"enabled": False},
                "security": {"mtls": {"enabled": False}},
            },
            "staging": {
                "services": {
                    "gpu_management": {"replicas": 1},
                    "embedding_service": {"replicas": 2},
                    "reranking_service": {"replicas": 1},
                    "docling_vlm_service": {"replicas": 1},
                },
                "gateway": {"replicas": 2},
                "monitoring": {"enabled": True},
                "security": {"mtls": {"enabled": True}},
            },
            "production": {
                "services": {
                    "gpu_management": {"replicas": 2},
                    "embedding_service": {"replicas": 3},
                    "reranking_service": {"replicas": 2},
                    "docling_vlm_service": {"replicas": 2},
                },
                "gateway": {"replicas": 3},
                "monitoring": {"enabled": True},
                "security": {"mtls": {"enabled": True}},
                "deployment": {
                    "strategy": "blue_green",
                    "validation_required": True,
                    "rollback_on_failure": True,
                },
            },
        }

        return env_configs.get(env, {})

    def apply_environment_overrides(self) -> "TorchIsolationConfig":
        """Apply environment-specific configuration overrides."""
        env_config = self.get_environment_config()

        # Apply service overrides
        if "services" in env_config:
            for service_name, overrides in env_config["services"].items():
                if service_name in self.services:
                    # Update existing service config
                    current_config = self.services[service_name].dict()
                    current_config.update(overrides)
                    self.services[service_name] = ServiceConfig(**current_config)
                else:
                    # Create new service config
                    self.services[service_name] = ServiceConfig(name=service_name, **overrides)

        # Apply gateway overrides
        if "gateway" in env_config:
            gateway_config = self.gateway.dict()
            gateway_config.update(env_config["gateway"])
            self.gateway = GatewayConfig(**gateway_config)

        # Apply monitoring overrides
        if "monitoring" in env_config:
            monitoring_config = self.monitoring.dict()
            monitoring_config.update(env_config["monitoring"])
            self.monitoring = MonitoringConfig(**monitoring_config)

        # Apply security overrides
        if "security" in env_config:
            security_config = self.security.dict()
            security_config.update(env_config["security"])
            self.security = SecurityConfig(**security_config)

        # Apply deployment overrides
        if "deployment" in env_config:
            deployment_config = self.deployment.dict()
            deployment_config.update(env_config["deployment"])
            self.deployment = DeploymentConfig(**deployment_config)

        return self

    def validate_configuration(self) -> list[str]:
        """Validate the configuration and return any errors."""
        errors = []

        # Validate service configurations
        for service_name, service_config in self.services.items():
            try:
                # Check if service has required resources
                if not service_config.resources:
                    errors.append(f"Service {service_name} has no resource requirements")

                # Check if service has health check configuration
                if not service_config.health_check:
                    errors.append(f"Service {service_name} has no health check configuration")

                # Check if service has scaling configuration
                if not service_config.scaling:
                    errors.append(f"Service {service_name} has no scaling configuration")

            except Exception as e:
                errors.append(f"Service {service_name} configuration error: {e}")

        # Validate gateway configuration
        try:
            if not self.gateway.health_check:
                errors.append("Gateway has no health check configuration")

            if not self.gateway.scaling:
                errors.append("Gateway has no scaling configuration")

        except Exception as e:
            errors.append(f"Gateway configuration error: {e}")

        # Validate monitoring configuration
        if self.monitoring.enabled:
            if not self.monitoring.prometheus:
                errors.append("Monitoring is enabled but Prometheus configuration is missing")

            if not self.monitoring.grafana:
                errors.append("Monitoring is enabled but Grafana configuration is missing")

        # Validate security configuration
        if self.security.mtls.get("enabled", False):
            required_cert_paths = [
                "ca_cert_path",
                "ca_key_path",
                "service_cert_path",
                "service_key_path",
            ]
            for path_key in required_cert_paths:
                if not self.security.mtls.get(path_key):
                    errors.append(f"mTLS is enabled but {path_key} is missing")

        # Validate storage configuration
        if self.storage.persistent_volumes.get("enabled", False):
            if not self.storage.persistent_volumes.get("storage_class"):
                errors.append("Persistent volumes are enabled but storage class is missing")

        return errors

    def get_manifest_paths(self) -> dict[str, str]:
        """Get Kubernetes manifest paths for services."""
        return {
            "gpu_management": "ops/k8s/gpu-management-service.yaml",
            "embedding_service": "ops/k8s/embedding-service.yaml",
            "reranking_service": "ops/k8s/reranking-service.yaml",
            "docling_vlm_service": "ops/k8s/docling-vlm-service.yaml",
            "gateway": "ops/k8s/gateway-deployment-torch-free.yaml",
            "hpa": "ops/k8s/hpa-gpu-services.yaml",
            "monitoring": "ops/k8s/gpu-metrics-exporter.yaml",
        }

    def get_validation_rules(self) -> dict[str, Any]:
        """Get validation rules for deployment."""
        return {
            "torch_free_gateway": {
                "description": "Validate that the main gateway is torch-free",
                "timeout": 60,
                "retries": 3,
            },
            "gpu_service_functionality": {
                "description": "Validate GPU service functionality",
                "timeout": 120,
                "retries": 3,
            },
            "embedding_service_functionality": {
                "description": "Validate embedding service functionality",
                "timeout": 120,
                "retries": 3,
            },
            "reranking_service_functionality": {
                "description": "Validate reranking service functionality",
                "timeout": 120,
                "retries": 3,
            },
            "docling_vlm_service_functionality": {
                "description": "Validate Docling VLM service functionality",
                "timeout": 180,
                "retries": 3,
            },
            "service_communication": {
                "description": "Validate service communication and error handling",
                "timeout": 60,
                "retries": 3,
            },
            "deployment_health": {
                "description": "Validate overall deployment health",
                "timeout": 300,
                "retries": 3,
            },
        }

    def get_rollback_procedures(self) -> dict[str, Any]:
        """Get rollback procedures for deployment."""
        return {
            "automatic_rollback": {
                "enabled": self.deployment.rollback_on_failure,
                "triggers": ["health_check_failure", "service_unavailable", "validation_failure"],
                "timeout": 300,
            },
            "manual_rollback": {
                "steps": [
                    "Stop new deployments",
                    "Scale down new services",
                    "Scale up previous services",
                    "Validate rollback success",
                    "Update service endpoints",
                ],
                "timeout": 600,
            },
            "emergency_rollback": {
                "steps": [
                    "Immediate service shutdown",
                    "Restore previous configuration",
                    "Emergency health checks",
                    "Service restart",
                    "Post-rollback validation",
                ],
                "timeout": 180,
            },
        }


def load_config(config_path: str | None = None) -> TorchIsolationConfig:
    """Load torch isolation configuration."""
    if config_path:
        return TorchIsolationConfig.from_yaml(config_path)
    else:
        return TorchIsolationConfig.from_env()


def validate_config(config: TorchIsolationConfig) -> bool:
    """Validate torch isolation configuration."""
    errors = config.validate_configuration()

    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = None

    config = load_config(config_path)
    config = config.apply_environment_overrides()

    if validate_config(config):
        print("✅ Configuration is valid")
        sys.exit(0)
    else:
        print("❌ Configuration validation failed")
        sys.exit(1)
