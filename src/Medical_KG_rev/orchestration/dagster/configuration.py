"""Dagster configuration and resource management."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from Medical_KG_rev.config.settings import AppSettings
from Medical_KG_rev.orchestration.stages.contracts import PipelineState
from Medical_KG_rev.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DagsterConfig:
    """Configuration for Dagster operations."""

    app_settings: AppSettings
    pipeline_resources: dict[str, Any] = field(default_factory=dict)
    job_configs: dict[str, Any] = field(default_factory=dict)
    asset_configs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResourceConfig:
    """Configuration for pipeline resources."""

    name: str
    resource_type: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobConfig:
    """Configuration for Dagster jobs."""

    name: str
    description: str
    resource_defs: dict[str, Any] = field(default_factory=dict)
    config_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetConfig:
    """Configuration for Dagster assets."""

    name: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DagsterConfigurationManager:
    """Manages Dagster configuration and resources."""

    def __init__(self, config: DagsterConfig) -> None:
        """Initialize the configuration manager."""
        self.config = config
        self.logger = get_logger(__name__)

    def create_pipeline_resources(self) -> dict[str, Any]:
        """Create pipeline resources from configuration."""
        resources = {}

        for name, resource_config in self.config.pipeline_resources.items():
            try:
                resource = self._create_resource(resource_config)
                resources[name] = resource
                self.logger.info(f"Created pipeline resource: {name}")
            except Exception as exc:
                self.logger.error(f"Failed to create resource {name}: {exc}")
                continue

        return resources

    def create_job_configs(self) -> dict[str, JobConfig]:
        """Create job configurations."""
        job_configs = {}

        for name, config_data in self.config.job_configs.items():
            job_config = JobConfig(
                name=name,
                description=config_data.get("description", ""),
                resource_defs=config_data.get("resource_defs", {}),
                config_schema=config_data.get("config_schema", {}),
            )
            job_configs[name] = job_config

        return job_configs

    def create_asset_configs(self) -> dict[str, AssetConfig]:
        """Create asset configurations."""
        asset_configs = {}

        for name, config_data in self.config.asset_configs.items():
            asset_config = AssetConfig(
                name=name,
                description=config_data.get("description", ""),
                dependencies=config_data.get("dependencies", []),
                metadata=config_data.get("metadata", {}),
            )
            asset_configs[name] = asset_config

        return asset_configs

    def _create_resource(self, resource_config: dict[str, Any]) -> Any:
        """Create a resource from configuration."""
        resource_type = resource_config.get("type")
        config = resource_config.get("config", {})

        if resource_type == "gpu_service":
            return self._create_gpu_service_resource(config)
        elif resource_type == "embedding_service":
            return self._create_embedding_service_resource(config)
        elif resource_type == "chunking_service":
            return self._create_chunking_service_resource(config)
        else:
            raise ValueError(f"Unknown resource type: {resource_type}")

    def _create_gpu_service_resource(self, config: dict[str, Any]) -> Any:
        """Create GPU service resource."""
        # Mock implementation
        return {"type": "gpu_service", "config": config}

    def _create_embedding_service_resource(self, config: dict[str, Any]) -> Any:
        """Create embedding service resource."""
        # Mock implementation
        return {"type": "embedding_service", "config": config}

    def _create_chunking_service_resource(self, config: dict[str, Any]) -> Any:
        """Create chunking service resource."""
        # Mock implementation
        return {"type": "chunking_service", "config": config}

    def validate_configuration(self) -> bool:
        """Validate the Dagster configuration."""
        try:
            # Validate pipeline resources
            for name, resource_config in self.config.pipeline_resources.items():
                if not isinstance(resource_config, dict):
                    self.logger.error(f"Invalid resource config for {name}")
                    return False

                if "type" not in resource_config:
                    self.logger.error(f"Missing resource type for {name}")
                    return False

            # Validate job configs
            for name, job_config in self.config.job_configs.items():
                if not isinstance(job_config, dict):
                    self.logger.error(f"Invalid job config for {name}")
                    return False

            # Validate asset configs
            for name, asset_config in self.config.asset_configs.items():
                if not isinstance(asset_config, dict):
                    self.logger.error(f"Invalid asset config for {name}")
                    return False

            return True

        except Exception as exc:
            self.logger.error(f"Configuration validation failed: {exc}")
            return False

    def get_resource_config(self, name: str) -> dict[str, Any] | None:
        """Get configuration for a specific resource."""
        return self.config.pipeline_resources.get(name)

    def get_job_config(self, name: str) -> JobConfig | None:
        """Get configuration for a specific job."""
        config_data = self.config.job_configs.get(name)
        if config_data:
            return JobConfig(
                name=name,
                description=config_data.get("description", ""),
                resource_defs=config_data.get("resource_defs", {}),
                config_schema=config_data.get("config_schema", {}),
            )
        return None

    def get_asset_config(self, name: str) -> AssetConfig | None:
        """Get configuration for a specific asset."""
        config_data = self.config.asset_configs.get(name)
        if config_data:
            return AssetConfig(
                name=name,
                description=config_data.get("description", ""),
                dependencies=config_data.get("dependencies", []),
                metadata=config_data.get("metadata", {}),
            )
        return None


def create_default_dagster_config(app_settings: AppSettings) -> DagsterConfig:
    """Create default Dagster configuration."""
    return DagsterConfig(
        app_settings=app_settings,
        pipeline_resources={
            "gpu_service": {
                "type": "gpu_service",
                "config": {
                    "endpoint": "localhost:50051",
                    "timeout": 30,
                },
            },
            "embedding_service": {
                "type": "embedding_service",
                "config": {
                    "endpoint": "localhost:50052",
                    "timeout": 30,
                },
            },
            "chunking_service": {
                "type": "chunking_service",
                "config": {
                    "endpoint": "localhost:50053",
                    "timeout": 30,
                },
            },
        },
        job_configs={
            "ingestion_job": {
                "description": "Document ingestion job",
                "resource_defs": ["gpu_service", "embedding_service"],
                "config_schema": {
                    "batch_size": {"type": "int", "default": 10},
                    "timeout": {"type": "int", "default": 300},
                },
            },
            "embedding_job": {
                "description": "Embedding generation job",
                "resource_defs": ["embedding_service"],
                "config_schema": {
                    "model": {"type": "str", "default": "bert-base-uncased"},
                    "batch_size": {"type": "int", "default": 32},
                },
            },
        },
        asset_configs={
            "document_asset": {
                "description": "Document processing asset",
                "dependencies": [],
                "metadata": {"version": "1.0"},
            },
            "embedding_asset": {
                "description": "Embedding generation asset",
                "dependencies": ["document_asset"],
                "metadata": {"version": "1.0"},
            },
        },
    )
