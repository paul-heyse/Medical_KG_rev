"""YAML-based adapter configuration parser."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from Medical_KG_rev.adapters.base import AdapterContext, BaseAdapter

# from Medical_KG_rev.adapters.plugins.config import AdapterConfig  # Not implemented
from Medical_KG_rev.utils.http_client import HttpClient

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class AdapterConfig:
    name: str
    type: str
    base_url: str
    description: str = ''
    parameters: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_config: dict[str, Any] = field(default_factory=dict)
    rate_limit_config: dict[str, Any] = field(default_factory=dict)
    circuit_breaker_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = self.type.strip()
        self.base_url = self.base_url.strip()



class YAMLParser:
    """Parser for YAML-based adapter configurations."""

    def __init__(self) -> None:
        """Initialize the YAML parser."""
        self.logger = logger

    def parse_config(self, yaml_content: str) -> AdapterConfig:
        """Parse YAML content into adapter configuration."""
        try:
            data = yaml.safe_load(yaml_content)
            return self._validate_config(data)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Configuration parsing failed: {exc}") from exc

    def parse_file(self, file_path: Path) -> AdapterConfig:
        """Parse YAML file into adapter configuration."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return self.parse_config(content)
        except FileNotFoundError as exc:
            raise ValueError(f"Configuration file not found: {file_path}") from exc
        except Exception as exc:
            raise ValueError(f"Failed to read configuration file: {exc}") from exc

    def _validate_config(self, data: dict[str, Any]) -> AdapterConfig:
        """Validate and create adapter configuration."""
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a dictionary")

        # Required fields
        required_fields = ["name", "type", "base_url"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Create configuration
        config = AdapterConfig(
            name=data["name"],
            type=data["type"],
            base_url=data["base_url"],
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            headers=data.get("headers", {}),
            timeout=data.get("timeout", 30),
            retry_config=data.get("retry_config", {}),
            rate_limit_config=data.get("rate_limit_config", {}),
            circuit_breaker_config=data.get("circuit_breaker_config", {}),
        )

        return config

    def validate_config(self, config: AdapterConfig) -> bool:
        """Validate adapter configuration."""
        try:
            # Check required fields
            if not config.name:
                raise ValueError("Adapter name is required")

            if not config.type:
                raise ValueError("Adapter type is required")

            if not config.base_url:
                raise ValueError("Base URL is required")

            # Validate URL format
            if not config.base_url.startswith(('http://', 'https://')):
                raise ValueError("Base URL must start with http:// or https://")

            # Validate timeout
            if config.timeout <= 0:
                raise ValueError("Timeout must be positive")

            return True

        except Exception as exc:
            self.logger.error(f"Configuration validation failed: {exc}")
            return False

    def create_adapter(self, config: AdapterConfig, client: HttpClient | None = None) -> BaseAdapter:
        """Create adapter instance from configuration."""
        try:
            # Validate configuration
            if not self.validate_config(config):
                raise ValueError("Invalid adapter configuration")

            # Create HTTP client if not provided
            if client is None:
                client = HttpClient(
                    base_url=config.base_url,
                    timeout=config.timeout,
                    headers=config.headers,
                )

            # Create adapter based on type
            adapter = self._create_adapter_by_type(config, client)

            self.logger.info(f"Created adapter: {config.name} ({config.type})")
            return adapter

        except Exception as exc:
            self.logger.error(f"Failed to create adapter: {exc}")
            raise exc

    def _create_adapter_by_type(self, config: AdapterConfig, client: HttpClient) -> BaseAdapter:
        """Create adapter instance based on type."""
        adapter_type = config.type.lower()

        if adapter_type == "http":
            return self._create_http_adapter(config, client)
        if adapter_type == "rest":
            return self._create_rest_adapter(config, client)
        if adapter_type == "graphql":
            return self._create_graphql_adapter(config, client)
        raise ValueError(f"Unsupported adapter type: {adapter_type}")

    def _create_http_adapter(self, config: AdapterConfig, client: HttpClient) -> BaseAdapter:
        """Create HTTP adapter."""

        class HTTPAdapter(BaseAdapter):
            def __init__(self, name: str, client: HttpClient):
                super().__init__(name)
                self.client = client

            def fetch(self, context: AdapterContext) -> list[dict[str, Any]]:
                return []

            def parse(self, payloads: list[dict[str, Any]], context: AdapterContext) -> list[Any]:
                return []

            def write(self, documents: list[Any], context: AdapterContext) -> None:
                return None

        return HTTPAdapter(config.name, client)

    def _create_rest_adapter(self, config: AdapterConfig, client: HttpClient) -> BaseAdapter:
        """Create REST adapter."""
        # Mock implementation
        class RESTAdapter(BaseAdapter):
            def __init__(self, name: str, client: HttpClient):
                super().__init__(name)
                self.client = client

            def fetch(self, context: AdapterContext) -> list[dict[str, Any]]:
                return []

            def parse(self, payloads: list[dict[str, Any]], context: AdapterContext) -> list[Any]:
                return []

            def write(self, documents: list[Any], context: AdapterContext) -> None:
                return None

        return RESTAdapter(config.name, client)

    def _create_graphql_adapter(self, config: AdapterConfig, client: HttpClient) -> BaseAdapter:
        """Create GraphQL adapter."""
        # Mock implementation
        class GraphQLAdapter(BaseAdapter):
            def __init__(self, name: str, client: HttpClient):
                super().__init__(name)
                self.client = client

            def fetch(self, context: AdapterContext) -> list[dict[str, Any]]:
                return []

            def parse(self, payloads: list[dict[str, Any]], context: AdapterContext) -> list[Any]:
                return []

            def write(self, documents: list[Any], context: AdapterContext) -> None:
                return None

        return GraphQLAdapter(config.name, client)


def load_adapter_config(file_path: Path) -> AdapterConfig:
    """Load adapter configuration from YAML file."""
    parser = YAMLParser()
    return parser.parse_file(file_path)


def create_adapter_from_config(
    config: AdapterConfig,
    client: HttpClient | None = None
) -> BaseAdapter:
    """Create adapter instance from configuration."""
    parser = YAMLParser()
    return parser.create_adapter(config, client)


def validate_yaml_config(yaml_content: str) -> bool:
    """Validate YAML configuration content."""
    parser = YAMLParser()
    try:
        config = parser.parse_config(yaml_content)
        return parser.validate_config(config)
    except Exception:
        return False


def parse_yaml_config(yaml_content: str) -> AdapterConfig:
    """Parse YAML configuration content."""
    parser = YAMLParser()
    return parser.parse_config(yaml_content)
