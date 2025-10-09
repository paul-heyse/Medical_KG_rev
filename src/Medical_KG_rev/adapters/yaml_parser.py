"""Parser for declarative adapter configuration files.

This module provides utilities for parsing YAML-based adapter configurations,
creating adapter instances from declarative specifications, and handling
rate limiting, request/response mapping, and data transformation.

The parser supports:
- YAML configuration validation using Pydantic models
- Dynamic parameter substitution in request paths and parameters
- JSONPath-like data extraction from API responses
- Automatic document construction from mapped data fields
- Rate limiting configuration and enforcement

Responsibilities:
- Parse and validate YAML adapter configuration files
- Create BaseAdapter instances from validated configurations
- Handle parameter substitution and data path resolution
- Transform API responses into Document objects

Collaborators:
- BaseAdapter: Interface for adapter implementations
- ResilientHTTPAdapter: Base class for HTTP-based adapters
- HttpClient: HTTP client for making API requests
- Document, Section, Block: Data models for structured content

Side Effects:
- Reads YAML files from filesystem
- Makes HTTP requests to external APIs
- Creates Document objects with structured content

Thread Safety:
- Configuration parsing is stateless and thread-safe
- Adapter instances are not thread-safe and should not be shared

Performance Characteristics:
- Configuration parsing: O(n) where n is configuration size
- Data path resolution: O(m) where m is path depth
- Document creation: O(k) where k is number of items per response

Example:
    config = load_adapter_config(Path("adapters/config/openalex.yaml"))
    adapter = create_adapter_from_config(config)
    documents = adapter.fetch_and_parse(context)
"""

# IMPORTS

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from Medical_KG_rev.models import Block, Document, Section
from Medical_KG_rev.utils.http_client import BackoffStrategy, HttpClient, RetryConfig

from .base import AdapterContext, BaseAdapter
from .biomedical import ResilientHTTPAdapter

# TYPE DEFINITIONS & CONSTANTS

TOKEN_PATTERN = re.compile(r"[^\[\].]+|\[\d+\]")

# DATA MODELS

@dataclass(frozen=True)
class RateLimitConfig:
    """Configuration for rate limiting adapter requests.

    Attributes:
        requests: Number of requests allowed per time window
        per_seconds: Time window duration in seconds

    Invariants:
        - requests > 0
        - per_seconds > 0

    Example:
        RateLimitConfig(requests=100, per_seconds=60.0)  # 100 requests per minute
    """

    requests: int
    per_seconds: float

    @property
    def rate_per_second(self) -> float:
        """Calculate requests per second rate.

        Returns:
            Rate in requests per second.

        Example:
            >>> config = RateLimitConfig(requests=60, per_seconds=60.0)
            >>> config.rate_per_second
            1.0
        """
        return self.requests / self.per_seconds


@dataclass(frozen=True)
class RequestConfig:
    """Configuration for HTTP request parameters.

    Attributes:
        method: HTTP method (GET, POST, etc.)
        path: Request path with optional parameter placeholders
        params: Query parameters with optional placeholders
        headers: HTTP headers with optional placeholders

    Example:
        RequestConfig(
            method="GET",
            path="/works/{work_id}",
            params={"filter": "open_access.is_oa:true"},
            headers={"User-Agent": "Medical-KG-Rev/1.0"}
        )
    """

    method: str
    path: str
    params: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseConfig:
    """Configuration for parsing API responses.

    Attributes:
        items_path: JSONPath-like path to extract items from response

    Example:
        ResponseConfig(items_path="results")  # Extract from response.results
    """

    items_path: str | None = None


@dataclass(frozen=True)
class MappingConfig:
    """Configuration for mapping API data to document fields.

    Attributes:
        document_id: Path to document identifier in API response
        title: Path to document title (optional)
        summary: Path to document summary/abstract (optional)
        body: Path to document body content (optional)
        metadata: Mapping of metadata keys to API response paths

    Example:
        MappingConfig(
            document_id="id",
            title="title",
            summary="abstract",
            body="full_text",
            metadata={"doi": "doi", "authors": "authors"}
        )
    """

    document_id: str
    title: str | None = None
    summary: str | None = None
    body: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterConfig:
    """Complete configuration for a YAML-configured adapter.

    Attributes:
        name: Adapter name identifier
        source: Data source identifier
        base_url: Base URL for API requests
        request: HTTP request configuration
        response: Response parsing configuration
        mapping: Data mapping configuration
        rate_limit: Rate limiting configuration (optional)

    Example:
        AdapterConfig(
            name="openalex",
            source="OpenAlex",
            base_url="https://api.openalex.org",
            request=RequestConfig(method="GET", path="/works/{work_id}"),
            response=ResponseConfig(items_path="results"),
            mapping=MappingConfig(document_id="id", title="title"),
            rate_limit=RateLimitConfig(requests=100, per_seconds=60.0)
        )
    """

    name: str
    source: str
    base_url: str
    request: RequestConfig
    response: ResponseConfig
    mapping: MappingConfig
    rate_limit: RateLimitConfig | None = None


# PYDANTIC VALIDATION MODELS

class RateLimitModel(BaseModel):
    """Pydantic model for validating rate limit configuration.

    Attributes:
        requests: Number of requests allowed per time window
        per_seconds: Time window duration in seconds

    Validation:
        - requests must be greater than 0
        - per_seconds must be greater than 0
    """

    requests: int = Field(gt=0)
    per_seconds: float = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class RequestModel(BaseModel):
    """Pydantic model for validating request configuration.

    Attributes:
        method: HTTP method (defaults to GET)
        path: Request path with optional parameter placeholders
        params: Query parameters with optional placeholders
        headers: HTTP headers with optional placeholders
    """

    method: str = Field(default="GET")
    path: str
    params: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ResponseModel(BaseModel):
    """Pydantic model for validating response configuration.

    Attributes:
        items_path: JSONPath-like path to extract items from response
    """

    items_path: str | None = None

    model_config = ConfigDict(extra="forbid")


class MappingModel(BaseModel):
    """Pydantic model for validating data mapping configuration.

    Attributes:
        document_id: Path to document identifier (aliased as 'id')
        title: Path to document title (optional)
        summary: Path to document summary/abstract (optional)
        body: Path to document body content (optional)
        metadata: Mapping of metadata keys to API response paths
    """

    document_id: str = Field(alias="id")
    title: str | None = None
    summary: str | None = None
    body: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AdapterConfigModel(BaseModel):
    """Pydantic model for validating complete adapter configuration.

    Attributes:
        name: Adapter name identifier (optional, defaults to filename)
        source: Data source identifier
        base_url: Base URL for API requests
        request: HTTP request configuration
        response: Response parsing configuration
        mapping: Data mapping configuration
        rate_limit: Rate limiting configuration (optional)
    """

    name: str | None = None
    source: str
    base_url: str
    request: RequestModel
    response: ResponseModel = Field(default_factory=ResponseModel)
    mapping: MappingModel
    rate_limit: RateLimitModel | None = None

    model_config = ConfigDict(extra="forbid")


# CONFIGURATION LOADING

def load_adapter_config(path: Path) -> AdapterConfig:
    """Load and validate adapter configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Validated adapter configuration.

    Raises:
        ValueError: If configuration file is empty or invalid.
        ValidationError: If configuration fails Pydantic validation.

    Example:
        >>> config = load_adapter_config(Path("adapters/config/openalex.yaml"))
        >>> print(config.name)
        openalex
    """
    data = yaml.safe_load(path.read_text())
    if not data:
        raise ValueError("Adapter configuration is empty")
    model = AdapterConfigModel.model_validate(data)
    request = RequestConfig(
        method=model.request.method.upper(),
        path=model.request.path,
        params=model.request.params,
        headers=model.request.headers,
    )
    response = ResponseConfig(items_path=model.response.items_path)
    mapping = MappingConfig(
        document_id=model.mapping.document_id,
        title=model.mapping.title,
        summary=model.mapping.summary,
        body=model.mapping.body,
        metadata=model.mapping.metadata,
    )
    rate_limit = (
        RateLimitConfig(
            requests=model.rate_limit.requests, per_seconds=model.rate_limit.per_seconds
        )
        if model.rate_limit
        else None
    )
    return AdapterConfig(
        name=model.name or path.stem,
        source=model.source,
        base_url=model.base_url,
        request=request,
        response=response,
        mapping=mapping,
        rate_limit=rate_limit,
    )


# ADAPTER IMPLEMENTATION

class YAMLConfiguredAdapter(ResilientHTTPAdapter):
    """Adapter generated from a declarative YAML configuration.

    This adapter implements the BaseAdapter interface using configuration-driven
    behavior for HTTP requests, response parsing, and document construction.
    It supports parameter substitution, JSONPath-like data extraction, and
    automatic rate limiting.

    Attributes:
        _config: Validated adapter configuration

    Thread Safety:
        Not thread-safe. Instances should not be shared between threads.

    Example:
        >>> config = load_adapter_config(Path("config.yaml"))
        >>> adapter = YAMLConfiguredAdapter(config)
        >>> documents = adapter.fetch_and_parse(context)
    """

    def __init__(self, config: AdapterConfig, client: HttpClient | None = None) -> None:
        """Initialize adapter with configuration.

        Args:
            config: Validated adapter configuration
            client: Optional HTTP client (creates default if None)

        Example:
            >>> config = AdapterConfig(...)
            >>> adapter = YAMLConfiguredAdapter(config)
        """
        rate = config.rate_limit.rate_per_second if config.rate_limit else 5.0
        super().__init__(
            name=config.name,
            base_url=config.base_url,
            rate_limit_per_second=rate,
            retry=RetryConfig(
                attempts=3,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                backoff_initial=0.5,
                backoff_max=4.0,
            ),
            client=client,
        )
        self._config = config

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        """Fetch data from configured API endpoint.

        Args:
            context: Adapter context with parameters for substitution

        Returns:
            Iterable of raw API response items

        Raises:
            HTTPError: If API request fails
            ValueError: If required parameters are missing

        Example:
            >>> context = AdapterContext(parameters={"work_id": "W123"})
            >>> items = adapter.fetch(context)
        """
        formatter = _FormatDict(context.parameters)
        path = formatter.format(self._config.request.path)
        params = _format_structure(self._config.request.params, formatter)
        headers = _format_structure(self._config.request.headers, formatter)
        response = self._client.request(
            self._config.request.method,
            path,
            params=params or None,
            headers=headers or None,
        )
        response.raise_for_status()
        data = response.json()
        return _resolve_items(data, self._config.response.items_path)

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        """Parse raw API payloads into Document objects.

        Args:
            payloads: Raw API response items
            context: Adapter context (unused but required by interface)

        Returns:
            Sequence of parsed Document objects

        Example:
            >>> payloads = [{"id": "W123", "title": "Test Paper"}]
            >>> documents = adapter.parse(payloads, context)
        """
        documents: list[Document] = []
        for payload in payloads:
            document_id = _resolve_path(payload, self._config.mapping.document_id)
            if document_id is None:
                continue
            title_value = (
                _resolve_path(payload, self._config.mapping.title)
                if self._config.mapping.title
                else None
            )
            summary_value = (
                _resolve_path(payload, self._config.mapping.summary)
                if self._config.mapping.summary
                else None
            )
            body_value = (
                _resolve_path(payload, self._config.mapping.body)
                if self._config.mapping.body
                else None
            )
            metadata = {
                key: _resolve_path(payload, path)
                for key, path in self._config.mapping.metadata.items()
            }
            metadata = {key: value for key, value in metadata.items() if value is not None}

            sections: list[Section] = []
            if summary_value is not None:
                sections.append(
                    Section(
                        id="summary",
                        title="Summary",
                        blocks=[Block(id="summary-block", text=_to_text(summary_value), spans=[])],
                    )
                )
            if body_value is not None:
                sections.append(
                    Section(
                        id="body",
                        title="Body",
                        blocks=[Block(id="body-block", text=_to_text(body_value), spans=[])],
                    )
                )
            if not sections:
                sections.append(
                    Section(
                        id="data",
                        title="Data",
                        blocks=[
                            Block(
                                id="data-block",
                                text=_to_text(json.dumps(payload, default=str)),
                                spans=[],
                            )
                        ],
                    )
                )

            documents.append(
                Document(
                    id=_to_text(document_id),
                    source=self._config.source,
                    title=_to_text(title_value) or None,
                    sections=sections,
                    metadata=metadata,
                )
            )
        return documents


# FACTORY FUNCTIONS

def create_adapter_from_config(
    config: AdapterConfig, client: HttpClient | None = None
) -> BaseAdapter:
    """Instantiate an adapter from a validated configuration.

    Args:
        config: Validated adapter configuration
        client: Optional HTTP client (creates default if None)

    Returns:
        Configured adapter instance

    Example:
        >>> config = load_adapter_config(Path("config.yaml"))
        >>> adapter = create_adapter_from_config(config)
    """
    return YAMLConfiguredAdapter(config, client=client)

# PRIVATE HELPERS

class _FormatDict(dict):
    """Helper mapping that raises clear errors for missing keys.

    This class extends dict to provide parameter substitution with
    clear error messages when required parameters are missing.
    """

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        """Initialize with parameter mapping.

        Args:
            parameters: Parameter values for substitution
        """
        super().__init__(parameters)

    def __missing__(self, key: str) -> str:
        """Raise error for missing parameters.

        Args:
            key: Missing parameter name

        Raises:
            ValueError: Always raised for missing parameters
        """
        raise ValueError(f"Missing required parameter '{key}' for adapter configuration")

    def format(self, template: str) -> str:
        """Format template string with parameters.

        Args:
            template: String template with {parameter} placeholders

        Returns:
            Formatted string with substituted values
        """
        return template.format_map(self)


def _format_structure(value: Any, formatter: _FormatDict) -> Any:
    """Recursively format nested data structures.

    Args:
        value: Value to format (string, dict, list, or other)
        formatter: Parameter formatter instance

    Returns:
        Formatted value with parameter substitution
    """
    if isinstance(value, str):
        return formatter.format(value)
    if isinstance(value, Mapping):
        return {key: _format_structure(subvalue, formatter) for key, subvalue in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_format_structure(item, formatter) for item in value]
    return value


def _resolve_items(payload: Any, path: str | None) -> Sequence[Mapping[str, Any]]:
    """Extract items from API response using JSONPath-like syntax.

    Args:
        payload: Raw API response data
        path: JSONPath-like path to items (optional)

    Returns:
        Sequence of item mappings

    Example:
        >>> data = {"results": [{"id": 1}, {"id": 2}]}
        >>> items = _resolve_items(data, "results")
        >>> len(items)
        2
    """
    if path is None:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, Mapping):
            return [payload]
        return []
    resolved = _resolve_path(payload, path)
    if resolved is None:
        return []
    if isinstance(resolved, list):
        return resolved
    if isinstance(resolved, Mapping):
        return [resolved]
    return []


def _resolve_path(data: Any, path: str | None) -> Any:
    """Resolve value from nested data structure using JSONPath-like syntax.

    Args:
        data: Data structure to traverse
        path: JSONPath-like path (e.g., "results[0].title")

    Returns:
        Resolved value or None if path not found

    Example:
        >>> data = {"results": [{"title": "Test"}]}
        >>> value = _resolve_path(data, "results[0].title")
        >>> value
        "Test"
    """
    if path is None:
        return data
    current: Any = data
    for token in TOKEN_PATTERN.findall(path):
        if token.startswith("["):
            index = int(token[1:-1])
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
                return None
            if index >= len(current):
                return None
            current = current[index]
        else:
            if isinstance(current, Mapping):
                current = current.get(token)
            elif isinstance(current, Sequence) and token.isdigit():
                idx = int(token)
                if idx >= len(current):
                    return None
                current = current[idx]
            else:
                return None
        if current is None:
            return None
    return current


def _to_text(value: Any) -> str:
    """Convert any value to string representation.

    Args:
        value: Value to convert

    Returns:
        String representation of value

    Example:
        >>> _to_text(123)
        "123"
        >>> _to_text(None)
        ""
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)

# EXPORTS
