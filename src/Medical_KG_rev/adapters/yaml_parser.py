"""Parser for declarative adapter configuration files."""

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
from Medical_KG_rev.utils.http_client import HttpClient

from .base import AdapterContext, BaseAdapter
from .biomedical import ResilientHTTPAdapter

TOKEN_PATTERN = re.compile(r"[^\[\].]+|\[\d+\]")


@dataclass(frozen=True)
class RateLimitConfig:
    requests: int
    per_seconds: float

    @property
    def rate_per_second(self) -> float:
        return self.requests / self.per_seconds


@dataclass(frozen=True)
class RequestConfig:
    method: str
    path: str
    params: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseConfig:
    items_path: str | None = None


@dataclass(frozen=True)
class MappingConfig:
    document_id: str
    title: str | None = None
    summary: str | None = None
    body: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterConfig:
    name: str
    source: str
    base_url: str
    request: RequestConfig
    response: ResponseConfig
    mapping: MappingConfig
    rate_limit: RateLimitConfig | None = None


class RateLimitModel(BaseModel):
    requests: int = Field(gt=0)
    per_seconds: float = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class RequestModel(BaseModel):
    method: str = Field(default="GET")
    path: str
    params: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ResponseModel(BaseModel):
    items_path: str | None = None

    model_config = ConfigDict(extra="forbid")


class MappingModel(BaseModel):
    document_id: str = Field(alias="id")
    title: str | None = None
    summary: str | None = None
    body: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AdapterConfigModel(BaseModel):
    name: str | None = None
    source: str
    base_url: str
    request: RequestModel
    response: ResponseModel = Field(default_factory=ResponseModel)
    mapping: MappingModel
    rate_limit: RateLimitModel | None = None

    model_config = ConfigDict(extra="forbid")


def load_adapter_config(path: Path) -> AdapterConfig:
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


class YAMLConfiguredAdapter(ResilientHTTPAdapter):
    """Adapter generated from a declarative configuration."""

    def __init__(self, config: AdapterConfig, client: HttpClient | None = None) -> None:
        rate = config.rate_limit.rate_per_second if config.rate_limit else 5.0
        super().__init__(
            name=config.name,
            base_url=config.base_url,
            rate_limit_per_second=rate,
            retry_attempts=3,
            backoff_factor=0.5,
            client=client,
        )
        self._config = config

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
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


def create_adapter_from_config(
    config: AdapterConfig, client: HttpClient | None = None
) -> BaseAdapter:
    """Instantiate an adapter from a validated configuration."""

    return YAMLConfiguredAdapter(config, client=client)


class _FormatDict(dict):
    """Helper mapping that raises clear errors for missing keys."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        super().__init__(parameters)

    def __missing__(self, key: str) -> str:
        raise ValueError(f"Missing required parameter '{key}' for adapter configuration")

    def format(self, template: str) -> str:
        return template.format_map(self)


def _format_structure(value: Any, formatter: _FormatDict) -> Any:
    if isinstance(value, str):
        return formatter.format(value)
    if isinstance(value, Mapping):
        return {key: _format_structure(subvalue, formatter) for key, subvalue in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_format_structure(item, formatter) for item in value]
    return value


def _resolve_items(payload: Any, path: str | None) -> Sequence[Mapping[str, Any]]:
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
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
