"""Utilities for managing multi-domain configuration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import importlib.util
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

_YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None

if _YAML_AVAILABLE:
    from yaml import safe_load as _safe_load  # type: ignore
else:  # pragma: no cover - optional dependency fallback
    def _safe_load(_: str) -> Mapping[str, Any]:
        logger.warning(
            "config.yaml.unavailable",
            message="PyYAML not installed; using empty domain registry",
        )
        return {}


class _YamlFacade:
    @staticmethod
    def safe_load(content: str) -> Mapping[str, Any]:
        return _safe_load(content)


yaml = _YamlFacade()


class DomainConfig(BaseModel):
    """Single domain configuration entry."""

    id: str
    description: str
    adapters: Mapping[str, str] = Field(default_factory=dict)
    default_features: Mapping[str, bool] = Field(default_factory=dict)


class DomainRegistry(BaseModel):
    """Collection of domain configurations loaded from YAML."""

    domains: dict[str, DomainConfig]

    def get(self, domain_id: str) -> DomainConfig:
        try:
            return self.domains[domain_id]
        except KeyError as exc:
            raise KeyError(f"Domain '{domain_id}' is not registered") from exc

    def __iter__(self) -> Iterator[DomainConfig]:
        return iter(self.domains.values())

    @classmethod
    def from_path(cls, path: Path) -> DomainRegistry:
        raw = yaml.safe_load(path.read_text()) if path.exists() else {}
        data = raw or {}
        domains = {
            item["id"]: DomainConfig.model_validate(item) for item in data.get("domains", [])
        }
        return cls(domains=domains)
