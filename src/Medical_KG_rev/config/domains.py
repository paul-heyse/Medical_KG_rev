"""Simple YAML-backed domain configuration models."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None

if _YAML_AVAILABLE:  # pragma: no cover - exercised in environments with PyYAML
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


@dataclass(slots=True)
class DomainConfig:
    """Single domain configuration entry."""

    id: str
    description: str
    adapters: dict[str, str] = field(default_factory=dict)
    default_features: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DomainConfig:
        return cls(
            id=str(data["id"]),
            description=str(data.get("description", "")),
            adapters=dict(data.get("adapters", {})),
            default_features={k: bool(v) for k, v in dict(data.get("default_features", {})).items()},
        )


@dataclass(slots=True)
class DomainRegistry:
    """Collection of domain configurations loaded from YAML."""

    domains: dict[str, DomainConfig] = field(default_factory=dict)

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
        domains: dict[str, DomainConfig] = {}
        for item in data.get("domains", []):
            if not isinstance(item, Mapping):  # pragma: no cover - defensive
                continue
            config = DomainConfig.from_mapping(item)
            domains[config.id] = config
        return cls(domains=domains)


__all__ = ["DomainConfig", "DomainRegistry", "yaml"]
