"""Utilities for managing multi-domain configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Mapping

import yaml
from pydantic import BaseModel, Field


class DomainConfig(BaseModel):
    """Single domain configuration entry."""

    id: str
    description: str
    adapters: Mapping[str, str] = Field(default_factory=dict)
    default_features: Mapping[str, bool] = Field(default_factory=dict)


class DomainRegistry(BaseModel):
    """Collection of domain configurations loaded from YAML."""

    domains: Dict[str, DomainConfig]

    def get(self, domain_id: str) -> DomainConfig:
        try:
            return self.domains[domain_id]
        except KeyError as exc:
            raise KeyError(f"Domain '{domain_id}' is not registered") from exc

    def __iter__(self) -> Iterator[DomainConfig]:
        return iter(self.domains.values())

    @classmethod
    def from_path(cls, path: Path) -> "DomainRegistry":
        raw = yaml.safe_load(path.read_text()) if path.exists() else {}
        data = raw or {}
        domains = {item["id"]: DomainConfig.model_validate(item) for item in data.get("domains", [])}
        return cls(domains=domains)
