"""Domain adapter registries (placeholder)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

from Medical_KG_rev.adapters.plugins.models import AdapterDomain, AdapterMetadata


@dataclass
class DomainAdapterRegistry:
    _registry: Dict[AdapterDomain, Tuple[str, ...]] = field(default_factory=dict)

    def register(self, metadata: AdapterMetadata) -> AdapterMetadata:
        self._registry.setdefault(metadata.domain, tuple())
        return metadata

    def unregister(self, name: str) -> None:
        return None

    def list(self, domain: AdapterDomain) -> Iterable[AdapterMetadata]:
        return []

    def domains(self) -> Dict[AdapterDomain, Tuple[str, ...]]:
        return self._registry


__all__ = ["DomainAdapterRegistry"]
