"""Cross-domain adapter registry built on top of Pluggy metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from Medical_KG_rev.adapters.plugins.models import AdapterDomain, AdapterMetadata

from .metadata import DomainAdapterMetadata, as_metadata



class DomainAdapterRegistry:
    """Maintain adapters grouped by domain for quick lookups."""

    def __init__(self) -> None:
        self._by_domain: dict[AdapterDomain, dict[str, DomainAdapterMetadata]] = defaultdict(dict)

    def register(self, metadata: AdapterMetadata) -> DomainAdapterMetadata:
        domain_metadata = as_metadata(metadata)
        domain = domain_metadata.domain
        self._by_domain[domain][domain_metadata.name] = domain_metadata
        return domain_metadata

    def unregister(self, name: str) -> None:
        for domain_map in self._by_domain.values():
            domain_map.pop(name, None)

    def list(self, domain: AdapterDomain | None = None) -> list[DomainAdapterMetadata]:
        if domain is None:
            values: Iterable[DomainAdapterMetadata] = (
                meta for per_domain in self._by_domain.values() for meta in per_domain.values()
            )
        else:
            values = self._by_domain.get(domain, {}).values()
        return sorted(values, key=lambda meta: meta.name)

    def domains(self) -> Mapping[AdapterDomain, tuple[str, ...]]:
        return {domain: tuple(sorted(values.keys())) for domain, values in self._by_domain.items()}
