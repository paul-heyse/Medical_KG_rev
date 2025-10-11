"""Abstract base class for domain-specific Prometheus metric registries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class BaseMetricRegistry(ABC):
    """Abstract base class for domain-specific Prometheus metric registries."""

    _domain: str
    _collectors: dict[str, Counter | Gauge | Histogram]

    def __init__(self, domain: str | None = None, registry: CollectorRegistry | None = None):
        """Initialize the metric registry.

        Args:
            domain: Domain name for this registry (e.g., "gpu", "grpc", "external_api")
            registry: Prometheus collector registry to use (default registry if None)
        """
        self._domain = domain or "unknown"
        self._collectors = {}
        self._registry = registry  # Use default registry if None

    @abstractmethod
    def initialize_collectors(self) -> None:
        """Initialize domain-specific Prometheus collectors.

        Subclasses must implement this method to define their specific
        metric collectors (Counter, Gauge, Histogram instances).
        """
        ...

    def get_collector(self, name: str) -> Counter | Gauge | Histogram:
        """Retrieve a registered collector by name.

        Args:
            name: Name of the collector to retrieve

        Returns:
            The requested collector instance

        Raises:
            KeyError: If collector with given name is not found
        """
        if name not in self._collectors:
            raise KeyError(f"Collector '{name}' not found in {self._domain} registry")
        return self._collectors[name]

    @property
    def domain(self) -> str:
        """Get the domain name for this registry.

        Returns:
            Domain name string
        """
        return self._domain
