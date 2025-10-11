"""Factory for creating and managing domain-specific metric registries."""

from __future__ import annotations

from typing import Any, Dict, Type

from Medical_KG_rev.observability.registries.base import BaseMetricRegistry
from Medical_KG_rev.observability.registries.cache import CacheMetricRegistry
from Medical_KG_rev.observability.registries.external_api import ExternalAPIMetricRegistry
from Medical_KG_rev.observability.registries.gpu import GPUMetricRegistry
from Medical_KG_rev.observability.registries.grpc import gRPCMetricRegistry
from Medical_KG_rev.observability.registries.pipeline import PipelineMetricRegistry
from Medical_KG_rev.observability.registries.reranking import RerankingMetricRegistry
from prometheus_client import CollectorRegistry


class MetricRegistryFactory:
    """Factory for creating and managing domain-specific metric registries."""

    _registry_classes: Dict[str, Type[Any]] = {
        "gpu": GPUMetricRegistry,
        "external_api": ExternalAPIMetricRegistry,
        "grpc": gRPCMetricRegistry,
        "pipeline": PipelineMetricRegistry,
        "cache": CacheMetricRegistry,
        "reranking": RerankingMetricRegistry,
    }

    _instances: Dict[str, BaseMetricRegistry] = {}
    _registry: CollectorRegistry | None = None

    @classmethod
    def initialize(cls, registry: CollectorRegistry | None = None) -> None:
        """Initialize the factory with a Prometheus collector registry.

        Args:
            registry: Prometheus collector registry to use for all metric registries
        """
        cls._registry = registry
        cls._instances = {}

    @classmethod
    def get_registry(cls, domain: str) -> BaseMetricRegistry:
        """Get or create a metric registry for the specified domain.

        Args:
            domain: Domain name (gpu, external_api, grpc, pipeline, cache, reranking)

        Returns:
            The metric registry for the domain

        Raises:
            ValueError: If the domain is not supported
        """
        if domain not in cls._registry_classes:
            supported_domains = ", ".join(cls._registry_classes.keys())
            raise ValueError(f"Unsupported domain '{domain}'. Supported domains: {supported_domains}")

        if domain not in cls._instances:
            registry_class = cls._registry_classes[domain]
            cls._instances[domain] = registry_class(registry=cls._registry)

        return cls._instances[domain]

    @classmethod
    def get_gpu_registry(cls) -> GPUMetricRegistry:
        """Get the GPU metric registry.

        Returns:
            The GPU metric registry
        """
        return cls.get_registry("gpu")  # type: ignore

    @classmethod
    def get_external_api_registry(cls) -> ExternalAPIMetricRegistry:
        """Get the External API metric registry.

        Returns:
            The External API metric registry
        """
        return cls.get_registry("external_api")  # type: ignore

    @classmethod
    def get_grpc_registry(cls) -> gRPCMetricRegistry:
        """Get the gRPC metric registry.

        Returns:
            The gRPC metric registry
        """
        return cls.get_registry("grpc")  # type: ignore

    @classmethod
    def get_pipeline_registry(cls) -> PipelineMetricRegistry:
        """Get the Pipeline metric registry.

        Returns:
            The Pipeline metric registry
        """
        return cls.get_registry("pipeline")  # type: ignore

    @classmethod
    def get_cache_registry(cls) -> CacheMetricRegistry:
        """Get the Cache metric registry.

        Returns:
            The Cache metric registry
        """
        return cls.get_registry("cache")  # type: ignore

    @classmethod
    def get_reranking_registry(cls) -> RerankingMetricRegistry:
        """Get the Reranking metric registry.

        Returns:
            The Reranking metric registry
        """
        return cls.get_registry("reranking")  # type: ignore

    @classmethod
    def get_all_registries(cls) -> Dict[str, BaseMetricRegistry]:
        """Get all created metric registries.

        Returns:
            Dictionary mapping domain names to their registries
        """
        return cls._instances.copy()

    @classmethod
    def clear_registries(cls) -> None:
        """Clear all created registries."""
        cls._instances.clear()

    @classmethod
    def supported_domains(cls) -> list[str]:
        """Get list of supported domains.

        Returns:
            List of supported domain names
        """
        return list(cls._registry_classes.keys())


# Track whether the global factory has been initialised.
_global_factory: type[MetricRegistryFactory] | None = None


def get_metric_registry_factory(registry: CollectorRegistry | None = None) -> type[MetricRegistryFactory]:
    """Get the global metric registry factory instance.

    Args:
        registry: Prometheus collector registry to use (only used on first call)

    Returns:
        The global metric registry factory
    """
    global _global_factory
    if _global_factory is None:
        MetricRegistryFactory.initialize(registry=registry)
        _global_factory = MetricRegistryFactory
    return _global_factory


def reset_metric_registry_factory() -> None:
    """Reset the global metric registry factory. Used for testing."""
    global _global_factory
    _global_factory = None
