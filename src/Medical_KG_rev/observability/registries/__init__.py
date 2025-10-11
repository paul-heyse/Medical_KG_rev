"""Domain-specific metric registries for improved observability."""

from Medical_KG_rev.observability.registries.base import BaseMetricRegistry
from Medical_KG_rev.observability.registries.cache import CacheMetricRegistry
from Medical_KG_rev.observability.registries.external_api import ExternalAPIMetricRegistry
from Medical_KG_rev.observability.registries.factory import (
    MetricRegistryFactory,
    get_metric_registry_factory,
    reset_metric_registry_factory,
)
from Medical_KG_rev.observability.registries.gpu import GPUMetricRegistry
from Medical_KG_rev.observability.registries.grpc import gRPCMetricRegistry
from Medical_KG_rev.observability.registries.pipeline import PipelineMetricRegistry
from Medical_KG_rev.observability.registries.reranking import RerankingMetricRegistry

__all__ = [
    "BaseMetricRegistry",
    "GPUMetricRegistry",
    "ExternalAPIMetricRegistry",
    "gRPCMetricRegistry",
    "PipelineMetricRegistry",
    "CacheMetricRegistry",
    "RerankingMetricRegistry",
    "MetricRegistryFactory",
    "get_metric_registry_factory",
    "reset_metric_registry_factory",
]
