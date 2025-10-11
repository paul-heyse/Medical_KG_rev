"""Migration utilities for domain-specific metric registries.

This module provides utilities to migrate from legacy Prometheus metrics to
domain-specific metric registries for improved observability clarity and
reduced label cardinality.

Key Responsibilities:
    - Route metrics to appropriate domain-specific registries based on feature flags
    - Provide backward compatibility during migration period
    - Abstract metric recording logic from domain-specific implementation details
    - Support gradual rollout of new metric collection patterns

Collaborators:
    - Upstream: Code that needs to record metrics (services, coordinators, etc.)
    - Downstream: Domain-specific metric registries (GPU, gRPC, External API, etc.)
    - Configuration: Feature flags that control registry usage

Side Effects:
    - Routes metrics to appropriate registries based on feature flags
    - May emit metrics to different registries depending on configuration

Thread Safety:
    - Thread-safe: All registry operations are atomic
    - Concurrent calls to migration helper methods are safe

Performance Characteristics:
    - Low overhead: Simple routing logic with minimal performance impact
    - Memory usage: No additional memory allocation beyond registry operations
    - No external dependencies or blocking operations

Example:
    >>> from Medical_KG_rev.observability.metrics_migration import get_migration_helper
    >>> from Medical_KG_rev.config.settings import get_settings
    >>>
    >>> settings = get_settings()
    >>> helper = get_migration_helper(settings)
    >>>
    >>> # Records to appropriate domain registry based on feature flags
    >>> helper.record_external_api_call("GET", "/api/v1/search", "200")
    >>> helper.record_pipeline_stage("pipeline", "embedding", "completed", 1.5)

"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

from typing import Any

from Medical_KG_rev.config.settings import AppSettings, get_settings
from Medical_KG_rev.observability.registries.factory import get_metric_registry_factory

# ==============================================================================
# DATA MODELS
# ==============================================================================

class MetricsMigrationHelper:
    """Helper class to migrate metrics to domain-specific registries.

    This class provides a unified interface for recording metrics that automatically
    routes to appropriate domain-specific registries based on feature flags. It
    enables gradual migration from legacy metrics to the new domain-specific
    registry system.

    Key Features:
        - Feature flag-based routing between legacy and new metrics
        - Domain-specific metric recording methods
        - Backward compatibility during migration
        - Centralized metric recording logic

    Attributes:
        settings: Application settings containing feature flags
        use_domain_registries: Whether to use domain-specific registries
        factory: Factory for creating metric registries

    Thread Safety:
        - Thread-safe: Registry operations are atomic
        - Concurrent access to helper instance is safe

    Example:
        >>> helper = MetricsMigrationHelper(settings)
        >>> helper.record_external_api_call("GET", "/api/v1/search", "200")
        >>> helper.record_pipeline_stage("pipeline", "embedding", "completed", 1.5)

    """

    def __init__(self, settings: AppSettings) -> None:
        """Initialize the migration helper.

        Sets up the migration helper with application settings and registry factory.

        Args:
            settings: Application settings containing feature flags for metric routing

        Note:
            The helper automatically determines whether to use domain-specific
            registries based on the domain_specific_metric_registries feature flag.

        """
        self.settings = settings
        self.use_domain_registries = settings.feature_flags.domain_specific_metric_registries
        self.factory = get_metric_registry_factory()


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

    def record_gpu_hardware_metric(
        self,
        device_id: str,
        device_name: str,
        metric_type: str,
        value: float,
        **kwargs: Any
    ) -> None:
        """Record GPU hardware metrics in the appropriate registry.

        Args:
            device_id: GPU device identifier
            device_name: Human-readable device name
            metric_type: Type of metric (memory_usage, utilization, temperature)
            value: Metric value
            **kwargs: Additional metric-specific parameters

        """
        if self.use_domain_registries:
            gpu_registry = self.factory.get_gpu_registry()

            if metric_type == "memory_usage":
                gpu_registry.set_memory_usage(device_id, device_name, int(value))
            elif metric_type == "utilization":
                gpu_registry.set_utilization(device_id, device_name, value)
            elif metric_type == "temperature":
                gpu_registry.set_temperature(device_id, device_name, value)
            elif metric_type == "device_status":
                gpu_registry.set_health_status(device_id, device_name, kwargs.get("status") == "healthy")
        else:
            # Fallback to legacy metrics for backward compatibility
            from Medical_KG_rev.observability.metrics import (
                GPU_MEMORY_USAGE_MB,
                GPU_SERVICE_HEALTH_STATUS,
                GPU_UTILIZATION,
            )

            if metric_type == "memory_usage":
                GPU_MEMORY_USAGE_MB.labels(device_id=device_id, device_name=device_name).set(value)
            elif metric_type == "utilization":
                GPU_UTILIZATION.labels(device_id=device_id, device_name=device_name).set(value)
            elif metric_type == "device_status":
                GPU_SERVICE_HEALTH_STATUS.labels(service_name=f"{device_id}_{device_name}").set(
                    1 if kwargs.get("status") == "healthy" else 0
                )

    def record_gpu_service_call(
        self,
        service: str,
        method: str,
        status: str,
        duration_seconds: float | None = None,
        **kwargs: Any
    ) -> None:
        """Record GPU service call metrics in the appropriate registry.

        Args:
            service: Service name
            method: Method name
            status: Call status (success, error, timeout)
            duration_seconds: Call duration in seconds
            **kwargs: Additional call-specific parameters

        Returns:
            None

        Raises:
            None

        """
        if self.use_domain_registries:
            gpu_registry = self.factory.get_gpu_registry()
            gpu_registry.record_gpu_service_call(service, method, status)

            if duration_seconds is not None:
                gpu_registry.observe_gpu_service_duration(service, method, duration_seconds)
        else:
            # Fallback to legacy metrics
            from Medical_KG_rev.observability.metrics import (
                GPU_SERVICE_CALL_DURATION_SECONDS,
                GPU_SERVICE_CALLS_TOTAL,
            )

            GPU_SERVICE_CALLS_TOTAL.labels(service=service, method=method, status=status).inc()
            if duration_seconds is not None:
                GPU_SERVICE_CALL_DURATION_SECONDS.labels(service=service, method=method).observe(duration_seconds)

    def record_external_api_call(
        self,
        protocol: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration_seconds: float | None = None,
        **kwargs: Any
    ) -> None:
        """Record external API call metrics in the appropriate registry.

        Args:
            protocol: Protocol type (REST, GraphQL, SOAP, OData)
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
            duration_seconds: Call duration in seconds
            **kwargs: Additional call-specific parameters

        """
        if self.use_domain_registries:
            external_api_registry = self.factory.get_external_api_registry()
            external_api_registry.record_client_request(protocol, endpoint, method, status_code)

            if duration_seconds is not None:
                external_api_registry.observe_request_duration(protocol, endpoint, duration_seconds)
        else:
            # Fallback to legacy metrics (using GPU metrics incorrectly)
            from Medical_KG_rev.observability.metrics import (
                GPU_SERVICE_CALL_DURATION_SECONDS,
                GPU_SERVICE_CALLS_TOTAL,
            )

            GPU_SERVICE_CALLS_TOTAL.labels(service="gateway", method=f"{protocol}_{endpoint}", status=str(status_code)).inc()
            if duration_seconds is not None:
                GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="gateway", method=f"{protocol}_{endpoint}").observe(duration_seconds)

    def record_pipeline_stage(
        self,
        pipeline_name: str,
        stage_name: str,
        status: str,
        duration_seconds: float | None = None,
        **kwargs: Any
    ) -> None:
        """Record pipeline stage metrics in the appropriate registry.

        Args:
            pipeline_name: Name of the pipeline
            stage_name: Name of the stage
            status: Stage status (started, completed, failed)
            duration_seconds: Stage duration in seconds
            **kwargs: Additional stage-specific parameters

        """
        if self.use_domain_registries:
            pipeline_registry = self.factory.get_pipeline_registry()
            pipeline_registry.record_stage_execution(pipeline_name, stage_name, status)

            if duration_seconds is not None:
                pipeline_registry.observe_stage_duration(pipeline_name, stage_name, duration_seconds)
        else:
            # Fallback to legacy metrics (using GPU metrics incorrectly)
            from Medical_KG_rev.observability.metrics import (
                GPU_SERVICE_CALL_DURATION_SECONDS,
                GPU_SERVICE_CALLS_TOTAL,
            )

            GPU_SERVICE_CALLS_TOTAL.labels(service="pipeline", method=stage_name, status=status).inc()
            if duration_seconds is not None:
                GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="pipeline", method=stage_name).observe(duration_seconds)

    def record_cache_operation(
        self,
        cache_name: str,
        operation: str,
        status: str,
        duration_seconds: float | None = None,
        **kwargs: Any
    ) -> None:
        """Record cache operation metrics in the appropriate registry.

        Args:
            cache_name: Name of the cache
            operation: Cache operation (get, set, delete, clear)
            status: Operation status (success, error, hit, miss)
            duration_seconds: Operation duration in seconds
            **kwargs: Additional operation-specific parameters

        """
        if self.use_domain_registries:
            cache_registry = self.factory.get_cache_registry()
            cache_registry.record_cache_operation(cache_name, operation, status)

            if duration_seconds is not None:
                cache_registry.observe_cache_operation_duration(cache_name, operation, duration_seconds)
        else:
            # Fallback to legacy metrics (using GPU metrics incorrectly)
            from Medical_KG_rev.observability.metrics import (
                GPU_SERVICE_CALL_DURATION_SECONDS,
                GPU_SERVICE_CALLS_TOTAL,
            )

            GPU_SERVICE_CALLS_TOTAL.labels(service="cache", method=f"{cache_name}_{operation}", status=status).inc()
            if duration_seconds is not None:
                GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="cache", method=f"{cache_name}_{operation}").observe(duration_seconds)

    def record_reranking_operation(
        self,
        reranker_type: str,
        status: str,
        duration_seconds: float | None = None,
        **kwargs: Any
    ) -> None:
        """Record reranking operation metrics in the appropriate registry.

        Records reranking operation metrics, routing to the reranking registry
        when domain-specific registries are enabled, or falling back to legacy
        metrics for backward compatibility.

        Args:
            reranker_type: Type of reranker (lexical, semantic, hybrid)
            status: Operation status (started, completed, failed)
            duration_seconds: Operation duration in seconds
            **kwargs: Additional operation-specific parameters

        Note:
            When domain registries are enabled, routes to RerankingMetricRegistry.
            When disabled, falls back to legacy GPU metrics (incorrectly categorized).

        """
        if self.use_domain_registries:
            reranking_registry = self.factory.get_reranking_registry()
            reranking_registry.record_reranking_operation(reranker_type, status)

            if duration_seconds is not None:
                reranking_registry.observe_reranking_duration(reranker_type, duration_seconds)
        else:
            # Fallback to legacy metrics (using GPU metrics incorrectly)
            from Medical_KG_rev.observability.metrics import (
                GPU_SERVICE_CALL_DURATION_SECONDS,
                GPU_SERVICE_CALLS_TOTAL,
            )

            GPU_SERVICE_CALLS_TOTAL.labels(service="reranking", method=reranker_type, status=status).inc()
            if duration_seconds is not None:
                GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="reranking", method=reranker_type).observe(duration_seconds)


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def get_migration_helper(settings: AppSettings | None = None) -> MetricsMigrationHelper:
    """Get a configured metrics migration helper instance.

    Creates and returns a MetricsMigrationHelper instance configured with
    the current application settings. This factory function provides a
    convenient way to get a properly configured helper without manually
    passing settings.

    Args:
        settings: Optional application settings. If None, uses get_settings().

    Returns:
        Configured MetricsMigrationHelper instance ready for use

    Raises:
        RuntimeError: If settings cannot be loaded or are invalid

    Example:
        >>> helper = get_migration_helper()
        >>> helper.record_external_api_call("GET", "/api/v1/search", "200")

    """
    if settings is None:
        settings = get_settings()
    return MetricsMigrationHelper(settings)


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "MetricsMigrationHelper",
    "get_migration_helper",
]
