"""Cache metric registry for caching layer metrics."""

from Medical_KG_rev.observability.registries.base import BaseMetricRegistry
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class CacheMetricRegistry(BaseMetricRegistry):
    """Metric registry for caching layer metrics.

    Scope:
        - Cache hit/miss ratios
        - Cache operation performance
        - Cache memory usage
        - Cache eviction metrics

    Out of Scope:
        - Service communication metrics (use gRPCMetricRegistry)
        - Pipeline metrics (use PipelineMetricRegistry)
        - GPU metrics (use GPUMetricRegistry)
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize Cache metric registry.

        Args:
            registry: Prometheus collector registry to use (default registry if None)
        """
        super().__init__(domain="cache", registry=registry)
        self.initialize_collectors()

    def initialize_collectors(self) -> None:
        """Initialize all cache-specific metrics."""
        # Cache operation counters
        self._collectors["cache_operations_total"] = Counter(
            "cache_operations_total",
            "Total number of cache operations",
            ["cache_name", "operation", "status"],
            registry=self._registry
        )

        # Cache hit/miss counters
        self._collectors["cache_hits_total"] = Counter(
            "cache_hits_total",
            "Total number of cache hits",
            ["cache_name"],
            registry=self._registry
        )

        self._collectors["cache_misses_total"] = Counter(
            "cache_misses_total",
            "Total number of cache misses",
            ["cache_name"],
            registry=self._registry
        )

        # Cache operation duration histogram
        self._collectors["cache_operation_duration_seconds"] = Histogram(
            "cache_operation_duration_seconds",
            "Cache operation duration in seconds",
            ["cache_name", "operation"],
            registry=self._registry
        )

        # Cache size gauge
        self._collectors["cache_size_bytes"] = Gauge(
            "cache_size_bytes",
            "Cache size in bytes",
            ["cache_name"],
            registry=self._registry
        )

        # Cache entry count gauge
        self._collectors["cache_entries"] = Gauge(
            "cache_entries",
            "Number of entries in cache",
            ["cache_name"],
            registry=self._registry
        )

        # Cache eviction counter
        self._collectors["cache_evictions_total"] = Counter(
            "cache_evictions_total",
            "Total number of cache evictions",
            ["cache_name", "reason"],
            registry=self._registry
        )

        # Cache memory usage gauge
        self._collectors["cache_memory_usage_bytes"] = Gauge(
            "cache_memory_usage_bytes",
            "Cache memory usage in bytes",
            ["cache_name"],
            registry=self._registry
        )

    def record_cache_operation(self, cache_name: str, operation: str, status: str) -> None:
        """Record a cache operation.

        Args:
            cache_name: Name of the cache
            operation: Operation type (get, set, delete, clear)
            status: Operation status (success, error)
        """
        self._collectors["cache_operations_total"].labels(
            cache_name=cache_name,
            operation=operation,
            status=status
        ).inc()

    def record_cache_hit(self, cache_name: str) -> None:
        """Record a cache hit.

        Args:
            cache_name: Name of the cache
        """
        self._collectors["cache_hits_total"].labels(
            cache_name=cache_name
        ).inc()

    def record_cache_miss(self, cache_name: str) -> None:
        """Record a cache miss.

        Args:
            cache_name: Name of the cache
        """
        self._collectors["cache_misses_total"].labels(
            cache_name=cache_name
        ).inc()

    def observe_cache_operation_duration(self, cache_name: str, operation: str, duration_seconds: float) -> None:
        """Observe cache operation duration.

        Args:
            cache_name: Name of the cache
            operation: Operation type
            duration_seconds: Duration in seconds
        """
        self._collectors["cache_operation_duration_seconds"].labels(
            cache_name=cache_name,
            operation=operation
        ).observe(duration_seconds)

    def set_cache_size(self, cache_name: str, size_bytes: int) -> None:
        """Set cache size.

        Args:
            cache_name: Name of the cache
            size_bytes: Size in bytes
        """
        self._collectors["cache_size_bytes"].labels(
            cache_name=cache_name
        ).set(size_bytes)

    def set_cache_entries(self, cache_name: str, count: int) -> None:
        """Set cache entry count.

        Args:
            cache_name: Name of the cache
            count: Number of entries
        """
        self._collectors["cache_entries"].labels(
            cache_name=cache_name
        ).set(count)

    def record_cache_eviction(self, cache_name: str, reason: str) -> None:
        """Record a cache eviction.

        Args:
            cache_name: Name of the cache
            reason: Eviction reason (size_limit, ttl_expired, manual)
        """
        self._collectors["cache_evictions_total"].labels(
            cache_name=cache_name,
            reason=reason
        ).inc()

    def set_cache_memory_usage(self, cache_name: str, usage_bytes: int) -> None:
        """Set cache memory usage.

        Args:
            cache_name: Name of the cache
            usage_bytes: Memory usage in bytes
        """
        self._collectors["cache_memory_usage_bytes"].labels(
            cache_name=cache_name
        ).set(usage_bytes)
