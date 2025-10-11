"""Reranking metric registry for search result reranking metrics."""

from Medical_KG_rev.observability.registries.base import BaseMetricRegistry
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class RerankingMetricRegistry(BaseMetricRegistry):
    """Metric registry for search result reranking metrics.

    Scope:
        - Reranking operation performance
        - Reranking model metrics
        - Search result quality metrics
        - Reranking pipeline metrics

    Out of Scope:
        - GPU hardware metrics (use GPUMetricRegistry)
        - Service communication metrics (use gRPCMetricRegistry)
        - Cache operations (use CacheMetricRegistry)
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize Reranking metric registry.

        Args:
            registry: Prometheus collector registry to use (default registry if None)
        """
        super().__init__(domain="reranking", registry=registry)
        self.initialize_collectors()

    def initialize_collectors(self) -> None:
        """Initialize all reranking-specific metrics."""
        # Reranking operation counters
        self._collectors["reranking_operations_total"] = Counter(
            "reranking_operations_total",
            "Total number of reranking operations",
            ["reranker_type", "status"],
            registry=self._registry
        )

        # Reranking duration histogram
        self._collectors["reranking_duration_seconds"] = Histogram(
            "reranking_duration_seconds",
            "Reranking operation duration in seconds",
            ["reranker_type"],
            registry=self._registry
        )

        # Reranking model performance metrics
        self._collectors["reranking_model_inference_time_seconds"] = Histogram(
            "reranking_model_inference_time_seconds",
            "Reranking model inference time in seconds",
            ["model_name", "model_version"],
            registry=self._registry
        )

        # Search result quality metrics
        self._collectors["reranking_score_distribution"] = Histogram(
            "reranking_score_distribution",
            "Distribution of reranking scores",
            ["reranker_type"],
            registry=self._registry
        )

        # Reranking pipeline metrics
        self._collectors["reranking_pipeline_stage_duration_seconds"] = Histogram(
            "reranking_pipeline_stage_duration_seconds",
            "Reranking pipeline stage duration in seconds",
            ["stage_name"],
            registry=self._registry
        )

        # Active reranking operations gauge
        self._collectors["active_reranking_operations"] = Gauge(
            "active_reranking_operations",
            "Number of currently active reranking operations",
            registry=self._registry
        )

        # Reranking queue depth gauge
        self._collectors["reranking_queue_depth"] = Gauge(
            "reranking_queue_depth",
            "Number of reranking operations waiting in queue",
            registry=self._registry
        )

        # Reranking error counter
        self._collectors["reranking_errors_total"] = Counter(
            "reranking_errors_total",
            "Total number of reranking errors",
            ["reranker_type", "error_type"],
            registry=self._registry
        )

        # Reranking batch size histogram
        self._collectors["reranking_batch_size"] = Histogram(
            "reranking_batch_size",
            "Distribution of reranking batch sizes",
            ["reranker_type"],
            registry=self._registry
        )

    def record_reranking_operation(self, reranker_type: str, status: str) -> None:
        """Record a reranking operation.

        Args:
            reranker_type: Type of reranker (lexical, semantic, hybrid)
            status: Operation status (started, completed, failed)
        """
        self._collectors["reranking_operations_total"].labels(
            reranker_type=reranker_type,
            status=status
        ).inc()

    def observe_reranking_duration(self, reranker_type: str, duration_seconds: float) -> None:
        """Observe reranking operation duration.

        Args:
            reranker_type: Type of reranker
            duration_seconds: Duration in seconds
        """
        self._collectors["reranking_duration_seconds"].labels(
            reranker_type=reranker_type
        ).observe(duration_seconds)

    def observe_model_inference_time(self, model_name: str, model_version: str, duration_seconds: float) -> None:
        """Observe reranking model inference time.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            duration_seconds: Inference time in seconds
        """
        self._collectors["reranking_model_inference_time_seconds"].labels(
            model_name=model_name,
            model_version=model_version
        ).observe(duration_seconds)

    def observe_reranking_score(self, reranker_type: str, score: float) -> None:
        """Observe a reranking score.

        Args:
            reranker_type: Type of reranker
            score: Reranking score
        """
        self._collectors["reranking_score_distribution"].labels(
            reranker_type=reranker_type
        ).observe(score)

    def observe_pipeline_stage_duration(self, stage_name: str, duration_seconds: float) -> None:
        """Observe reranking pipeline stage duration.

        Args:
            stage_name: Name of the pipeline stage
            duration_seconds: Duration in seconds
        """
        self._collectors["reranking_pipeline_stage_duration_seconds"].labels(
            stage_name=stage_name
        ).observe(duration_seconds)

    def set_active_reranking_operations(self, count: int) -> None:
        """Set number of active reranking operations.

        Args:
            count: Number of active operations
        """
        self._collectors["active_reranking_operations"].set(count)

    def set_reranking_queue_depth(self, depth: int) -> None:
        """Set reranking queue depth.

        Args:
            depth: Number of operations in queue
        """
        self._collectors["reranking_queue_depth"].set(depth)

    def record_reranking_error(self, reranker_type: str, error_type: str) -> None:
        """Record a reranking error.

        Args:
            reranker_type: Type of reranker
            error_type: Type of error (model_error, timeout, validation_error)
        """
        self._collectors["reranking_errors_total"].labels(
            reranker_type=reranker_type,
            error_type=error_type
        ).inc()

    def observe_reranking_batch_size(self, reranker_type: str, batch_size: int) -> None:
        """Observe reranking batch size.

        Args:
            reranker_type: Type of reranker
            batch_size: Size of the batch
        """
        self._collectors["reranking_batch_size"].labels(
            reranker_type=reranker_type
        ).observe(batch_size)
