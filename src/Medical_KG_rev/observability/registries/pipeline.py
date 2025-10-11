"""Pipeline metric registry for orchestration pipeline metrics."""

from Medical_KG_rev.observability.registries.base import BaseMetricRegistry
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class PipelineMetricRegistry(BaseMetricRegistry):
    """Metric registry for orchestration pipeline metrics.

    Scope:
        - Pipeline stage execution metrics
        - Pipeline context and state tracking
        - Stage-to-stage transitions
        - Pipeline-level performance metrics

    Out of Scope:
        - Individual service communication (use gRPCMetricRegistry)
        - GPU hardware metrics (use GPUMetricRegistry)
        - Cache operations (use CacheMetricRegistry)
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize Pipeline metric registry.

        Args:
            registry: Prometheus collector registry to use (default registry if None)
        """
        super().__init__(domain="pipeline", registry=registry)
        self.initialize_collectors()

    def initialize_collectors(self) -> None:
        """Initialize all pipeline-specific metrics."""
        # Pipeline execution counters
        self._collectors["pipeline_executions_total"] = Counter(
            "pipeline_executions_total",
            "Total number of pipeline executions",
            ["pipeline_name", "status"],
            registry=self._registry
        )

        # Stage execution counters
        self._collectors["stage_executions_total"] = Counter(
            "stage_executions_total",
            "Total number of stage executions",
            ["pipeline_name", "stage_name", "status"],
            registry=self._registry
        )

        # Pipeline duration histogram
        self._collectors["pipeline_duration_seconds"] = Histogram(
            "pipeline_duration_seconds",
            "Pipeline execution duration in seconds",
            ["pipeline_name"],
            registry=self._registry
        )

        # Stage duration histogram
        self._collectors["stage_duration_seconds"] = Histogram(
            "stage_duration_seconds",
            "Stage execution duration in seconds",
            ["pipeline_name", "stage_name"],
            registry=self._registry
        )

        # Pipeline context size gauge
        self._collectors["pipeline_context_size"] = Gauge(
            "pipeline_context_size",
            "Size of pipeline context in bytes",
            ["pipeline_name"],
            registry=self._registry
        )

        # Active pipelines gauge
        self._collectors["active_pipelines"] = Gauge(
            "active_pipelines",
            "Number of currently active pipelines",
            registry=self._registry
        )

        # Pipeline queue depth gauge
        self._collectors["pipeline_queue_depth"] = Gauge(
            "pipeline_queue_depth",
            "Number of pipelines waiting in queue",
            registry=self._registry
        )

    def record_pipeline_execution(self, pipeline_name: str, status: str) -> None:
        """Record a pipeline execution.

        Args:
            pipeline_name: Name of the pipeline
            status: Execution status (started, completed, failed)
        """
        self._collectors["pipeline_executions_total"].labels(
            pipeline_name=pipeline_name,
            status=status
        ).inc()

    def record_stage_execution(self, pipeline_name: str, stage_name: str, status: str) -> None:
        """Record a stage execution.

        Args:
            pipeline_name: Name of the pipeline
            stage_name: Name of the stage
            status: Execution status (started, completed, failed)
        """
        self._collectors["stage_executions_total"].labels(
            pipeline_name=pipeline_name,
            stage_name=stage_name,
            status=status
        ).inc()

    def observe_pipeline_duration(self, pipeline_name: str, duration_seconds: float) -> None:
        """Observe pipeline execution duration.

        Args:
            pipeline_name: Name of the pipeline
            duration_seconds: Duration in seconds
        """
        self._collectors["pipeline_duration_seconds"].labels(
            pipeline_name=pipeline_name
        ).observe(duration_seconds)

    def observe_stage_duration(self, pipeline_name: str, stage_name: str, duration_seconds: float) -> None:
        """Observe stage execution duration.

        Args:
            pipeline_name: Name of the pipeline
            stage_name: Name of the stage
            duration_seconds: Duration in seconds
        """
        self._collectors["stage_duration_seconds"].labels(
            pipeline_name=pipeline_name,
            stage_name=stage_name
        ).observe(duration_seconds)

    def set_pipeline_context_size(self, pipeline_name: str, size_bytes: int) -> None:
        """Set pipeline context size.

        Args:
            pipeline_name: Name of the pipeline
            size_bytes: Context size in bytes
        """
        self._collectors["pipeline_context_size"].labels(
            pipeline_name=pipeline_name
        ).set(size_bytes)

    def set_active_pipelines(self, count: int) -> None:
        """Set number of active pipelines.

        Args:
            count: Number of active pipelines
        """
        self._collectors["active_pipelines"].set(count)

    def set_pipeline_queue_depth(self, depth: int) -> None:
        """Set pipeline queue depth.

        Args:
            depth: Number of pipelines in queue
        """
        self._collectors["pipeline_queue_depth"].set(depth)
