"""gRPC metric registry for internal service-to-service communication."""

from Medical_KG_rev.observability.registries.base import BaseMetricRegistry
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class gRPCMetricRegistry(BaseMetricRegistry):
    """Metric registry for internal gRPC service-to-service communication.

    Scope:
        - Gateway → GPU services (embedding, reranking, Docling VLM)
        - Gateway → Orchestration services
        - Orchestration → GPU services
        - Any internal Docker container communication

    Out of Scope:
        - External client API requests (use ExternalAPIMetricRegistry)
        - GPU-specific metrics like memory/utilization (use GPUMetricRegistry)
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize gRPC metric registry.

        Args:
            registry: Prometheus collector registry to use (default registry if None)
        """
        super().__init__(domain="grpc", registry=registry)
        self.initialize_collectors()

    def initialize_collectors(self) -> None:
        """Initialize gRPC-specific Prometheus collectors."""
        self._collectors["rpc_calls"] = Counter(
            "grpc_calls_total",
            "Internal gRPC calls",
            ["service", "method", "status_code"],
            registry=self._registry
        )

        self._collectors["rpc_duration"] = Histogram(
            "grpc_call_duration_seconds",
            "gRPC call duration",
            ["service", "method"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self._registry
        )

        self._collectors["rpc_errors"] = Counter(
            "grpc_errors_total",
            "gRPC call errors",
            ["service", "method", "error_code"],
            registry=self._registry
        )

        self._collectors["stream_messages"] = Counter(
            "grpc_stream_messages_total",
            "gRPC streaming messages",
            ["service", "method", "direction"],  # direction: sent/received
            registry=self._registry
        )

        self._collectors["circuit_breaker_state"] = Gauge(
            "grpc_circuit_breaker_state",
            "gRPC circuit breaker state (0=closed, 1=open, 2=half-open)",
            ["service", "method"],
            registry=self._registry
        )

        self._collectors["connection_pool_size"] = Gauge(
            "grpc_connection_pool_size",
            "gRPC connection pool size",
            ["service", "endpoint"],
            registry=self._registry
        )

        self._collectors["connection_pool_active"] = Gauge(
            "grpc_connection_pool_active",
            "Active gRPC connections in pool",
            ["service", "endpoint"],
            registry=self._registry
        )

        self._collectors["retry_attempts"] = Counter(
            "grpc_retry_attempts_total",
            "gRPC retry attempts",
            ["service", "method", "attempt_number"],
            registry=self._registry
        )

    def record_rpc_call(
        self,
        service: str,
        method: str,
        status_code: str
    ) -> None:
        """Record gRPC RPC call.

        Args:
            service: Service name (e.g., "embedding", "reranking")
            method: Method name (e.g., "BatchEmbed", "ScorePairs")
            status_code: gRPC status code (e.g., "OK", "UNAVAILABLE")
        """
        self.get_collector("rpc_calls").labels(
            service=service,
            method=method,
            status_code=status_code
        ).inc()

    def observe_rpc_duration(
        self,
        service: str,
        method: str,
        duration: float
    ) -> None:
        """Observe gRPC RPC call duration.

        Args:
            service: Service name
            method: Method name
            duration: Duration in seconds
        """
        self.get_collector("rpc_duration").labels(
            service=service,
            method=method
        ).observe(duration)

    def record_rpc_error(
        self,
        service: str,
        method: str,
        error_code: str
    ) -> None:
        """Record gRPC RPC error.

        Args:
            service: Service name
            method: Method name
            error_code: gRPC error code (e.g., "UNAVAILABLE", "DEADLINE_EXCEEDED")
        """
        self.get_collector("rpc_errors").labels(
            service=service,
            method=method,
            error_code=error_code
        ).inc()

    def record_stream_message(
        self,
        service: str,
        method: str,
        direction: str
    ) -> None:
        """Record gRPC streaming message.

        Args:
            service: Service name
            method: Method name
            direction: Direction ("sent" or "received")
        """
        self.get_collector("stream_messages").labels(
            service=service,
            method=method,
            direction=direction
        ).inc()

    def set_circuit_breaker_state(
        self,
        service: str,
        method: str,
        state: int
    ) -> None:
        """Set gRPC circuit breaker state.

        Args:
            service: Service name
            method: Method name
            state: Circuit breaker state (0=closed, 1=open, 2=half-open)
        """
        self.get_collector("circuit_breaker_state").labels(
            service=service,
            method=method
        ).set(state)

    def set_connection_pool_size(
        self,
        service: str,
        endpoint: str,
        size: int
    ) -> None:
        """Set gRPC connection pool size.

        Args:
            service: Service name
            endpoint: Endpoint address
            size: Pool size
        """
        self.get_collector("connection_pool_size").labels(
            service=service,
            endpoint=endpoint
        ).set(size)

    def set_connection_pool_active(
        self,
        service: str,
        endpoint: str,
        active: int
    ) -> None:
        """Set active gRPC connections in pool.

        Args:
            service: Service name
            endpoint: Endpoint address
            active: Number of active connections
        """
        self.get_collector("connection_pool_active").labels(
            service=service,
            endpoint=endpoint
        ).set(active)

    def record_retry_attempt(
        self,
        service: str,
        method: str,
        attempt_number: int
    ) -> None:
        """Record gRPC retry attempt.

        Args:
            service: Service name
            method: Method name
            attempt_number: Attempt number (1, 2, 3, etc.)
        """
        self.get_collector("retry_attempts").labels(
            service=service,
            method=method,
            attempt_number=str(attempt_number)
        ).inc()
