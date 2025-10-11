"""External API metric registry for external-facing HTTP traffic."""

from Medical_KG_rev.observability.registries.base import BaseMetricRegistry
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class ExternalAPIMetricRegistry(BaseMetricRegistry):
    """Metric registry for external-facing API gateway and adapter HTTP clients.

    Scope:
        - Client requests to REST/GraphQL/SOAP/OData endpoints
        - Adapter outbound HTTP requests to external APIs
        - External database HTTP client requests

    Out of Scope:
        - Internal gRPC service-to-service communication (use gRPCMetricRegistry)
        - Internal GPU service calls (use GPUMetricRegistry)
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize External API metric registry.

        Args:
            registry: Prometheus collector registry to use (default registry if None)
        """
        super().__init__(domain="external_api", registry=registry)
        self.initialize_collectors()

    def initialize_collectors(self) -> None:
        """Initialize external API-specific Prometheus collectors."""
        self._collectors["client_requests"] = Counter(
            "external_api_requests_total",
            "External client API requests",
            ["protocol", "endpoint", "method", "status_code"],
            registry=self._registry
        )

        self._collectors["adapter_requests"] = Counter(
            "adapter_http_requests_total",
            "Adapter outbound HTTP requests to external APIs",
            ["adapter_name", "target_host", "status_code"],
            registry=self._registry
        )

        self._collectors["request_duration"] = Histogram(
            "external_api_duration_seconds",
            "External API request duration",
            ["protocol", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self._registry
        )

        self._collectors["request_size"] = Histogram(
            "external_api_request_size_bytes",
            "External API request size in bytes",
            ["protocol", "endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000],
            registry=self._registry
        )

        self._collectors["response_size"] = Histogram(
            "external_api_response_size_bytes",
            "External API response size in bytes",
            ["protocol", "endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000],
            registry=self._registry
        )

        self._collectors["rate_limit_hits"] = Counter(
            "external_api_rate_limit_hits_total",
            "External API rate limit hits",
            ["protocol", "endpoint"],
            registry=self._registry
        )

        self._collectors["circuit_breaker_state"] = Gauge(
            "external_api_circuit_breaker_state",
            "External API circuit breaker state (0=closed, 1=open, 2=half-open)",
            ["protocol", "endpoint"],
            registry=self._registry
        )

    def record_client_request(
        self,
        protocol: str,
        endpoint: str,
        method: str,
        status_code: int
    ) -> None:
        """Record external client API request.

        Args:
            protocol: Protocol (REST, GraphQL, SOAP, OData)
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
        """
        self.get_collector("client_requests").labels(
            protocol=protocol,
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()

    def record_adapter_request(
        self,
        adapter_name: str,
        target_host: str,
        status_code: int
    ) -> None:
        """Record adapter outbound HTTP request.

        Args:
            adapter_name: Name of the adapter
            target_host: Target host/API
            status_code: HTTP status code
        """
        self.get_collector("adapter_requests").labels(
            adapter_name=adapter_name,
            target_host=target_host,
            status_code=str(status_code)
        ).inc()

    def observe_request_duration(
        self,
        protocol: str,
        endpoint: str,
        duration: float
    ) -> None:
        """Observe external API request duration.

        Args:
            protocol: Protocol (REST, GraphQL, SOAP, OData)
            endpoint: API endpoint
            duration: Duration in seconds
        """
        self.get_collector("request_duration").labels(
            protocol=protocol,
            endpoint=endpoint
        ).observe(duration)

    def observe_request_size(
        self,
        protocol: str,
        endpoint: str,
        size_bytes: int
    ) -> None:
        """Observe external API request size.

        Args:
            protocol: Protocol (REST, GraphQL, SOAP, OData)
            endpoint: API endpoint
            size_bytes: Request size in bytes
        """
        self.get_collector("request_size").labels(
            protocol=protocol,
            endpoint=endpoint
        ).observe(size_bytes)

    def observe_response_size(
        self,
        protocol: str,
        endpoint: str,
        size_bytes: int
    ) -> None:
        """Observe external API response size.

        Args:
            protocol: Protocol (REST, GraphQL, SOAP, OData)
            endpoint: API endpoint
            size_bytes: Response size in bytes
        """
        self.get_collector("response_size").labels(
            protocol=protocol,
            endpoint=endpoint
        ).observe(size_bytes)

    def record_rate_limit_hit(
        self,
        protocol: str,
        endpoint: str
    ) -> None:
        """Record external API rate limit hit.

        Args:
            protocol: Protocol (REST, GraphQL, SOAP, OData)
            endpoint: API endpoint
        """
        self.get_collector("rate_limit_hits").labels(
            protocol=protocol,
            endpoint=endpoint
        ).inc()

    def set_circuit_breaker_state(
        self,
        protocol: str,
        endpoint: str,
        state: int
    ) -> None:
        """Set external API circuit breaker state.

        Args:
            protocol: Protocol (REST, GraphQL, SOAP, OData)
            endpoint: API endpoint
            state: Circuit breaker state (0=closed, 1=open, 2=half-open)
        """
        self.get_collector("circuit_breaker_state").labels(
            protocol=protocol,
            endpoint=endpoint
        ).set(state)
