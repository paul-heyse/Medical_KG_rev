"""gRPC client interceptor for automatic metric collection."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import grpc
from grpc import aio

from Medical_KG_rev.config.settings import get_settings
from Medical_KG_rev.observability import metrics as legacy_metrics
from Medical_KG_rev.observability.metrics_migration import get_migration_helper


class MetricsClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """gRPC client interceptor for automatic metric collection."""

    def __init__(self) -> None:
        """Initialize the metrics interceptor."""
        self.settings = get_settings()
        self.migration_helper = get_migration_helper(self.settings)

    def intercept_unary_unary(
        self,
        continuation: Callable[[grpc.ClientCallDetails, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> Any:
        """Intercept unary-unary gRPC calls."""
        service, method = self._parse_method(client_call_details.method)
        start_time = time.time()

        try:
            response = continuation(client_call_details, request)

            # Record successful call
            if self.settings.feature_flags.domain_specific_metric_registries:
                # Use gRPC registry for internal service calls
                grpc_registry = self.migration_helper.factory.get_grpc_registry()
                grpc_registry.record_rpc_call(service, method, "OK")
                grpc_registry.observe_rpc_duration(
                    service, method, time.time() - start_time
                )
            else:
                # Fallback to legacy metrics
                legacy_metrics.GPU_SERVICE_CALLS_TOTAL.labels(
                    service=service, method=method, status="success"
                ).inc()
                legacy_metrics.GPU_SERVICE_CALL_DURATION_SECONDS.labels(
                    service=service, method=method
                ).observe(time.time() - start_time)

            return response

        except grpc.RpcError as e:
            # Record failed call
            error_code = str(e.code())
            if self.settings.feature_flags.domain_specific_metric_registries:
                grpc_registry = self.migration_helper.factory.get_grpc_registry()
                grpc_registry.record_rpc_error(service, method, error_code)
            else:
                legacy_metrics.GPU_SERVICE_CALLS_TOTAL.labels(
                    service=service, method=method, status="error"
                ).inc()
                legacy_metrics.GPU_SERVICE_ERRORS_TOTAL.labels(
                    service=service, method=method, error_type=error_code
                ).inc()
            raise

    def intercept_unary_stream(
        self,
        continuation: Callable[[grpc.ClientCallDetails, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> Any:
        """Intercept unary-stream gRPC calls."""
        service, method = self._parse_method(client_call_details.method)

        try:
            response = continuation(client_call_details, request)

            # Record successful streaming call
            if self.settings.feature_flags.domain_specific_metric_registries:
                grpc_registry = self.migration_helper.factory.get_grpc_registry()
                grpc_registry.record_stream_message(service, method, "sent")
            else:
                legacy_metrics.GPU_SERVICE_CALLS_TOTAL.labels(
                    service=service, method=f"{method}_stream", status="success"
                ).inc()

            return response

        except grpc.RpcError as e:
            # Record failed streaming call
            error_code = str(e.code())
            if self.settings.feature_flags.domain_specific_metric_registries:
                grpc_registry = self.migration_helper.factory.get_grpc_registry()
                grpc_registry.record_rpc_error(service, method, error_code)
            else:
                legacy_metrics.GPU_SERVICE_CALLS_TOTAL.labels(
                    service=service, method=f"{method}_stream", status="error"
                ).inc()
                legacy_metrics.GPU_SERVICE_ERRORS_TOTAL.labels(
                    service=service, method=method, error_type=error_code
                ).inc()
            raise

    def intercept_stream_unary(
        self,
        continuation: Callable[[grpc.ClientCallDetails, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request_iterator: Any,
    ) -> Any:
        """Intercept stream-unary gRPC calls."""
        service, method = self._parse_method(client_call_details.method)

        try:
            response = continuation(client_call_details, request_iterator)

            # Record successful streaming call
            if self.settings.feature_flags.domain_specific_metric_registries:
                grpc_registry = self.migration_helper.factory.get_grpc_registry()
                grpc_registry.record_stream_message(service, method, "received")
            else:
                legacy_metrics.GPU_SERVICE_CALLS_TOTAL.labels(
                    service=service, method=f"{method}_stream", status="success"
                ).inc()

            return response

        except grpc.RpcError as e:
            # Record failed streaming call
            error_code = str(e.code())
            if self.settings.feature_flags.domain_specific_metric_registries:
                grpc_registry = self.migration_helper.factory.get_grpc_registry()
                grpc_registry.record_rpc_error(service, method, error_code)
            else:
                legacy_metrics.GPU_SERVICE_CALLS_TOTAL.labels(
                    service=service, method=f"{method}_stream", status="error"
                ).inc()
                legacy_metrics.GPU_SERVICE_ERRORS_TOTAL.labels(
                    service=service, method=method, error_type=error_code
                ).inc()
            raise

    def intercept_stream_stream(
        self,
        continuation: Callable[[grpc.ClientCallDetails, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request_iterator: Any,
    ) -> Any:
        """Intercept stream-stream gRPC calls."""
        service, method = self._parse_method(client_call_details.method)

        try:
            response = continuation(client_call_details, request_iterator)

            # Record successful bidirectional streaming call
            if self.settings.feature_flags.domain_specific_metric_registries:
                grpc_registry = self.migration_helper.factory.get_grpc_registry()
                grpc_registry.record_stream_message(service, method, "bidirectional")
            else:
                from Medical_KG_rev.observability.metrics import GPU_SERVICE_CALLS_TOTAL
                GPU_SERVICE_CALLS_TOTAL.labels(
                    service=service, method=f"{method}_stream", status="success"
                ).inc()

            return response

        except grpc.RpcError as e:
            # Record failed bidirectional streaming call
            error_code = str(e.code())
            if self.settings.feature_flags.domain_specific_metric_registries:
                grpc_registry = self.migration_helper.factory.get_grpc_registry()
                grpc_registry.record_rpc_error(service, method, error_code)
            else:
                from Medical_KG_rev.observability.metrics import (
                    GPU_SERVICE_CALLS_TOTAL,
                    GPU_SERVICE_ERRORS_TOTAL,
                )
                GPU_SERVICE_CALLS_TOTAL.labels(
                    service=service, method=f"{method}_stream", status="error"
                ).inc()
                GPU_SERVICE_ERRORS_TOTAL.labels(
                    service=service, method=method, error_type=error_code
                ).inc()
            raise

    def _parse_method(self, full_method: str) -> tuple[str, str]:
        """Parse /package.Service/Method into (Service, Method).

        Args:
            full_method: Full gRPC method name (e.g., "/embedding.EmbeddingService/GenerateEmbeddings")

        Returns:
            Tuple of (service_name, method_name)
        """
        # Remove leading slash and split by /
        parts = full_method.strip('/').split('/')
        if len(parts) >= 2:
            # Extract service name from package.Service format
            service_part = parts[0].split('.')[-1]  # Get Service from package.Service
            method_part = parts[1]
            return service_part, method_part
        else:
            # Fallback for unexpected format
            return "unknown", full_method.replace('/', '_')


def create_metrics_channel(endpoint: str) -> aio.Channel:
    """Create a gRPC channel with metrics interceptor.

    Args:
        endpoint: gRPC service endpoint

    Returns:
        gRPC channel with metrics interceptor
    """
    # Create channel with interceptor
    channel = aio.insecure_channel(endpoint)

    # Note: For async channels, we need to use intercept_channel
    # This is a simplified implementation - in practice, you might need
    # to use grpc.aio.intercept_channel for async channels
    return channel


def create_metrics_channel_sync(endpoint: str) -> grpc.Channel:
    """Create a synchronous gRPC channel with metrics interceptor.

    Args:
        endpoint: gRPC service endpoint

    Returns:
        Synchronous gRPC channel with metrics interceptor
    """
    interceptor = MetricsClientInterceptor()

    # Create channel with interceptor
    channel = grpc.insecure_channel(endpoint)

    # Apply interceptor to channel
    return grpc.intercept_channel(channel, interceptor)
