"""Unit tests for MetricsClientInterceptor."""

from unittest.mock import Mock, patch

import grpc

from Medical_KG_rev.observability.grpc_interceptor import (
    MetricsClientInterceptor,
    create_metrics_channel_sync,
)


class TestMetricsClientInterceptor:
    """Test cases for MetricsClientInterceptor."""

    @patch('Medical_KG_rev.observability.grpc_interceptor.get_settings')
    def test_init(self, mock_get_settings) -> None:
        """Test interceptor initialization."""
        mock_settings = Mock()
        mock_settings.feature_flags.domain_specific_metric_registries = True
        mock_get_settings.return_value = mock_settings

        interceptor = MetricsClientInterceptor()

        assert interceptor.settings is mock_settings
        assert interceptor.settings.feature_flags.domain_specific_metric_registries is True

    @patch('Medical_KG_rev.observability.grpc_interceptor.get_settings')
    def test_parse_method_valid(self, mock_get_settings) -> None:
        """Test parsing valid gRPC method."""
        mock_settings = Mock()
        mock_settings.feature_flags.domain_specific_metric_registries = True
        mock_get_settings.return_value = mock_settings

        interceptor = MetricsClientInterceptor()

        service, method = interceptor._parse_method("/embedding.EmbeddingService/GenerateEmbeddings")
        assert service == "EmbeddingService"
        assert method == "GenerateEmbeddings"

    @patch('Medical_KG_rev.observability.grpc_interceptor.get_settings')
    def test_parse_method_without_package(self, mock_get_settings) -> None:
        """Test parsing gRPC method without package."""
        mock_settings = Mock()
        mock_settings.feature_flags.domain_specific_metric_registries = True
        mock_get_settings.return_value = mock_settings

        interceptor = MetricsClientInterceptor()

        service, method = interceptor._parse_method("/EmbeddingService/GenerateEmbeddings")
        assert service == "EmbeddingService"
        assert method == "GenerateEmbeddings"

    @patch('Medical_KG_rev.observability.grpc_interceptor.get_settings')
    def test_parse_method_invalid_format(self, mock_get_settings) -> None:
        """Test parsing invalid gRPC method format."""
        mock_settings = Mock()
        mock_settings.feature_flags.domain_specific_metric_registries = True
        mock_get_settings.return_value = mock_settings

        interceptor = MetricsClientInterceptor()

        service, method = interceptor._parse_method("invalid_method")
        assert service == "unknown"
        assert method == "invalid_method"

    @patch('Medical_KG_rev.observability.grpc_interceptor.get_settings')
    def test_intercept_unary_unary_success(self, mock_get_settings) -> None:
        """Test intercepting successful unary-unary call."""
        mock_settings = Mock()
        mock_settings.feature_flags.domain_specific_metric_registries = True
        mock_get_settings.return_value = mock_settings

        mock_registry = Mock()
        mock_helper = Mock()
        mock_helper.factory.get_grpc_registry.return_value = mock_registry

        interceptor = MetricsClientInterceptor()
        interceptor.migration_helper = mock_helper

        mock_continuation = Mock()
        mock_client_call_details = Mock()
        mock_client_call_details.method = "/embedding.EmbeddingService/GenerateEmbeddings"
        mock_request = Mock()
        mock_response = Mock()
        mock_continuation.return_value = mock_response

        result = interceptor.intercept_unary_unary(mock_continuation, mock_client_call_details, mock_request)

        assert result is mock_response
        mock_registry.record_rpc_call.assert_called_once_with("EmbeddingService", "GenerateEmbeddings", "OK")
        mock_registry.observe_rpc_duration.assert_called_once()

    @patch('Medical_KG_rev.observability.grpc_interceptor.get_settings')
    def test_intercept_unary_unary_error(self, mock_get_settings) -> None:
        """Test intercepting failed unary-unary call."""
        mock_settings = Mock()
        mock_settings.feature_flags.domain_specific_metric_registries = True
        mock_get_settings.return_value = mock_settings

        mock_registry = Mock()
        mock_helper = Mock()
        mock_helper.factory.get_grpc_registry.return_value = mock_registry

        interceptor = MetricsClientInterceptor()
        interceptor.migration_helper = mock_helper

        mock_continuation = Mock()
        mock_client_call_details = Mock()
        mock_client_call_details.method = "/embedding.EmbeddingService/GenerateEmbeddings"
        mock_request = Mock()

        # Create a real gRPC error
        mock_error = grpc.RpcError()
        mock_error.code = lambda: grpc.StatusCode.UNAVAILABLE
        mock_continuation.side_effect = mock_error

        try:
            interceptor.intercept_unary_unary(mock_continuation, mock_client_call_details, mock_request)
        except grpc.RpcError:
            pass  # Expected to raise the gRPC error

        mock_registry.record_rpc_error.assert_called_once_with("EmbeddingService", "GenerateEmbeddings", "StatusCode.UNAVAILABLE")

    @patch('Medical_KG_rev.observability.grpc_interceptor.get_settings')
    def test_intercept_unary_unary_feature_flag_disabled(self, mock_get_settings) -> None:
        """Test intercepting unary-unary call with feature flag disabled."""
        mock_settings = Mock()
        mock_settings.feature_flags.domain_specific_metric_registries = False
        mock_get_settings.return_value = mock_settings

        interceptor = MetricsClientInterceptor()

        mock_continuation = Mock()
        mock_client_call_details = Mock()
        mock_client_call_details.method = "/embedding.EmbeddingService/GenerateEmbeddings"
        mock_request = Mock()
        mock_response = Mock()
        mock_continuation.return_value = mock_response

        with patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALLS_TOTAL') as mock_counter, \
             patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALL_DURATION_SECONDS') as mock_histogram:
            result = interceptor.intercept_unary_unary(mock_continuation, mock_client_call_details, mock_request)

            assert result is mock_response
            mock_counter.labels.assert_called_once()
            mock_histogram.labels.assert_called_once()


class TestChannelCreation:
    """Test cases for channel creation functions."""

    @patch('Medical_KG_rev.observability.grpc_interceptor.grpc.insecure_channel')
    @patch('Medical_KG_rev.observability.grpc_interceptor.grpc.intercept_channel')
    @patch('Medical_KG_rev.observability.grpc_interceptor.get_settings')
    def test_create_metrics_channel_sync(self, mock_get_settings, mock_intercept_channel, mock_insecure_channel) -> None:
        """Test creating synchronous metrics channel."""
        mock_settings = Mock()
        mock_settings.feature_flags.domain_specific_metric_registries = True
        mock_get_settings.return_value = mock_settings

        mock_channel = Mock()
        mock_intercepted_channel = Mock()
        mock_insecure_channel.return_value = mock_channel
        mock_intercept_channel.return_value = mock_intercepted_channel

        result = create_metrics_channel_sync("localhost:50051")

        assert result is mock_intercepted_channel
        mock_insecure_channel.assert_called_once_with("localhost:50051")
        mock_intercept_channel.assert_called_once()
