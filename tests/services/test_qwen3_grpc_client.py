"""Unit tests for Qwen3 gRPC client."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from Medical_KG_rev.services.clients.qwen3_grpc_client import (
    Qwen3GRPCClient,
    Qwen3ServiceUnavailableError,
)


class TestQwen3GRPCClient:
    """Test cases for Qwen3GRPCClient."""

    def test_init(self) -> None:
        """Test client initialization."""
        client = Qwen3GRPCClient(
            endpoint="localhost:50051",
            timeout=30.0,
            max_retries=3,
            retry_delay=1.0,
        )

        assert client.endpoint == "localhost:50051"
        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client.retry_delay == 1.0
        assert client.stub is not None

    def test_init_defaults(self) -> None:
        """Test client initialization with defaults."""
        client = Qwen3GRPCClient()

        assert client.endpoint == "localhost:50051"
        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client.retry_delay == 1.0

    @patch("Medical_KG_rev.services.clients.qwen3_grpc_client.create_metrics_channel_sync")
    def test_embed_texts_success(self, mock_create_channel: Mock) -> None:
        """Test successful embedding generation."""
        # Mock channel and stub
        mock_channel = Mock()
        mock_stub = Mock()
        mock_create_channel.return_value = mock_channel

        # Mock response
        mock_response = Mock()
        mock_embedding_vector = Mock()
        mock_embedding_vector.values = [0.1, 0.2, 0.3]
        mock_response.embeddings = [mock_embedding_vector]
        mock_stub.Embed.return_value = mock_response

        client = Qwen3GRPCClient(endpoint="localhost:50051")
        client.stub = mock_stub

        # Test embedding generation
        result = client.embed_texts(["test text"])

        assert result == [[0.1, 0.2, 0.3]]
        mock_stub.Embed.assert_called_once()

    @patch("Medical_KG_rev.services.clients.qwen3_grpc_client.create_metrics_channel_sync")
    def test_embed_texts_empty_input(self, mock_create_channel: Mock) -> None:
        """Test embedding generation with empty input."""
        mock_channel = Mock()
        mock_create_channel.return_value = mock_channel

        client = Qwen3GRPCClient(endpoint="localhost:50051")

        result = client.embed_texts([])

        assert result == []

    @patch("Medical_KG_rev.services.clients.qwen3_grpc_client.create_metrics_channel_sync")
    def test_embed_texts_grpc_error(self, mock_create_channel: Mock) -> None:
        """Test embedding generation with gRPC error."""
        import grpc

        # Mock channel and stub
        mock_channel = Mock()
        mock_stub = Mock()
        mock_create_channel.return_value = mock_channel

        # Mock gRPC error - create a proper exception
        class MockRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self):
                return "Service unavailable"

        mock_stub.Embed.side_effect = MockRpcError()

        client = Qwen3GRPCClient(endpoint="localhost:50051")
        client.stub = mock_stub

        # Test error handling
        with pytest.raises(Qwen3ServiceUnavailableError) as exc_info:
            client.embed_texts(["test text"])

        assert "Service unavailable" in str(exc_info.value)
        assert exc_info.value.endpoint == "localhost:50051"

    @patch("Medical_KG_rev.services.clients.qwen3_grpc_client.create_metrics_channel_sync")
    def test_health_check_success(self, mock_create_channel: Mock) -> None:
        """Test successful health check."""
        # Mock channel and stub
        mock_channel = Mock()
        mock_stub = Mock()
        mock_create_channel.return_value = mock_channel

        # Mock response
        mock_response = Mock()
        mock_response.valid = True
        mock_stub.ValidateTexts.return_value = mock_response

        client = Qwen3GRPCClient(endpoint="localhost:50051")
        client.stub = mock_stub

        # Test health check
        result = client.health_check()

        assert result["status"] == "healthy"
        assert result["endpoint"] == "localhost:50051"
        assert result["service_available"] is True

    @patch("Medical_KG_rev.services.clients.qwen3_grpc_client.create_metrics_channel_sync")
    def test_health_check_failure(self, mock_create_channel: Mock) -> None:
        """Test health check failure."""
        import grpc

        # Mock channel and stub
        mock_channel = Mock()
        mock_stub = Mock()
        mock_create_channel.return_value = mock_channel

        # Mock gRPC error - create a proper exception
        class MockRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self):
                return "Service unavailable"

        mock_stub.ValidateTexts.side_effect = MockRpcError()

        client = Qwen3GRPCClient(endpoint="localhost:50051")
        client.stub = mock_stub

        # Test health check failure bubbles error
        with pytest.raises(Qwen3ServiceUnavailableError) as exc_info:
            client.health_check()

        assert "Service unavailable" in str(exc_info.value)

    @patch("Medical_KG_rev.services.clients.qwen3_grpc_client.create_metrics_channel_sync")
    def test_get_service_info_success(self, mock_create_channel: Mock) -> None:
        """Test successful service info retrieval."""
        # Mock channel and stub
        mock_channel = Mock()
        mock_stub = Mock()
        mock_create_channel.return_value = mock_channel

        # Mock response
        mock_response = Mock()
        mock_namespace_info = Mock()
        mock_namespace_info.id = "default"
        mock_namespace_info.provider = "qwen3"
        mock_namespace_info.kind = "embedding"
        mock_namespace_info.dimension = 4096
        mock_namespace_info.max_tokens = 2048
        mock_namespace_info.enabled = True
        mock_response.namespaces = [mock_namespace_info]
        mock_stub.ListNamespaces.return_value = mock_response

        client = Qwen3GRPCClient(endpoint="localhost:50051")
        client.stub = mock_stub

        # Test service info
        result = client.get_service_info()

        assert result["endpoint"] == "localhost:50051"
        assert len(result["available_namespaces"]) == 1
        assert result["available_namespaces"][0]["id"] == "default"

    @patch("Medical_KG_rev.services.clients.qwen3_grpc_client.create_metrics_channel_sync")
    def test_context_manager(self, mock_create_channel: Mock) -> None:
        """Test context manager usage."""
        mock_channel = Mock()
        mock_create_channel.return_value = mock_channel

        with Qwen3GRPCClient(endpoint="localhost:50051") as client:
            assert client.endpoint == "localhost:50051"
            assert client.stub is not None

        # Channel close should be called (though it may warn about async)
        # We can't easily test this without mocking the close method
