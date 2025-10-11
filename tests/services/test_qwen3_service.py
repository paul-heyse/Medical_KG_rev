"""Unit tests for Qwen3Service with gRPC support."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from Medical_KG_rev.services.retrieval.qwen3_service import (
    Qwen3ProcessingError,
    Qwen3Result,
    Qwen3Service,
)


class TestQwen3Service:
    """Test cases for Qwen3Service."""

    @patch("Medical_KG_rev.services.retrieval.qwen3_service.get_settings")
    def test_init_grpc_mode(self, mock_get_settings: Mock) -> None:
        """Test service initialization in gRPC mode."""
        # Mock settings
        mock_settings = Mock()
        mock_qwen3_settings = Mock()
        mock_qwen3_settings.use_grpc = True
        mock_qwen3_settings.grpc_endpoint = "localhost:50051"
        mock_qwen3_settings.grpc_timeout = 30.0
        mock_qwen3_settings.grpc_max_retries = 3
        mock_qwen3_settings.grpc_retry_delay = 1.0
        mock_settings.retrieval.qwen3 = mock_qwen3_settings

        mock_get_settings.return_value = mock_settings

        # Mock gRPC client
        with patch("Medical_KG_rev.services.retrieval.qwen3_service.Qwen3GRPCClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            service = Qwen3Service()

            assert service.use_grpc is True
            assert service._is_loaded is True
            assert service.grpc_client == mock_client
            mock_client_class.assert_called_once_with(
                endpoint="localhost:50051",
                timeout=30.0,
                max_retries=3,
                retry_delay=1.0,
            )

    @patch("Medical_KG_rev.services.retrieval.qwen3_service.get_settings")
    def test_init_legacy_mode(self, mock_get_settings: Mock) -> None:
        """Test service initialization in legacy mode."""
        # Mock settings
        mock_settings = Mock()
        mock_qwen3_settings = Mock()
        mock_qwen3_settings.use_grpc = False
        mock_settings.retrieval.qwen3 = mock_qwen3_settings

        mock_get_settings.return_value = mock_settings

        service = Qwen3Service()

        assert service.use_grpc is False
        assert service._is_loaded is False
        assert service.grpc_client is None

    @patch("Medical_KG_rev.services.retrieval.qwen3_service.get_settings")
    def test_preprocess_text(self, mock_get_settings: Mock) -> None:
        """Test text preprocessing."""
        # Mock settings
        mock_settings = Mock()
        mock_qwen3_settings = Mock()
        mock_qwen3_settings.use_grpc = False
        mock_settings.retrieval.qwen3 = mock_qwen3_settings

        mock_get_settings.return_value = mock_settings

        service = Qwen3Service(max_seq_length=100)

        # Test normal text
        result = service._preprocess_text("Hello world")
        assert result == "Hello world"

        # Test text that needs truncation
        long_text = "x" * 500  # Longer than max_seq_length * 4
        result = service._preprocess_text(long_text)
        assert len(result) <= 400  # max_seq_length * 4

    @patch("Medical_KG_rev.services.retrieval.qwen3_service.get_settings")
    def test_generate_embedding_grpc_mode(self, mock_get_settings: Mock) -> None:
        """Test embedding generation in gRPC mode."""
        # Mock settings
        mock_settings = Mock()
        mock_qwen3_settings = Mock()
        mock_qwen3_settings.use_grpc = True
        mock_settings.retrieval.qwen3 = mock_qwen3_settings

        mock_get_settings.return_value = mock_settings

        # Mock gRPC client
        with patch("Medical_KG_rev.services.retrieval.qwen3_service.Qwen3GRPCClient") as mock_client_class:
            mock_client = Mock()
            mock_client.embed_texts.return_value = [[0.1, 0.2, 0.3]]
            mock_client_class.return_value = mock_client

            service = Qwen3Service()

            result = service._generate_embedding("test text")

            assert result == [0.1, 0.2, 0.3]
            mock_client.embed_texts.assert_called_once_with(
                texts=["test text"],
                model_name="Qwen/Qwen2.5-7B-Instruct",
                namespace="default",
                normalize=True,
            )

    @patch("Medical_KG_rev.services.retrieval.qwen3_service.get_settings")
    def test_generate_embedding_legacy_mode(self, mock_get_settings: Mock) -> None:
        """Test embedding generation in legacy mode."""
        # Mock settings
        mock_settings = Mock()
        mock_qwen3_settings = Mock()
        mock_qwen3_settings.use_grpc = False
        mock_settings.retrieval.qwen3 = mock_qwen3_settings

        mock_get_settings.return_value = mock_settings

        service = Qwen3Service()

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            service._generate_embedding("test text")

        assert "In-process embedding generation is deprecated" in str(exc_info.value)

    @patch("Medical_KG_rev.services.retrieval.qwen3_service.get_settings")
    def test_generate_embedding_success(self, mock_get_settings: Mock) -> None:
        """Test successful embedding generation."""
        # Mock settings
        mock_settings = Mock()
        mock_qwen3_settings = Mock()
        mock_qwen3_settings.use_grpc = True
        mock_settings.retrieval.qwen3 = mock_qwen3_settings

        mock_get_settings.return_value = mock_settings

        # Mock gRPC client
        with patch("Medical_KG_rev.services.retrieval.qwen3_service.Qwen3GRPCClient") as mock_client_class:
            mock_client = Mock()
            mock_client.embed_texts.return_value = [[0.1, 0.2, 0.3]]
            mock_client_class.return_value = mock_client

            service = Qwen3Service()

            result = service.generate_embedding("chunk_1", "test text")

            assert isinstance(result, Qwen3Result)
            assert result.chunk_id == "chunk_1"
            assert result.embedding == [0.1, 0.2, 0.3]
            assert result.model_name == "Qwen/Qwen2.5-7B-Instruct"
            assert result.processing_time_seconds > 0

    @patch("Medical_KG_rev.services.retrieval.qwen3_service.get_settings")
    def test_generate_embedding_error(self, mock_get_settings: Mock) -> None:
        """Test embedding generation error handling."""
        from Medical_KG_rev.services.clients.qwen3_grpc_client import Qwen3ServiceUnavailableError

        # Mock settings
        mock_settings = Mock()
        mock_qwen3_settings = Mock()
        mock_qwen3_settings.use_grpc = True
        mock_settings.retrieval.qwen3 = mock_qwen3_settings

        mock_get_settings.return_value = mock_settings

        # Mock gRPC client
        with patch("Medical_KG_rev.services.retrieval.qwen3_service.Qwen3GRPCClient") as mock_client_class:
            mock_client = Mock()
            mock_client.embed_texts.side_effect = Qwen3ServiceUnavailableError("Service unavailable", "localhost:50051")
            mock_client_class.return_value = mock_client

            service = Qwen3Service()

            with pytest.raises(Qwen3ProcessingError) as exc_info:
                service.generate_embedding("chunk_1", "test text")

            assert "Qwen3 service unavailable" in str(exc_info.value)

    @patch("Medical_KG_rev.services.retrieval.qwen3_service.get_settings")
    def test_generate_embeddings_batch_grpc_mode(self, mock_get_settings: Mock) -> None:
        """Test batch embedding generation in gRPC mode."""
        # Mock settings
        mock_settings = Mock()
        mock_qwen3_settings = Mock()
        mock_qwen3_settings.use_grpc = True
        mock_settings.retrieval.qwen3 = mock_qwen3_settings

        mock_get_settings.return_value = mock_settings

        # Mock gRPC client
        with patch("Medical_KG_rev.services.retrieval.qwen3_service.Qwen3GRPCClient") as mock_client_class:
            mock_client = Mock()
            mock_client.embed_texts.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_client_class.return_value = mock_client

            service = Qwen3Service()

            chunks = [("chunk_1", "text 1"), ("chunk_2", "text 2")]
            results = service.generate_embeddings_batch(chunks)

            assert len(results) == 2
            assert all(isinstance(result, Qwen3Result) for result in results)
            assert results[0].chunk_id == "chunk_1"
            assert results[1].chunk_id == "chunk_2"
            assert results[0].embedding == [0.1, 0.2, 0.3]
            assert results[1].embedding == [0.4, 0.5, 0.6]

    @patch("Medical_KG_rev.services.retrieval.qwen3_service.get_settings")
    def test_health_check_grpc_mode(self, mock_get_settings: Mock) -> None:
        """Test health check in gRPC mode."""
        # Mock settings
        mock_settings = Mock()
        mock_qwen3_settings = Mock()
        mock_qwen3_settings.use_grpc = True
        mock_settings.retrieval.qwen3 = mock_qwen3_settings

        mock_get_settings.return_value = mock_settings

        # Mock gRPC client
        with patch("Medical_KG_rev.services.retrieval.qwen3_service.Qwen3GRPCClient") as mock_client_class:
            mock_client = Mock()
            mock_client.health_check.return_value = {
                "status": "healthy",
                "endpoint": "localhost:50051",
                "service_available": True,
            }
            mock_client_class.return_value = mock_client

            service = Qwen3Service()

            result = service.health_check()

            assert result["status"] == "healthy"
            assert result["mode"] == "grpc"
            assert result["model_name"] == "Qwen/Qwen2.5-7B-Instruct"

    @patch("Medical_KG_rev.services.retrieval.qwen3_service.get_settings")
    def test_get_service_stats_grpc_mode(self, mock_get_settings: Mock) -> None:
        """Test service stats in gRPC mode."""
        # Mock settings
        mock_settings = Mock()
        mock_qwen3_settings = Mock()
        mock_qwen3_settings.use_grpc = True
        mock_settings.retrieval.qwen3 = mock_qwen3_settings

        mock_get_settings.return_value = mock_settings

        # Mock gRPC client
        with patch("Medical_KG_rev.services.retrieval.qwen3_service.Qwen3GRPCClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get_service_info.return_value = {
                "endpoint": "localhost:50051",
                "timeout": 30.0,
                "available_namespaces": [],
            }
            mock_client_class.return_value = mock_client

            service = Qwen3Service()

            result = service.get_service_stats()

            assert result["mode"] == "grpc"
            assert result["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
            assert result["embedding_dimension"] == 4096

    @patch("Medical_KG_rev.services.retrieval.qwen3_service.get_settings")
    def test_context_manager(self, mock_get_settings: Mock) -> None:
        """Test context manager usage."""
        # Mock settings
        mock_settings = Mock()
        mock_qwen3_settings = Mock()
        mock_qwen3_settings.use_grpc = True
        mock_settings.retrieval.qwen3 = mock_qwen3_settings

        mock_get_settings.return_value = mock_settings

        # Mock gRPC client
        with patch("Medical_KG_rev.services.retrieval.qwen3_service.Qwen3GRPCClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            with Qwen3Service() as service:
                assert service.use_grpc is True
                assert service.grpc_client == mock_client

            # Client close should be called
            mock_client.close.assert_called_once()
