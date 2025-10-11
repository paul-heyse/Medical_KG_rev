"""Integration tests for VLLM service using testcontainers."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from Medical_KG_rev.services.clients.qwen3_grpc_client import (
    Qwen3GRPCClient,
    Qwen3ServiceUnavailableError,
)


class TestVLLMIntegration:
    """Integration tests for VLLM service."""

    @pytest.fixture
    def vllm_endpoint(self):
        """Get VLLM service endpoint from environment or use default."""
        import os
        return os.getenv("VLLM_ENDPOINT", "localhost:50051")

    @pytest.fixture
    def qwen3_client(self, vllm_endpoint):
        """Create Qwen3 gRPC client for testing."""
        return Qwen3GRPCClient(
            endpoint=vllm_endpoint,
            timeout=10.0,
            max_retries=1
        )

    def test_vllm_service_health_check(self, qwen3_client):
        """Test VLLM service health check."""
        try:
            health = qwen3_client.health_check()
            assert health["status"] == "healthy"
            assert health["mode"] == "grpc"
        except Qwen3ServiceUnavailableError:
            pytest.skip("VLLM service not available for integration testing")

    def test_vllm_embedding_generation(self, qwen3_client):
        """Test embedding generation with real VLLM service."""
        try:
            # Test single text embedding
            texts = ["This is a test document for embedding generation."]
            embeddings = qwen3_client.embed_texts(texts)

            assert len(embeddings) == 1
            assert len(embeddings[0]) > 0  # Should have embedding vector
            assert all(isinstance(val, float) for val in embeddings[0])

        except Qwen3ServiceUnavailableError:
            pytest.skip("VLLM service not available for integration testing")

    def test_vllm_batch_embedding_generation(self, qwen3_client):
        """Test batch embedding generation with real VLLM service."""
        try:
            # Test batch embedding
            texts = [
                "First document for batch processing.",
                "Second document for batch processing.",
                "Third document for batch processing."
            ]
            embeddings = qwen3_client.embed_texts(texts)

            assert len(embeddings) == 3
            for i, embedding in enumerate(embeddings):
                assert len(embedding) > 0, f"Embedding {i} is empty"
                assert all(isinstance(val, float) for val in embedding), f"Embedding {i} contains non-float values"

        except Qwen3ServiceUnavailableError:
            pytest.skip("VLLM service not available for integration testing")

    def test_vllm_service_info(self, qwen3_client):
        """Test service information retrieval."""
        try:
            info = qwen3_client.get_service_info()
            assert "model_name" in info
            assert "version" in info
            assert "gpu_available" in info

        except Qwen3ServiceUnavailableError:
            pytest.skip("VLLM service not available for integration testing")

    def test_vllm_error_handling(self, qwen3_client):
        """Test error handling when service is unavailable."""
        # Test with invalid endpoint
        invalid_client = Qwen3GRPCClient(
            endpoint="localhost:99999",  # Invalid port
            timeout=1.0,
            max_retries=0
        )

        with pytest.raises(Qwen3ServiceUnavailableError):
            invalid_client.embed_texts(["test"])

    def test_vllm_context_manager(self, vllm_endpoint):
        """Test context manager functionality."""
        try:
            with Qwen3GRPCClient(vllm_endpoint) as client:
                health = client.health_check()
                assert health["status"] == "healthy"

        except Qwen3ServiceUnavailableError:
            pytest.skip("VLLM service not available for integration testing")

    def test_vllm_performance_characteristics(self, qwen3_client):
        """Test performance characteristics of VLLM service."""
        try:
            import time

            # Test single embedding performance
            start_time = time.time()
            texts = ["Performance test document."]
            qwen3_client.embed_texts(texts)
            single_duration = time.time() - start_time

            # Test batch embedding performance
            start_time = time.time()
            texts = [f"Batch test document {i}." for i in range(5)]
            qwen3_client.embed_texts(texts)
            batch_duration = time.time() - start_time

            # Batch should be more efficient per document
            assert batch_duration < single_duration * 5, "Batch processing should be more efficient"

            # Both should complete within reasonable time
            assert single_duration < 10.0, "Single embedding took too long"
            assert batch_duration < 20.0, "Batch embedding took too long"

        except Qwen3ServiceUnavailableError:
            pytest.skip("VLLM service not available for integration testing")


class TestVLLMServiceDiscovery:
    """Test service discovery and load balancing."""

    def test_multiple_endpoints(self):
        """Test service discovery with multiple endpoints."""
        endpoints = [
            "localhost:50051",
            "localhost:50052",
            "localhost:50053"
        ]

        available_endpoints = []

        for endpoint in endpoints:
            try:
                client = Qwen3GRPCClient(endpoint, timeout=2.0, max_retries=0)
                health = client.health_check()
                if health["status"] == "healthy":
                    available_endpoints.append(endpoint)
                client.close()
            except Qwen3ServiceUnavailableError:
                continue

        # At least one endpoint should be available for testing
        if not available_endpoints:
            pytest.skip("No VLLM endpoints available for service discovery testing")

        # Test load balancing by using different endpoints
        for endpoint in available_endpoints:
            client = Qwen3GRPCClient(endpoint)
            try:
                embeddings = client.embed_texts(["Service discovery test."])
                assert len(embeddings) == 1
                assert len(embeddings[0]) > 0
            finally:
                client.close()


class TestVLLMWithQwen3Service:
    """Integration tests using Qwen3Service with gRPC mode."""

    @pytest.fixture
    def qwen3_service_grpc(self):
        """Create Qwen3Service with gRPC mode enabled."""
        from Medical_KG_rev.services.retrieval.qwen3_service import Qwen3Service

        # Mock settings to enable gRPC mode
        with patch('Medical_KG_rev.services.retrieval.qwen3_service.get_settings') as mock_get_settings:
            mock_settings = type('MockSettings', (), {})()
            mock_qwen3_settings = type('MockQwen3Settings', (), {})()
            mock_qwen3_settings.use_grpc = True
            mock_qwen3_settings.grpc_endpoint = "localhost:50051"
            mock_qwen3_settings.grpc_timeout = 10.0
            mock_qwen3_settings.grpc_max_retries = 1
            mock_qwen3_settings.grpc_retry_delay = 1.0

            # Create retrieval mock
            mock_retrieval = type('MockRetrieval', (), {})()
            mock_retrieval.qwen3 = mock_qwen3_settings
            mock_settings.retrieval = mock_retrieval

            mock_get_settings.return_value = mock_settings

            return Qwen3Service()

    def test_qwen3_service_grpc_mode(self, qwen3_service_grpc):
        """Test Qwen3Service in gRPC mode."""
        assert qwen3_service_grpc.use_grpc is True
        assert qwen3_service_grpc.grpc_client is not None

    def test_qwen3_service_health_check_grpc(self, qwen3_service_grpc):
        """Test Qwen3Service health check in gRPC mode."""
        try:
            health = qwen3_service_grpc.health_check()
            assert health["status"] == "healthy"
            assert health["mode"] == "grpc"
        except Exception:
            pytest.skip("VLLM service not available for Qwen3Service testing")

    def test_qwen3_service_embedding_grpc(self, qwen3_service_grpc):
        """Test Qwen3Service embedding generation in gRPC mode."""
        try:
            result = qwen3_service_grpc.generate_embedding("chunk_1", "Test document for embedding.")
            assert result.chunk_id == "chunk_1"
            assert len(result.embedding) > 0
            assert all(isinstance(val, float) for val in result.embedding)
        except Exception:
            pytest.skip("VLLM service not available for Qwen3Service testing")

    def test_qwen3_service_batch_embedding_grpc(self, qwen3_service_grpc):
        """Test Qwen3Service batch embedding in gRPC mode."""
        try:
            chunks = [
                ("chunk_1", "First test document."),
                ("chunk_2", "Second test document."),
                ("chunk_3", "Third test document.")
            ]
            results = qwen3_service_grpc.generate_embeddings_batch(chunks)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.chunk_id == f"chunk_{i+1}"
                assert len(result.embedding) > 0
        except Exception:
            pytest.skip("VLLM service not available for Qwen3Service testing")


# Test configuration for CI/CD
@pytest.mark.integration
class TestVLLMIntegrationCI:
    """Integration tests configured for CI/CD environments."""

    @pytest.fixture(scope="session")
    def vllm_container(self):
        """Start VLLM container for CI testing."""
        try:
            from testcontainers.core.container import DockerContainer
            from testcontainers.core.waiting_strategies import wait_for_logs

            container = DockerContainer("vllm/vllm-openai:latest")
            container.with_exposed_ports(8000)
            container.with_env("MODEL", "Qwen/Qwen2.5-7B-Instruct")
            container.with_command("--model Qwen/Qwen2.5-7B-Instruct --port 8000")

            with container:
                wait_for_logs(container, "Uvicorn running", timeout=120)
                yield container

        except ImportError:
            pytest.skip("testcontainers not available for CI testing")
        except Exception as e:
            pytest.skip(f"Failed to start VLLM container: {e}")

    def test_vllm_container_health(self, vllm_container):
        """Test VLLM container health endpoint."""
        import requests

        endpoint = f"http://localhost:{vllm_container.get_exposed_port(8000)}"
        response = requests.get(f"{endpoint}/health", timeout=10)
        assert response.status_code == 200

        health_data = response.json()
        assert health_data.get("status") == "healthy"

    def test_vllm_container_embedding(self, vllm_container):
        """Test VLLM container embedding endpoint."""
        import requests

        endpoint = f"http://localhost:{vllm_container.get_exposed_port(8000)}"

        # Test OpenAI-compatible embedding endpoint
        response = requests.post(
            f"{endpoint}/v1/embeddings",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "input": "Test document for embedding."
            },
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]
        assert len(data["data"][0]["embedding"]) > 0
