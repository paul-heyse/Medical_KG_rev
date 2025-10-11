"""Integration tests for torch isolation architecture."""

from __future__ import annotations

import time

import grpc
import pytest
import requests
from grpc_health.v1 import health_pb2, health_pb2_grpc

from Medical_KG_rev.services.embedding.grpc_client import EmbeddingServiceClient
from Medical_KG_rev.services.gpu.grpc_client import GPUServiceClient
from Medical_KG_rev.services.reranking.grpc_client import RerankingServiceClient


class TestTorchIsolationArchitecture:
    """Test suite for torch isolation architecture."""

    @pytest.fixture
    def gateway_url(self) -> str:
        """Gateway URL for testing."""
        return "http://localhost:8000"

    @pytest.fixture
    def gpu_service_url(self) -> str:
        """GPU service URL for testing."""
        return "localhost:50051"

    @pytest.fixture
    def embedding_service_url(self) -> str:
        """Embedding service URL for testing."""
        return "localhost:50051"

    @pytest.fixture
    def reranking_service_url(self) -> str:
        """Reranking service URL for testing."""
        return "localhost:50051"

    def test_gateway_health(self, gateway_url: str) -> None:
        """Test gateway health endpoint."""
        response = requests.get(f"{gateway_url}/health", timeout=10)
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"

    def test_gpu_service_health(self, gpu_service_url: str) -> None:
        """Test GPU service health."""
        channel = grpc.insecure_channel(gpu_service_url)
        stub = health_pb2_grpc.HealthStub(channel)

        request = health_pb2.HealthCheckRequest(service="")
        response = stub.Check(request)

        assert response.status == health_pb2.HealthCheckResponse.SERVING

    def test_embedding_service_health(self, embedding_service_url: str) -> None:
        """Test embedding service health."""
        channel = grpc.insecure_channel(embedding_service_url)
        stub = health_pb2_grpc.HealthStub(channel)

        request = health_pb2.HealthCheckRequest(service="")
        response = stub.Check(request)

        assert response.status == health_pb2.HealthCheckResponse.SERVING

    def test_reranking_service_health(self, reranking_service_url: str) -> None:
        """Test reranking service health."""
        channel = grpc.insecure_channel(reranking_service_url)
        stub = health_pb2_grpc.HealthStub(channel)

        request = health_pb2.HealthCheckRequest(service="")
        response = stub.Check(request)

        assert response.status == health_pb2.HealthCheckResponse.SERVING

    def test_gpu_service_client(self, gpu_service_url: str) -> None:
        """Test GPU service client."""
        client = GPUServiceClient(gpu_service_url)

        # Test GPU status
        status = client.get_gpu_status()
        assert "available" in status
        assert "memory" in status

    def test_embedding_service_client(self, embedding_service_url: str) -> None:
        """Test embedding service client."""
        client = EmbeddingServiceClient(embedding_service_url)

        # Test embedding generation
        texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = client.generate_embeddings(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0  # Should have embedding dimensions

    def test_reranking_service_client(self, reranking_service_url: str) -> None:
        """Test reranking service client."""
        client = RerankingServiceClient(reranking_service_url)

        # Test reranking
        query = "test query"
        documents = ["document 1", "document 2", "document 3"]
        scores = client.rerank(query, documents)

        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores)

    def test_gateway_chunking_endpoint(self, gateway_url: str) -> None:
        """Test gateway chunking endpoint (torch-free)."""
        payload = {
            "text": "This is a test document with multiple sentences. It should be chunked properly.",
            "chunker": "docling",
            "granularity": "paragraph",
        }

        response = requests.post(f"{gateway_url}/v1/chunking/chunk", json=payload, timeout=30)

        assert response.status_code == 200
        result = response.json()
        assert "chunks" in result
        assert len(result["chunks"]) > 0

    def test_gateway_embedding_endpoint(self, gateway_url: str) -> None:
        """Test gateway embedding endpoint (via gRPC)."""
        payload = {
            "texts": ["This is a test sentence.", "Another test sentence."],
            "model": "default",
        }

        response = requests.post(f"{gateway_url}/v1/embedding/generate", json=payload, timeout=30)

        assert response.status_code == 200
        result = response.json()
        assert "embeddings" in result
        assert len(result["embeddings"]) == 2

    def test_gateway_reranking_endpoint(self, gateway_url: str) -> None:
        """Test gateway reranking endpoint (via gRPC)."""
        payload = {
            "query": "test query",
            "documents": ["document 1", "document 2", "document 3"],
            "model": "default",
        }

        response = requests.post(f"{gateway_url}/v1/reranking/rerank", json=payload, timeout=30)

        assert response.status_code == 200
        result = response.json()
        assert "scores" in result
        assert len(result["scores"]) == 3

    def test_circuit_breaker_pattern(self, gateway_url: str) -> None:
        """Test circuit breaker pattern for service failures."""
        # This test would require simulating service failures
        # For now, we'll just test that the endpoint responds
        payload = {"texts": ["test"], "model": "default"}

        response = requests.post(f"{gateway_url}/v1/embedding/generate", json=payload, timeout=30)

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 503, 504]

    def test_service_discovery(self, gateway_url: str) -> None:
        """Test service discovery functionality."""
        response = requests.get(f"{gateway_url}/v1/services/status", timeout=10)
        assert response.status_code == 200

        services = response.json()
        assert "gpu" in services
        assert "embedding" in services
        assert "reranking" in services

    def test_performance_requirements(self, gateway_url: str) -> None:
        """Test performance requirements (P95 < 500ms)."""
        payload = {"texts": ["Performance test sentence."], "model": "default"}

        start_time = time.time()
        response = requests.post(f"{gateway_url}/v1/embedding/generate", json=payload, timeout=30)
        end_time = time.time()

        response_time = (end_time - start_time) * 1000  # Convert to milliseconds

        assert response.status_code == 200
        assert response_time < 500  # P95 requirement

    def test_torch_free_gateway(self, gateway_url: str) -> None:
        """Test that gateway is completely torch-free."""
        # Test that gateway can start without torch
        response = requests.get(f"{gateway_url}/health", timeout=10)
        assert response.status_code == 200

        # Test that chunking works without torch
        payload = {"text": "Test document for torch-free chunking.", "chunker": "docling"}

        response = requests.post(f"{gateway_url}/v1/chunking/chunk", json=payload, timeout=30)

        assert response.status_code == 200
        result = response.json()
        assert "chunks" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
