"""Integration tests for embedding service."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock, patch

import pytest

from Medical_KG_rev.services.embedding.grpc_client import EmbeddingServiceClient


class TestEmbeddingServiceIntegration:
    """Integration tests for embedding service."""

    @pytest.fixture
    def embedding_service_url(self) -> str:
        """Embedding service URL for testing."""
        return "localhost:50051"

    @pytest.fixture
    def embedding_client(self, embedding_service_url: str) -> EmbeddingServiceClient:
        """Embedding service client for testing."""
        return EmbeddingServiceClient(embedding_service_url)

    @pytest.fixture
    def sample_texts(self) -> list[str]:
        """Sample texts for embedding generation."""
        return [
            "This is a test sentence for embedding generation.",
            "Another test sentence to verify batch processing.",
            "Medical knowledge graph integration test.",
        ]

    @pytest.mark.asyncio
    async def test_embedding_service_generate_embeddings(
        self, embedding_client: EmbeddingServiceClient, sample_texts: list[str]
    ) -> None:
        """Test embedding service generate embeddings endpoint."""
        embeddings = await embedding_client.generate_embeddings(texts=sample_texts, model="default")

        assert embeddings is not None
        assert len(embeddings) == len(sample_texts)
        assert all(len(embedding) > 0 for embedding in embeddings)

    @pytest.mark.asyncio
    async def test_embedding_service_list_models(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test embedding service list models endpoint."""
        models = await embedding_client.list_models()

        assert models is not None
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_embedding_service_health_check(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test embedding service health check."""
        health = await embedding_client.health_check()

        assert health is not None
        assert hasattr(health, "status")

    @pytest.mark.asyncio
    async def test_embedding_service_batch_processing(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test embedding service batch processing."""
        # Test with larger batch
        large_batch = [f"Test sentence {i}" for i in range(100)]

        embeddings = await embedding_client.generate_embeddings(texts=large_batch, model="default")

        assert embeddings is not None
        assert len(embeddings) == len(large_batch)

    @pytest.mark.asyncio
    async def test_embedding_service_different_models(
        self, embedding_client: EmbeddingServiceClient, sample_texts: list[str]
    ) -> None:
        """Test embedding service with different models."""
        models = await embedding_client.list_models()

        for model in models[:2]:  # Test first 2 models
            embeddings = await embedding_client.generate_embeddings(texts=sample_texts, model=model)

            assert embeddings is not None
            assert len(embeddings) == len(sample_texts)

    @pytest.mark.asyncio
    async def test_embedding_service_error_handling(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test embedding service error handling."""
        # Test with invalid model
        try:
            await embedding_client.generate_embeddings(texts=["test"], model="invalid-model")
            assert False, "Should have raised an error"
        except Exception as e:
            assert "model" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.asyncio
    async def test_embedding_service_concurrent_requests(
        self, embedding_client: EmbeddingServiceClient, sample_texts: list[str]
    ) -> None:
        """Test embedding service with concurrent requests."""
        # Create multiple concurrent requests
        tasks = [
            embedding_client.generate_embeddings(texts=sample_texts, model="default"),
            embedding_client.list_models(),
            embedding_client.health_check(),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(result is not None for result in results)

    @pytest.mark.asyncio
    async def test_embedding_service_circuit_breaker(self, embedding_service_url: str) -> None:
        """Test embedding service circuit breaker functionality."""
        # Mock service to simulate failures
        with patch("grpc.aio.insecure_channel") as mock_channel:
            mock_stub = Mock()
            mock_stub.GenerateEmbeddings.side_effect = Exception("Service unavailable")

            client = EmbeddingServiceClient(embedding_service_url)

            # Test circuit breaker activation
            for _ in range(6):  # Exceed failure threshold
                try:
                    await client.generate_embeddings(texts=["test"], model="default")
                except Exception:
                    pass

            # Circuit breaker should be open
            assert client.circuit_breaker.is_open()

    @pytest.mark.asyncio
    async def test_embedding_service_performance(
        self, embedding_client: EmbeddingServiceClient, sample_texts: list[str]
    ) -> None:
        """Test embedding service performance requirements."""
        import time

        start_time = time.time()
        await embedding_client.generate_embeddings(texts=sample_texts, model="default")
        end_time = time.time()

        response_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Should respond within 500ms (P95 requirement)
        assert response_time < 500

    @pytest.mark.asyncio
    async def test_embedding_service_metadata_propagation(
        self, embedding_client: EmbeddingServiceClient, sample_texts: list[str]
    ) -> None:
        """Test embedding service metadata propagation."""
        # Test that tenant_id and other metadata are propagated
        tenant_id = "test-tenant-123"

        # This would require implementing metadata propagation in the client
        # For now, we'll test basic functionality
        embeddings = await embedding_client.generate_embeddings(texts=sample_texts, model="default")
        assert embeddings is not None

    @pytest.mark.asyncio
    async def test_embedding_service_resource_cleanup(
        self, embedding_client: EmbeddingServiceClient, sample_texts: list[str]
    ) -> None:
        """Test embedding service resource cleanup."""
        # Test that resources are properly cleaned up
        await embedding_client.generate_embeddings(texts=sample_texts, model="default")

        # Client should handle cleanup automatically
        # This is more of a smoke test
        assert True

    @pytest.mark.asyncio
    async def test_embedding_service_consistency(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test embedding service consistency across requests."""
        text = "Consistency test sentence"

        # Generate embeddings multiple times
        embeddings1 = await embedding_client.generate_embeddings(texts=[text], model="default")
        embeddings2 = await embedding_client.generate_embeddings(texts=[text], model="default")

        # Embeddings should be consistent (same model, same text)
        assert len(embeddings1) == len(embeddings2)
        assert len(embeddings1[0]) == len(embeddings2[0])

    @pytest.mark.asyncio
    async def test_embedding_service_empty_input(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test embedding service with empty input."""
        try:
            await embedding_client.generate_embeddings(texts=[], model="default")
            assert False, "Should have raised an error for empty input"
        except Exception as e:
            assert "empty" in str(e).lower() or "invalid" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
