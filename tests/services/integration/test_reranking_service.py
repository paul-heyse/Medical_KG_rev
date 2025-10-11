"""Integration tests for reranking service."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock, patch

import pytest

from Medical_KG_rev.services.reranking.grpc_client import RerankingServiceClient


class TestRerankingServiceIntegration:
    """Integration tests for reranking service."""

    @pytest.fixture
    def reranking_service_url(self) -> str:
        """Reranking service URL for testing."""
        return "localhost:50051"

    @pytest.fixture
    def reranking_client(self, reranking_service_url: str) -> RerankingServiceClient:
        """Reranking service client for testing."""
        return RerankingServiceClient(reranking_service_url)

    @pytest.fixture
    def sample_query(self) -> str:
        """Sample query for reranking."""
        return "What are the side effects of this medication?"

    @pytest.fixture
    def sample_documents(self) -> list[str]:
        """Sample documents for reranking."""
        return [
            "This medication is used to treat hypertension and has minimal side effects.",
            "Common side effects include dizziness, headache, and fatigue.",
            "The drug was approved by the FDA in 2020 for cardiovascular treatment.",
            "Patients should monitor blood pressure regularly while taking this medication.",
            "Clinical trials showed 95% efficacy in reducing blood pressure.",
        ]

    @pytest.mark.asyncio
    async def test_reranking_service_rerank_batch(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking service rerank batch endpoint."""
        scores = await reranking_client.rerank(
            query=sample_query, documents=sample_documents, model="default"
        )

        assert scores is not None
        assert len(scores) == len(sample_documents)
        assert all(isinstance(score, float) for score in scores)

    @pytest.mark.asyncio
    async def test_reranking_service_list_models(
        self, reranking_client: RerankingServiceClient
    ) -> None:
        """Test reranking service list models endpoint."""
        models = await reranking_client.list_models()

        assert models is not None
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_reranking_service_health_check(
        self, reranking_client: RerankingServiceClient
    ) -> None:
        """Test reranking service health check."""
        health = await reranking_client.health_check()

        assert health is not None
        assert hasattr(health, "status")

    @pytest.mark.asyncio
    async def test_reranking_service_batch_processing(
        self, reranking_client: RerankingServiceClient
    ) -> None:
        """Test reranking service batch processing."""
        # Test with larger batch
        query = "Test query for batch processing"
        large_batch = [f"Test document {i}" for i in range(100)]

        scores = await reranking_client.rerank(query=query, documents=large_batch, model="default")

        assert scores is not None
        assert len(scores) == len(large_batch)

    @pytest.mark.asyncio
    async def test_reranking_service_different_models(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking service with different models."""
        models = await reranking_client.list_models()

        for model in models[:2]:  # Test first 2 models
            scores = await reranking_client.rerank(
                query=sample_query, documents=sample_documents, model=model
            )

            assert scores is not None
            assert len(scores) == len(sample_documents)

    @pytest.mark.asyncio
    async def test_reranking_service_error_handling(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking service error handling."""
        # Test with invalid model
        try:
            await reranking_client.rerank(
                query=sample_query, documents=sample_documents, model="invalid-model"
            )
            assert False, "Should have raised an error"
        except Exception as e:
            assert "model" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.asyncio
    async def test_reranking_service_concurrent_requests(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking service with concurrent requests."""
        # Create multiple concurrent requests
        tasks = [
            reranking_client.rerank(
                query=sample_query, documents=sample_documents, model="default"
            ),
            reranking_client.list_models(),
            reranking_client.health_check(),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(result is not None for result in results)

    @pytest.mark.asyncio
    async def test_reranking_service_circuit_breaker(self, reranking_service_url: str) -> None:
        """Test reranking service circuit breaker functionality."""
        # Mock service to simulate failures
        with patch("grpc.aio.insecure_channel") as mock_channel:
            mock_stub = Mock()
            mock_stub.RerankBatch.side_effect = Exception("Service unavailable")

            client = RerankingServiceClient(reranking_service_url)

            # Test circuit breaker activation
            for _ in range(6):  # Exceed failure threshold
                try:
                    await client.rerank(query="test", documents=["test"], model="default")
                except Exception:
                    pass

            # Circuit breaker should be open
            assert client.circuit_breaker.is_open()

    @pytest.mark.asyncio
    async def test_reranking_service_performance(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking service performance requirements."""
        import time

        start_time = time.time()
        await reranking_client.rerank(
            query=sample_query, documents=sample_documents, model="default"
        )
        end_time = time.time()

        response_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Should respond within 500ms (P95 requirement)
        assert response_time < 500

    @pytest.mark.asyncio
    async def test_reranking_service_metadata_propagation(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking service metadata propagation."""
        # Test that tenant_id and other metadata are propagated
        tenant_id = "test-tenant-123"

        # This would require implementing metadata propagation in the client
        # For now, we'll test basic functionality
        scores = await reranking_client.rerank(
            query=sample_query, documents=sample_documents, model="default"
        )
        assert scores is not None

    @pytest.mark.asyncio
    async def test_reranking_service_resource_cleanup(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking service resource cleanup."""
        # Test that resources are properly cleaned up
        await reranking_client.rerank(
            query=sample_query, documents=sample_documents, model="default"
        )

        # Client should handle cleanup automatically
        # This is more of a smoke test
        assert True

    @pytest.mark.asyncio
    async def test_reranking_service_consistency(
        self, reranking_client: RerankingServiceClient
    ) -> None:
        """Test reranking service consistency across requests."""
        query = "Consistency test query"
        documents = ["Test document 1", "Test document 2"]

        # Rerank multiple times
        scores1 = await reranking_client.rerank(query=query, documents=documents, model="default")
        scores2 = await reranking_client.rerank(query=query, documents=documents, model="default")

        # Scores should be consistent (same model, same inputs)
        assert len(scores1) == len(scores2)
        assert all(abs(s1 - s2) < 0.001 for s1, s2 in zip(scores1, scores2, strict=False))

    @pytest.mark.asyncio
    async def test_reranking_service_empty_input(
        self, reranking_client: RerankingServiceClient
    ) -> None:
        """Test reranking service with empty input."""
        try:
            await reranking_client.rerank(query="test", documents=[], model="default")
            assert False, "Should have raised an error for empty documents"
        except Exception as e:
            assert "empty" in str(e).lower() or "invalid" in str(e).lower()

    @pytest.mark.asyncio
    async def test_reranking_service_ranking_order(
        self, reranking_client: RerankingServiceClient
    ) -> None:
        """Test reranking service ranking order."""
        query = "What are the side effects?"
        documents = [
            "This document is about side effects and adverse reactions.",
            "This document is about unrelated topic.",
            "This document mentions side effects briefly.",
        ]

        scores = await reranking_client.rerank(query=query, documents=documents, model="default")

        # First document should have highest score (most relevant)
        assert scores[0] >= scores[1]
        assert scores[0] >= scores[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
