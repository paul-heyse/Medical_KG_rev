"""Performance tests for service communication latency."""

from __future__ import annotations

import asyncio
import statistics
import time

import pytest

from Medical_KG_rev.services.embedding.grpc_client import EmbeddingServiceClient
from Medical_KG_rev.services.gpu.grpc_client import GPUServiceClient
from Medical_KG_rev.services.reranking.grpc_client import RerankingServiceClient


class TestServiceLatency:
    """Performance tests for service communication latency."""

    @pytest.fixture
    def gpu_client(self) -> GPUServiceClient:
        """GPU service client for testing."""
        return GPUServiceClient("localhost:50051")

    @pytest.fixture
    def embedding_client(self) -> EmbeddingServiceClient:
        """Embedding service client for testing."""
        return EmbeddingServiceClient("localhost:50051")

    @pytest.fixture
    def reranking_client(self) -> RerankingServiceClient:
        """Reranking service client for testing."""
        return RerankingServiceClient("localhost:50051")

    @pytest.fixture
    def sample_texts(self) -> list[str]:
        """Sample texts for embedding generation."""
        return [
            "This is a test sentence for embedding generation.",
            "Another test sentence to verify batch processing.",
            "Medical knowledge graph integration test.",
            "Performance testing for service communication.",
            "Latency measurement and optimization.",
        ]

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

    async def _measure_latency(self, coro, iterations: int = 10) -> dict:
        """Measure latency of an async operation."""
        latencies = []

        for _ in range(iterations):
            start_time = time.time()
            await coro
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        return {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[int(0.95 * len(latencies))],
            "p99": sorted(latencies)[int(0.99 * len(latencies))],
            "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        }

    @pytest.mark.asyncio
    async def test_gpu_service_latency(self, gpu_client: GPUServiceClient) -> None:
        """Test GPU service communication latency."""
        latency = await self._measure_latency(gpu_client.get_gpu_status(), iterations=20)

        print(f"GPU Service Latency: {latency}")

        # P95 latency should be < 500ms
        assert latency["p95"] < 500, f"P95 latency {latency['p95']}ms exceeds 500ms threshold"

        # Mean latency should be reasonable
        assert latency["mean"] < 200, f"Mean latency {latency['mean']}ms is too high"

    @pytest.mark.asyncio
    async def test_embedding_service_latency(
        self, embedding_client: EmbeddingServiceClient, sample_texts: list[str]
    ) -> None:
        """Test embedding service communication latency."""
        latency = await self._measure_latency(
            embedding_client.generate_embeddings(texts=sample_texts, model="default"), iterations=20
        )

        print(f"Embedding Service Latency: {latency}")

        # P95 latency should be < 500ms
        assert latency["p95"] < 500, f"P95 latency {latency['p95']}ms exceeds 500ms threshold"

        # Mean latency should be reasonable
        assert latency["mean"] < 300, f"Mean latency {latency['mean']}ms is too high"

    @pytest.mark.asyncio
    async def test_reranking_service_latency(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking service communication latency."""
        latency = await self._measure_latency(
            reranking_client.rerank(
                query=sample_query, documents=sample_documents, model="default"
            ),
            iterations=20,
        )

        print(f"Reranking Service Latency: {latency}")

        # P95 latency should be < 500ms
        assert latency["p95"] < 500, f"P95 latency {latency['p95']}ms exceeds 500ms threshold"

        # Mean latency should be reasonable
        assert latency["mean"] < 300, f"Mean latency {latency['mean']}ms is too high"

    @pytest.mark.asyncio
    async def test_concurrent_service_latency(
        self,
        gpu_client: GPUServiceClient,
        embedding_client: EmbeddingServiceClient,
        reranking_client: RerankingServiceClient,
        sample_texts: list[str],
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test service latency under concurrent load."""
        # Create concurrent requests
        tasks = [
            gpu_client.get_gpu_status(),
            embedding_client.generate_embeddings(texts=sample_texts, model="default"),
            reranking_client.rerank(
                query=sample_query, documents=sample_documents, model="default"
            ),
        ]

        latency = await self._measure_latency(asyncio.gather(*tasks), iterations=10)

        print(f"Concurrent Service Latency: {latency}")

        # P95 latency should still be < 500ms under concurrent load
        assert (
            latency["p95"] < 500
        ), f"P95 latency {latency['p95']}ms exceeds 500ms threshold under concurrent load"

    @pytest.mark.asyncio
    async def test_batch_processing_latency(self, embedding_client: EmbeddingServiceClient) -> None:
        """Test batch processing latency with different batch sizes."""
        batch_sizes = [1, 5, 10, 20, 50]
        results = {}

        for batch_size in batch_sizes:
            texts = [f"Test sentence {i}" for i in range(batch_size)]

            latency = await self._measure_latency(
                embedding_client.generate_embeddings(texts=texts, model="default"), iterations=5
            )

            results[batch_size] = latency
            print(f"Batch size {batch_size}: {latency}")

        # Latency should scale reasonably with batch size
        # P95 should not exceed 500ms even for larger batches
        for batch_size, latency in results.items():
            assert (
                latency["p95"] < 500
            ), f"P95 latency {latency['p95']}ms exceeds 500ms for batch size {batch_size}"

    @pytest.mark.asyncio
    async def test_service_latency_consistency(self, gpu_client: GPUServiceClient) -> None:
        """Test service latency consistency across multiple calls."""
        latencies = []

        for _ in range(50):
            start_time = time.time()
            await gpu_client.get_gpu_status()
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate consistency metrics
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies)
        cv = std_latency / mean_latency if mean_latency > 0 else 0

        print(
            f"Latency Consistency - Mean: {mean_latency:.2f}ms, Std: {std_latency:.2f}ms, CV: {cv:.3f}"
        )

        # Coefficient of variation should be < 0.5 (50% variation)
        assert cv < 0.5, f"Latency coefficient of variation {cv:.3f} is too high"

    @pytest.mark.asyncio
    async def test_service_latency_under_load(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test service latency under sustained load."""
        texts = ["Test sentence for load testing"]
        latencies = []

        # Run sustained load for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
            request_start = time.time()
            await embedding_client.generate_embeddings(texts=texts, model="default")
            request_end = time.time()

            latency_ms = (request_end - request_start) * 1000
            latencies.append(latency_ms)

        # Calculate metrics
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        mean_latency = statistics.mean(latencies)

        print(f"Sustained Load Latency - Mean: {mean_latency:.2f}ms, P95: {p95_latency:.2f}ms")

        # P95 latency should still be < 500ms under sustained load
        assert (
            p95_latency < 500
        ), f"P95 latency {p95_latency:.2f}ms exceeds 500ms under sustained load"

    @pytest.mark.asyncio
    async def test_service_latency_degradation(self, gpu_client: GPUServiceClient) -> None:
        """Test service latency degradation over time."""
        latencies = []

        # Measure latency over 100 requests
        for i in range(100):
            start_time = time.time()
            await gpu_client.get_gpu_status()
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Check for latency degradation (first 20 vs last 20 requests)
        first_20_mean = statistics.mean(latencies[:20])
        last_20_mean = statistics.mean(latencies[-20:])
        degradation = (last_20_mean - first_20_mean) / first_20_mean

        print(
            f"Latency Degradation - First 20: {first_20_mean:.2f}ms, Last 20: {last_20_mean:.2f}ms, Degradation: {degradation:.3f}"
        )

        # Degradation should be < 20%
        assert degradation < 0.2, f"Latency degradation {degradation:.3f} exceeds 20% threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
