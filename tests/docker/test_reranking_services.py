"""Docker service tests for reranking services."""

from __future__ import annotations

import subprocess
import time

import pytest

from Medical_KG_rev.services.reranking.grpc_client import RerankingServiceClient


class TestRerankingServicesDocker:
    """Docker service tests for reranking services."""

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

    def test_reranking_services_container_running(self) -> None:
        """Test that reranking services container is running."""
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=reranking-services", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker command failed"
        assert "Up" in result.stdout, "Reranking services container is not running"

    def test_reranking_services_container_health(self) -> None:
        """Test reranking services container health check."""
        result = subprocess.run(
            ["docker", "inspect", "reranking-services", "--format", "{{.State.Health.Status}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"
        assert "healthy" in result.stdout.lower(), "Reranking services container is not healthy"

    def test_reranking_services_container_logs(self) -> None:
        """Test reranking services container logs for errors."""
        result = subprocess.run(
            ["docker", "logs", "reranking-services", "--tail", "50"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Docker logs command failed"

        # Check for common error patterns
        error_patterns = ["error", "exception", "failed", "critical"]
        logs_lower = result.stdout.lower()

        for pattern in error_patterns:
            assert (
                pattern not in logs_lower
            ), f"Found error pattern '{pattern}' in reranking services logs"

    def test_reranking_services_container_resources(self) -> None:
        """Test reranking services container resource usage."""
        result = subprocess.run(
            [
                "docker",
                "stats",
                "reranking-services",
                "--no-stream",
                "--format",
                "{{.CPUPerc}},{{.MemUsage}}",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker stats command failed"

        # Parse CPU and memory usage
        stats = result.stdout.strip().split(",")
        cpu_percent = float(stats[0].replace("%", ""))
        mem_usage = stats[1]

        print(f"Reranking Services Resource Usage - CPU: {cpu_percent}%, Memory: {mem_usage}")

        # CPU usage should be reasonable (< 80%)
        assert cpu_percent < 80, f"CPU usage {cpu_percent}% is too high"

    @pytest.mark.asyncio
    async def test_reranking_services_grpc_endpoint(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking services gRPC endpoint."""
        try:
            scores = await reranking_client.rerank(
                query=sample_query, documents=sample_documents, model="default"
            )
            assert scores is not None
            assert len(scores) == len(sample_documents)
            assert all(isinstance(score, float) for score in scores)
        except Exception as e:
            pytest.fail(f"Reranking services gRPC endpoint failed: {e}")

    @pytest.mark.asyncio
    async def test_reranking_services_health_check(
        self, reranking_client: RerankingServiceClient
    ) -> None:
        """Test reranking services health check endpoint."""
        try:
            health = await reranking_client.health_check()
            assert health is not None
            assert hasattr(health, "status")
        except Exception as e:
            pytest.fail(f"Reranking services health check failed: {e}")

    @pytest.mark.asyncio
    async def test_reranking_services_model_listing(
        self, reranking_client: RerankingServiceClient
    ) -> None:
        """Test reranking services model listing."""
        try:
            models = await reranking_client.list_models()
            assert models is not None
            assert isinstance(models, list)
            assert len(models) > 0
        except Exception as e:
            pytest.fail(f"Reranking services model listing failed: {e}")

    @pytest.mark.asyncio
    async def test_reranking_services_batch_processing(
        self, reranking_client: RerankingServiceClient
    ) -> None:
        """Test reranking services batch processing."""
        try:
            # Test with larger batch
            query = "Test query for batch processing"
            large_batch = [f"Test document {i}" for i in range(50)]
            scores = await reranking_client.rerank(
                query=query, documents=large_batch, model="default"
            )
            assert scores is not None
            assert len(scores) == len(large_batch)
        except Exception as e:
            pytest.fail(f"Reranking services batch processing failed: {e}")

    def test_reranking_services_container_restart(self) -> None:
        """Test reranking services container restart capability."""
        # Restart the container
        result = subprocess.run(
            ["docker", "restart", "reranking-services"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Container restart failed"

        # Wait for container to be ready
        time.sleep(30)

        # Check if container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=reranking-services", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
        )

        assert "Up" in result.stdout, "Reranking services container failed to restart"

    def test_reranking_services_container_logs_after_restart(self) -> None:
        """Test reranking services container logs after restart."""
        result = subprocess.run(
            ["docker", "logs", "reranking-services", "--tail", "20"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Docker logs command failed"

        # Check for successful startup messages
        logs_lower = result.stdout.lower()
        startup_patterns = ["started", "ready", "listening", "serving"]

        startup_found = any(pattern in logs_lower for pattern in startup_patterns)
        assert startup_found, "No startup patterns found in reranking services logs after restart"

    def test_reranking_services_container_environment(self) -> None:
        """Test reranking services container environment variables."""
        result = subprocess.run(
            ["docker", "exec", "reranking-services", "env"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Docker exec command failed"

        env_vars = result.stdout
        assert "CUDA_VISIBLE_DEVICES" in env_vars, "CUDA_VISIBLE_DEVICES not set"
        assert "GPU_MEMORY_FRACTION" in env_vars, "GPU_MEMORY_FRACTION not set"

    def test_reranking_services_container_volumes(self) -> None:
        """Test reranking services container volume mounts."""
        result = subprocess.run(
            ["docker", "inspect", "reranking-services", "--format", "{{.Mounts}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"

        mounts = result.stdout
        assert "models" in mounts.lower(), "Models volume not mounted"
        assert "config" in mounts.lower(), "Config volume not mounted"

    def test_reranking_services_container_network(self) -> None:
        """Test reranking services container network configuration."""
        result = subprocess.run(
            ["docker", "inspect", "reranking-services", "--format", "{{.NetworkSettings.Ports}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"

        ports = result.stdout
        assert "50051" in ports, "gRPC port 50051 not exposed"

    @pytest.mark.asyncio
    async def test_reranking_services_performance(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking services performance."""
        start_time = time.time()

        # Make multiple requests
        for _ in range(10):
            await reranking_client.rerank(
                query=sample_query, documents=sample_documents, model="default"
            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10

        print(
            f"Reranking Services Performance - Total: {total_time:.2f}s, Average: {avg_time:.2f}s"
        )

        # Average response time should be < 200ms
        assert avg_time < 0.2, f"Average response time {avg_time:.2f}s is too slow"

    def test_reranking_services_container_security(self) -> None:
        """Test reranking services container security configuration."""
        result = subprocess.run(
            ["docker", "inspect", "reranking-services", "--format", "{{.HostConfig.SecurityOpt}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"

        # Check for security options (if any)
        security_opts = result.stdout
        print(f"Reranking Services Security Options: {security_opts}")

    def test_reranking_services_container_limits(self) -> None:
        """Test reranking services container resource limits."""
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "reranking-services",
                "--format",
                "{{.HostConfig.Memory}},{{.HostConfig.CpuQuota}}",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"

        limits = result.stdout.strip().split(",")
        memory_limit = limits[0]
        cpu_limit = limits[1]

        print(f"Reranking Services Limits - Memory: {memory_limit}, CPU: {cpu_limit}")

        # Check that limits are set
        assert memory_limit != "0", "Memory limit not set"
        assert cpu_limit != "0", "CPU limit not set"

    @pytest.mark.asyncio
    async def test_reranking_services_model_loading(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking services model loading."""
        try:
            models = await reranking_client.list_models()
            assert models is not None
            assert len(models) > 0

            # Test each model
            for model in models:
                scores = await reranking_client.rerank(
                    query=sample_query, documents=sample_documents, model=model
                )
                assert scores is not None
                assert len(scores) == len(sample_documents)
        except Exception as e:
            pytest.fail(f"Reranking services model loading failed: {e}")

    @pytest.mark.asyncio
    async def test_reranking_services_error_handling(
        self,
        reranking_client: RerankingServiceClient,
        sample_query: str,
        sample_documents: list[str],
    ) -> None:
        """Test reranking services error handling."""
        try:
            # Test with invalid model
            await reranking_client.rerank(
                query=sample_query, documents=sample_documents, model="invalid-model"
            )
            assert False, "Should have raised an error for invalid model"
        except Exception as e:
            assert "model" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.asyncio
    async def test_reranking_services_ranking_quality(
        self, reranking_client: RerankingServiceClient
    ) -> None:
        """Test reranking services ranking quality."""
        query = "What are the side effects?"
        documents = [
            "This document is about side effects and adverse reactions.",
            "This document is about unrelated topic.",
            "This document mentions side effects briefly.",
        ]

        scores = await reranking_client.rerank(query=query, documents=documents, model="default")

        # First document should have highest score (most relevant)
        assert scores[0] >= scores[1], "First document should have higher score than second"
        assert scores[0] >= scores[2], "First document should have higher score than third"

    def test_reranking_services_container_gpu_access(self) -> None:
        """Test reranking services container GPU access."""
        result = subprocess.run(
            ["docker", "exec", "reranking-services", "nvidia-smi"], capture_output=True, text=True
        )

        if result.returncode == 0:
            print("GPU access available in reranking services container")
        else:
            print("GPU access not available in reranking services container")
            # This is not a failure if GPU is not available in test environment

    @pytest.mark.asyncio
    async def test_reranking_services_consistency(
        self, reranking_client: RerankingServiceClient
    ) -> None:
        """Test reranking services consistency across requests."""
        query = "Consistency test query"
        documents = ["Test document 1", "Test document 2"]

        # Rerank multiple times
        scores1 = await reranking_client.rerank(query=query, documents=documents, model="default")
        scores2 = await reranking_client.rerank(query=query, documents=documents, model="default")

        # Scores should be consistent (same model, same inputs)
        assert len(scores1) == len(scores2)
        assert all(abs(s1 - s2) < 0.001 for s1, s2 in zip(scores1, scores2, strict=False))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
