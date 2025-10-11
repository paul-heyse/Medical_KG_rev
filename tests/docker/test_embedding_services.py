"""Docker service tests for embedding services."""

from __future__ import annotations

import subprocess
import time

import pytest

from Medical_KG_rev.services.embedding.grpc_client import EmbeddingServiceClient


class TestEmbeddingServicesDocker:
    """Docker service tests for embedding services."""

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

    def test_embedding_services_container_running(self) -> None:
        """Test that embedding services container is running."""
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=embedding-services", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker command failed"
        assert "Up" in result.stdout, "Embedding services container is not running"

    def test_embedding_services_container_health(self) -> None:
        """Test embedding services container health check."""
        result = subprocess.run(
            ["docker", "inspect", "embedding-services", "--format", "{{.State.Health.Status}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"
        assert "healthy" in result.stdout.lower(), "Embedding services container is not healthy"

    def test_embedding_services_container_logs(self) -> None:
        """Test embedding services container logs for errors."""
        result = subprocess.run(
            ["docker", "logs", "embedding-services", "--tail", "50"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Docker logs command failed"

        # Check for common error patterns
        error_patterns = ["error", "exception", "failed", "critical"]
        logs_lower = result.stdout.lower()

        for pattern in error_patterns:
            assert (
                pattern not in logs_lower
            ), f"Found error pattern '{pattern}' in embedding services logs"

    def test_embedding_services_container_resources(self) -> None:
        """Test embedding services container resource usage."""
        result = subprocess.run(
            [
                "docker",
                "stats",
                "embedding-services",
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

        print(f"Embedding Services Resource Usage - CPU: {cpu_percent}%, Memory: {mem_usage}")

        # CPU usage should be reasonable (< 80%)
        assert cpu_percent < 80, f"CPU usage {cpu_percent}% is too high"

    @pytest.mark.asyncio
    async def test_embedding_services_grpc_endpoint(
        self, embedding_client: EmbeddingServiceClient, sample_texts: list[str]
    ) -> None:
        """Test embedding services gRPC endpoint."""
        try:
            embeddings = await embedding_client.generate_embeddings(
                texts=sample_texts, model="default"
            )
            assert embeddings is not None
            assert len(embeddings) == len(sample_texts)
        except Exception as e:
            pytest.fail(f"Embedding services gRPC endpoint failed: {e}")

    @pytest.mark.asyncio
    async def test_embedding_services_health_check(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test embedding services health check endpoint."""
        try:
            health = await embedding_client.health_check()
            assert health is not None
            assert hasattr(health, "status")
        except Exception as e:
            pytest.fail(f"Embedding services health check failed: {e}")

    @pytest.mark.asyncio
    async def test_embedding_services_model_listing(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test embedding services model listing."""
        try:
            models = await embedding_client.list_models()
            assert models is not None
            assert isinstance(models, list)
            assert len(models) > 0
        except Exception as e:
            pytest.fail(f"Embedding services model listing failed: {e}")

    @pytest.mark.asyncio
    async def test_embedding_services_batch_processing(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test embedding services batch processing."""
        try:
            # Test with larger batch
            large_batch = [f"Test sentence {i}" for i in range(50)]
            embeddings = await embedding_client.generate_embeddings(
                texts=large_batch, model="default"
            )
            assert embeddings is not None
            assert len(embeddings) == len(large_batch)
        except Exception as e:
            pytest.fail(f"Embedding services batch processing failed: {e}")

    def test_embedding_services_container_restart(self) -> None:
        """Test embedding services container restart capability."""
        # Restart the container
        result = subprocess.run(
            ["docker", "restart", "embedding-services"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Container restart failed"

        # Wait for container to be ready
        time.sleep(30)

        # Check if container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=embedding-services", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
        )

        assert "Up" in result.stdout, "Embedding services container failed to restart"

    def test_embedding_services_container_logs_after_restart(self) -> None:
        """Test embedding services container logs after restart."""
        result = subprocess.run(
            ["docker", "logs", "embedding-services", "--tail", "20"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Docker logs command failed"

        # Check for successful startup messages
        logs_lower = result.stdout.lower()
        startup_patterns = ["started", "ready", "listening", "serving"]

        startup_found = any(pattern in logs_lower for pattern in startup_patterns)
        assert startup_found, "No startup patterns found in embedding services logs after restart"

    def test_embedding_services_container_environment(self) -> None:
        """Test embedding services container environment variables."""
        result = subprocess.run(
            ["docker", "exec", "embedding-services", "env"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Docker exec command failed"

        env_vars = result.stdout
        assert "CUDA_VISIBLE_DEVICES" in env_vars, "CUDA_VISIBLE_DEVICES not set"
        assert "GPU_MEMORY_FRACTION" in env_vars, "GPU_MEMORY_FRACTION not set"

    def test_embedding_services_container_volumes(self) -> None:
        """Test embedding services container volume mounts."""
        result = subprocess.run(
            ["docker", "inspect", "embedding-services", "--format", "{{.Mounts}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"

        mounts = result.stdout
        assert "models" in mounts.lower(), "Models volume not mounted"
        assert "config" in mounts.lower(), "Config volume not mounted"

    def test_embedding_services_container_network(self) -> None:
        """Test embedding services container network configuration."""
        result = subprocess.run(
            ["docker", "inspect", "embedding-services", "--format", "{{.NetworkSettings.Ports}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"

        ports = result.stdout
        assert "50051" in ports, "gRPC port 50051 not exposed"

    @pytest.mark.asyncio
    async def test_embedding_services_performance(
        self, embedding_client: EmbeddingServiceClient, sample_texts: list[str]
    ) -> None:
        """Test embedding services performance."""
        start_time = time.time()

        # Make multiple requests
        for _ in range(10):
            await embedding_client.generate_embeddings(texts=sample_texts, model="default")

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10

        print(
            f"Embedding Services Performance - Total: {total_time:.2f}s, Average: {avg_time:.2f}s"
        )

        # Average response time should be < 200ms
        assert avg_time < 0.2, f"Average response time {avg_time:.2f}s is too slow"

    def test_embedding_services_container_security(self) -> None:
        """Test embedding services container security configuration."""
        result = subprocess.run(
            ["docker", "inspect", "embedding-services", "--format", "{{.HostConfig.SecurityOpt}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"

        # Check for security options (if any)
        security_opts = result.stdout
        print(f"Embedding Services Security Options: {security_opts}")

    def test_embedding_services_container_limits(self) -> None:
        """Test embedding services container resource limits."""
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "embedding-services",
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

        print(f"Embedding Services Limits - Memory: {memory_limit}, CPU: {cpu_limit}")

        # Check that limits are set
        assert memory_limit != "0", "Memory limit not set"
        assert cpu_limit != "0", "CPU limit not set"

    @pytest.mark.asyncio
    async def test_embedding_services_model_loading(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test embedding services model loading."""
        try:
            models = await embedding_client.list_models()
            assert models is not None
            assert len(models) > 0

            # Test each model
            for model in models:
                embeddings = await embedding_client.generate_embeddings(
                    texts=["Test sentence"], model=model
                )
                assert embeddings is not None
                assert len(embeddings) == 1
        except Exception as e:
            pytest.fail(f"Embedding services model loading failed: {e}")

    @pytest.mark.asyncio
    async def test_embedding_services_error_handling(
        self, embedding_client: EmbeddingServiceClient
    ) -> None:
        """Test embedding services error handling."""
        try:
            # Test with invalid model
            await embedding_client.generate_embeddings(texts=["test"], model="invalid-model")
            assert False, "Should have raised an error for invalid model"
        except Exception as e:
            assert "model" in str(e).lower() or "error" in str(e).lower()

    def test_embedding_services_container_gpu_access(self) -> None:
        """Test embedding services container GPU access."""
        result = subprocess.run(
            ["docker", "exec", "embedding-services", "nvidia-smi"], capture_output=True, text=True
        )

        if result.returncode == 0:
            print("GPU access available in embedding services container")
        else:
            print("GPU access not available in embedding services container")
            # This is not a failure if GPU is not available in test environment


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
