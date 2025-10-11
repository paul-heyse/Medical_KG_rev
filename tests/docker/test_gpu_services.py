"""Docker service tests for GPU services."""

from __future__ import annotations

import subprocess
import time

import pytest

from Medical_KG_rev.services.gpu.grpc_client import GPUServiceClient


class TestGPUServicesDocker:
    """Docker service tests for GPU services."""

    @pytest.fixture
    def gpu_service_url(self) -> str:
        """GPU service URL for testing."""
        return "localhost:50051"

    @pytest.fixture
    def gpu_client(self, gpu_service_url: str) -> GPUServiceClient:
        """GPU service client for testing."""
        return GPUServiceClient(gpu_service_url)

    def test_gpu_services_container_running(self) -> None:
        """Test that GPU services container is running."""
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=gpu-services", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker command failed"
        assert "Up" in result.stdout, "GPU services container is not running"

    def test_gpu_services_container_health(self) -> None:
        """Test GPU services container health check."""
        result = subprocess.run(
            ["docker", "inspect", "gpu-services", "--format", "{{.State.Health.Status}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"
        assert "healthy" in result.stdout.lower(), "GPU services container is not healthy"

    def test_gpu_services_container_logs(self) -> None:
        """Test GPU services container logs for errors."""
        result = subprocess.run(
            ["docker", "logs", "gpu-services", "--tail", "50"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Docker logs command failed"

        # Check for common error patterns
        error_patterns = ["error", "exception", "failed", "critical"]
        logs_lower = result.stdout.lower()

        for pattern in error_patterns:
            assert (
                pattern not in logs_lower
            ), f"Found error pattern '{pattern}' in GPU services logs"

    def test_gpu_services_container_resources(self) -> None:
        """Test GPU services container resource usage."""
        result = subprocess.run(
            [
                "docker",
                "stats",
                "gpu-services",
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

        print(f"GPU Services Resource Usage - CPU: {cpu_percent}%, Memory: {mem_usage}")

        # CPU usage should be reasonable (< 80%)
        assert cpu_percent < 80, f"CPU usage {cpu_percent}% is too high"

    @pytest.mark.asyncio
    async def test_gpu_services_grpc_endpoint(self, gpu_client: GPUServiceClient) -> None:
        """Test GPU services gRPC endpoint."""
        try:
            status = await gpu_client.get_gpu_status()
            assert status is not None
        except Exception as e:
            pytest.fail(f"GPU services gRPC endpoint failed: {e}")

    @pytest.mark.asyncio
    async def test_gpu_services_health_check(self, gpu_client: GPUServiceClient) -> None:
        """Test GPU services health check endpoint."""
        try:
            health = await gpu_client.health_check()
            assert health is not None
            assert hasattr(health, "status")
        except Exception as e:
            pytest.fail(f"GPU services health check failed: {e}")

    @pytest.mark.asyncio
    async def test_gpu_services_device_listing(self, gpu_client: GPUServiceClient) -> None:
        """Test GPU services device listing."""
        try:
            devices = await gpu_client.list_devices()
            assert devices is not None
            assert isinstance(devices, list)
        except Exception as e:
            pytest.fail(f"GPU services device listing failed: {e}")

    @pytest.mark.asyncio
    async def test_gpu_services_allocation(self, gpu_client: GPUServiceClient) -> None:
        """Test GPU services allocation."""
        try:
            allocation = await gpu_client.allocate_gpu(memory_mb=1024)
            assert allocation is not None
            assert hasattr(allocation, "success")
        except Exception as e:
            pytest.fail(f"GPU services allocation failed: {e}")

    def test_gpu_services_container_restart(self) -> None:
        """Test GPU services container restart capability."""
        # Restart the container
        result = subprocess.run(
            ["docker", "restart", "gpu-services"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Container restart failed"

        # Wait for container to be ready
        time.sleep(30)

        # Check if container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=gpu-services", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
        )

        assert "Up" in result.stdout, "GPU services container failed to restart"

    def test_gpu_services_container_logs_after_restart(self) -> None:
        """Test GPU services container logs after restart."""
        result = subprocess.run(
            ["docker", "logs", "gpu-services", "--tail", "20"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Docker logs command failed"

        # Check for successful startup messages
        logs_lower = result.stdout.lower()
        startup_patterns = ["started", "ready", "listening", "serving"]

        startup_found = any(pattern in logs_lower for pattern in startup_patterns)
        assert startup_found, "No startup patterns found in GPU services logs after restart"

    def test_gpu_services_container_environment(self) -> None:
        """Test GPU services container environment variables."""
        result = subprocess.run(
            ["docker", "exec", "gpu-services", "env"], capture_output=True, text=True
        )

        assert result.returncode == 0, "Docker exec command failed"

        env_vars = result.stdout
        assert "CUDA_VISIBLE_DEVICES" in env_vars, "CUDA_VISIBLE_DEVICES not set"
        assert "GPU_MEMORY_FRACTION" in env_vars, "GPU_MEMORY_FRACTION not set"

    def test_gpu_services_container_volumes(self) -> None:
        """Test GPU services container volume mounts."""
        result = subprocess.run(
            ["docker", "inspect", "gpu-services", "--format", "{{.Mounts}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"

        mounts = result.stdout
        assert "models" in mounts.lower(), "Models volume not mounted"
        assert "config" in mounts.lower(), "Config volume not mounted"

    def test_gpu_services_container_network(self) -> None:
        """Test GPU services container network configuration."""
        result = subprocess.run(
            ["docker", "inspect", "gpu-services", "--format", "{{.NetworkSettings.Ports}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"

        ports = result.stdout
        assert "50051" in ports, "gRPC port 50051 not exposed"

    @pytest.mark.asyncio
    async def test_gpu_services_performance(self, gpu_client: GPUServiceClient) -> None:
        """Test GPU services performance."""
        start_time = time.time()

        # Make multiple requests
        for _ in range(10):
            await gpu_client.get_gpu_status()

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10

        print(f"GPU Services Performance - Total: {total_time:.2f}s, Average: {avg_time:.2f}s")

        # Average response time should be < 100ms
        assert avg_time < 0.1, f"Average response time {avg_time:.2f}s is too slow"

    def test_gpu_services_container_security(self) -> None:
        """Test GPU services container security configuration."""
        result = subprocess.run(
            ["docker", "inspect", "gpu-services", "--format", "{{.HostConfig.SecurityOpt}}"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Docker inspect command failed"

        # Check for security options (if any)
        security_opts = result.stdout
        print(f"GPU Services Security Options: {security_opts}")

    def test_gpu_services_container_limits(self) -> None:
        """Test GPU services container resource limits."""
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "gpu-services",
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

        print(f"GPU Services Limits - Memory: {memory_limit}, CPU: {cpu_limit}")

        # Check that limits are set
        assert memory_limit != "0", "Memory limit not set"
        assert cpu_limit != "0", "CPU limit not set"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
