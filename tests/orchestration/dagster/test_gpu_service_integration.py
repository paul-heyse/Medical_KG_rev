"""Integration tests for Dagster GPU service integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from dagster import build_op_context

from Medical_KG_rev.orchestration.dagster.assets.gpu_assets import (
    allocate_gpu_resources,
    batch_process_documents,
    check_gpu_service_health,
    generate_embeddings,
    get_gpu_service_status,
    rerank_documents,
)
from Medical_KG_rev.orchestration.dagster.jobs.gpu_processing_job import (
    batch_processing_job,
    embedding_generation_job,
    gpu_processing_pipeline,
    reranking_job,
)
from Medical_KG_rev.orchestration.dagster.resources.gpu_services import GPUServiceResource


class TestDagsterGPUServiceIntegration:
    """Integration tests for Dagster GPU service integration."""

    @pytest.fixture
    def mock_gpu_service_resource(self) -> GPUServiceResource:
        """Mock GPU service resource for testing."""
        resource = GPUServiceResource(
            gpu_service_address="localhost:50051",
            embedding_service_address="localhost:50052",
            reranking_service_address="localhost:50053",
            timeout=30,
            max_retries=3,
        )

        # Mock the gRPC clients
        resource.gpu_client = Mock()
        resource.embedding_client = Mock()
        resource.reranking_client = Mock()

        return resource

    @pytest.fixture
    def op_context(self, mock_gpu_service_resource: GPUServiceResource) -> OpExecutionContext:
        """Create Dagster operation context for testing."""
        return build_op_context(resources={"gpu_service": mock_gpu_service_resource})

    @pytest.mark.asyncio
    async def test_check_gpu_service_health_success(
        self, op_context: OpExecutionContext, mock_gpu_service_resource: GPUServiceResource
    ) -> None:
        """Test successful GPU service health check."""
        # Mock health check responses
        mock_gpu_service_resource.gpu_client.health_check = AsyncMock(
            return_value=Mock(status="SERVING")
        )
        mock_gpu_service_resource.embedding_client.health_check = AsyncMock(
            return_value=Mock(status="SERVING")
        )
        mock_gpu_service_resource.reranking_client.health_check = AsyncMock(
            return_value=Mock(status="SERVING")
        )

        # Test health check
        result = await check_gpu_service_health(op_context)

        assert result["status"] == "healthy"
        assert "services" in result
        assert "checked_at" in result

    @pytest.mark.asyncio
    async def test_check_gpu_service_health_failure(
        self, op_context: OpExecutionContext, mock_gpu_service_resource: GPUServiceResource
    ) -> None:
        """Test GPU service health check failure."""
        # Mock health check failure
        mock_gpu_service_resource.gpu_client.health_check = AsyncMock(
            return_value=Mock(status="NOT_SERVING")
        )
        mock_gpu_service_resource.embedding_client.health_check = AsyncMock(
            return_value=Mock(status="SERVING")
        )
        mock_gpu_service_resource.reranking_client.health_check = AsyncMock(
            return_value=Mock(status="SERVING")
        )

        # Test health check should raise exception
        with pytest.raises(RuntimeError, match="Unhealthy GPU services"):
            await check_gpu_service_health(op_context)

    @pytest.mark.asyncio
    async def test_get_gpu_service_status(
        self, op_context: OpExecutionContext, mock_gpu_service_resource: GPUServiceResource
    ) -> None:
        """Test getting GPU service status."""
        # Mock GPU status response
        mock_gpu_status = Mock()
        mock_gpu_status.available = True

        mock_device = Mock()
        mock_device.id = "0"
        mock_device.name = "NVIDIA GeForce RTX 4090"
        mock_device.memory_total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_device.memory_used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_device.utilization = 75.5

        mock_gpu_service_resource.gpu_client.get_gpu_status = AsyncMock(
            return_value=mock_gpu_status
        )
        mock_gpu_service_resource.gpu_client.list_devices = AsyncMock(return_value=[mock_device])

        # Test getting GPU status
        result = await get_gpu_service_status(op_context)

        assert "gpu_status" in result
        assert result["gpu_status"]["available"] is True
        assert result["gpu_status"]["device_count"] == 1
        assert len(result["gpu_status"]["devices"]) == 1
        assert result["gpu_status"]["devices"][0]["id"] == "0"

    @pytest.mark.asyncio
    async def test_allocate_gpu_resources_success(
        self, op_context: OpExecutionContext, mock_gpu_service_resource: GPUServiceResource
    ) -> None:
        """Test successful GPU resource allocation."""
        # Mock allocation response
        mock_allocation = Mock()
        mock_allocation.success = True
        mock_allocation.allocation_id = "alloc-123"
        mock_allocation.device_id = "0"

        mock_gpu_service_resource.gpu_client.allocate_gpu = AsyncMock(return_value=mock_allocation)

        # Test allocation
        result = await allocate_gpu_resources(op_context, memory_mb=1024)

        assert result["allocation_id"] == "alloc-123"
        assert result["memory_mb"] == 1024
        assert result["device_id"] == "0"

    @pytest.mark.asyncio
    async def test_allocate_gpu_resources_failure(
        self, op_context: OpExecutionContext, mock_gpu_service_resource: GPUServiceResource
    ) -> None:
        """Test GPU resource allocation failure."""
        # Mock allocation failure
        mock_allocation = Mock()
        mock_allocation.success = False

        mock_gpu_service_resource.gpu_client.allocate_gpu = AsyncMock(return_value=mock_allocation)

        # Test allocation should raise exception
        with pytest.raises(RuntimeError, match="Failed to allocate"):
            await allocate_gpu_resources(op_context, memory_mb=1024)

    @pytest.mark.asyncio
    async def test_generate_embeddings(
        self, op_context: OpExecutionContext, mock_gpu_service_resource: GPUServiceResource
    ) -> None:
        """Test embedding generation."""
        # Mock embedding response
        mock_embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]

        mock_gpu_service_resource.embedding_client.generate_embeddings = AsyncMock(
            return_value=mock_embeddings
        )

        # Test data
        chunked_documents = [
            {"document_id": "doc-1", "chunks": [{"text": "Test chunk 1"}, {"text": "Test chunk 2"}]}
        ]

        # Test embedding generation
        result = await generate_embeddings(op_context, chunked_documents)

        assert len(result) == 1
        assert result[0]["document_id"] == "doc-1"
        assert len(result[0]["embeddings"]) == 2
        assert result[0]["metadata"]["chunk_count"] == 2

    @pytest.mark.asyncio
    async def test_rerank_documents(
        self, op_context: OpExecutionContext, mock_gpu_service_resource: GPUServiceResource
    ) -> None:
        """Test document reranking."""
        # Mock reranking response
        mock_scores = [0.85, 0.72, 0.63]

        mock_gpu_service_resource.reranking_client.rerank = AsyncMock(return_value=mock_scores)

        # Test data
        embedded_documents = [
            {
                "document_id": "doc-1",
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "metadata": {},
            }
        ]
        query = "test query"

        # Test reranking
        result = await rerank_documents(op_context, embedded_documents, query)

        assert len(result) == 1
        assert result[0]["document_id"] == "doc-1"
        assert result[0]["query"] == query
        assert len(result[0]["scores"]) == 1
        assert result[0]["metadata"]["embedding_count"] == 2

    @pytest.mark.asyncio
    async def test_batch_process_documents(
        self, op_context: OpExecutionContext, mock_gpu_service_resource: GPUServiceResource
    ) -> None:
        """Test batch document processing."""
        # Mock responses
        mock_embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]
        mock_scores = [0.85, 0.72]

        mock_gpu_service_resource.embedding_client.generate_embeddings = AsyncMock(
            return_value=mock_embeddings
        )
        mock_gpu_service_resource.reranking_client.rerank = AsyncMock(return_value=mock_scores)

        # Test data
        chunked_documents = [
            {"document_id": "doc-1", "chunks": [{"text": "Test chunk 1"}]},
            {"document_id": "doc-2", "chunks": [{"text": "Test chunk 2"}]},
        ]
        query = "test query"

        # Test batch processing
        result = await batch_process_documents(op_context, chunked_documents, query)

        assert len(result) == 2
        assert result[0]["document_id"] == "doc-1"
        assert result[1]["document_id"] == "doc-2"
        assert all(doc["query"] == query for doc in result)
        assert all(doc["metadata"]["batch_processed"] for doc in result)

    @pytest.mark.asyncio
    async def test_gpu_service_resource_health_check(
        self, mock_gpu_service_resource: GPUServiceResource
    ) -> None:
        """Test GPU service resource health check."""
        # Mock health check responses
        mock_gpu_service_resource.gpu_client.health_check = AsyncMock(
            return_value=Mock(status="SERVING")
        )
        mock_gpu_service_resource.embedding_client.health_check = AsyncMock(
            return_value=Mock(status="SERVING")
        )
        mock_gpu_service_resource.reranking_client.health_check = AsyncMock(
            return_value=Mock(status="SERVING")
        )

        # Test health check
        health_status = mock_gpu_service_resource.health_check()

        assert health_status["gpu_service"]["status"] == "healthy"
        assert health_status["embedding_service"]["status"] == "healthy"
        assert health_status["reranking_service"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_gpu_service_resource_health_check_failure(
        self, mock_gpu_service_resource: GPUServiceResource
    ) -> None:
        """Test GPU service resource health check failure."""
        # Mock health check failure
        mock_gpu_service_resource.gpu_client.health_check = AsyncMock(
            side_effect=Exception("Connection failed")
        )
        mock_gpu_service_resource.embedding_client.health_check = AsyncMock(
            return_value=Mock(status="SERVING")
        )
        mock_gpu_service_resource.reranking_client.health_check = AsyncMock(
            return_value=Mock(status="SERVING")
        )

        # Test health check
        health_status = mock_gpu_service_resource.health_check()

        assert health_status["gpu_service"]["status"] == "error"
        assert health_status["gpu_service"]["error"] == "Connection failed"
        assert health_status["embedding_service"]["status"] == "healthy"
        assert health_status["reranking_service"]["status"] == "healthy"

    def test_gpu_processing_pipeline_config(self) -> None:
        """Test GPU processing pipeline configuration."""
        # Test that the pipeline has the correct resource configuration
        assert gpu_processing_pipeline.resource_defs["gpu_service"] is not None

        # Test that the pipeline has the correct config
        assert "resources" in gpu_processing_pipeline.config
        assert "gpu_service" in gpu_processing_pipeline.config["resources"]

    def test_embedding_generation_job_config(self) -> None:
        """Test embedding generation job configuration."""
        # Test that the job has the correct resource configuration
        assert embedding_generation_job.resource_defs["gpu_service"] is not None

        # Test that the job has the correct config
        assert "resources" in embedding_generation_job.config
        assert "gpu_service" in embedding_generation_job.config["resources"]

    def test_reranking_job_config(self) -> None:
        """Test reranking job configuration."""
        # Test that the job has the correct resource configuration
        assert reranking_job.resource_defs["gpu_service"] is not None

        # Test that the job has the correct config
        assert "resources" in reranking_job.config
        assert "gpu_service" in reranking_job.config["resources"]

    def test_batch_processing_job_config(self) -> None:
        """Test batch processing job configuration."""
        # Test that the job has the correct resource configuration
        assert batch_processing_job.resource_defs["gpu_service"] is not None

        # Test that the job has the correct config
        assert "resources" in batch_processing_job.config
        assert "gpu_service" in batch_processing_job.config["resources"]

    def test_gpu_job_environment_configs(self) -> None:
        """Test GPU job configurations for different environments."""
        from Medical_KG_rev.orchestration.dagster.jobs.gpu_processing_job import (
            create_gpu_job_for_environment,
            get_gpu_job_config,
        )

        # Test development environment
        dev_config = create_gpu_job_for_environment("development")
        assert dev_config["gpu_service_address"] == "localhost:50051"
        assert dev_config["timeout"] == 60
        assert dev_config["max_retries"] == 5

        # Test staging environment
        staging_config = create_gpu_job_for_environment("staging")
        assert staging_config["gpu_service_address"] == "gpu-services.staging:50051"
        assert staging_config["timeout"] == 45
        assert staging_config["max_retries"] == 3

        # Test production environment
        prod_config = create_gpu_job_for_environment("production")
        assert prod_config["gpu_service_address"] == "gpu-services.production:50051"
        assert prod_config["timeout"] == 30
        assert prod_config["max_retries"] == 3

        # Test job config generation
        job_config = get_gpu_job_config("production")
        assert "resources" in job_config
        assert "gpu_service" in job_config["resources"]
        assert "config" in job_config["resources"]["gpu_service"]

    def test_unknown_environment_config(self) -> None:
        """Test GPU job configuration for unknown environment."""
        from Medical_KG_rev.orchestration.dagster.jobs.gpu_processing_job import (
            create_gpu_job_for_environment,
        )

        with pytest.raises(ValueError, match="Unknown environment"):
            create_gpu_job_for_environment("unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
