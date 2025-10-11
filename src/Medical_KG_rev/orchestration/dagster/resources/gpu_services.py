"""Dagster resources for GPU services."""

from __future__ import annotations

from typing import Any

from dagster import ConfigurableResource, get_dagster_logger
from pydantic import Field

from Medical_KG_rev.services.embedding.grpc_client import EmbeddingServiceClient
from Medical_KG_rev.services.gpu.grpc_client import GPUServiceClient
from Medical_KG_rev.services.reranking.grpc_client import RerankingServiceClient



class GPUServiceResource(ConfigurableResource):
    """Dagster resource providing gRPC clients for GPU services."""

    gpu_service_address: str = Field(
        default="gpu-services:50051", description="GPU service gRPC endpoint"
    )
    embedding_service_address: str = Field(
        default="embedding-services:50051", description="Embedding service gRPC endpoint"
    )
    reranking_service_address: str = Field(
        default="reranking-services:50051", description="Reranking service gRPC endpoint"
    )
    timeout: int = Field(default=30, description="gRPC call timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed calls")

    def setup_for_execution(self, context) -> None:
        """Setup GPU service clients for execution."""
        logger = get_dagster_logger()

        try:
            # Initialize gRPC clients
            self.gpu_client = GPUServiceClient(
                service_address=self.gpu_service_address,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

            self.embedding_client = EmbeddingServiceClient(
                service_address=self.embedding_service_address,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

            self.reranking_client = RerankingServiceClient(
                service_address=self.reranking_service_address,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

            logger.info(f"GPU service clients initialized for execution {context.run_id}")

        except Exception as e:
            logger.error(f"Failed to initialize GPU service clients: {e}")
            raise

    def teardown_after_execution(self, context) -> None:
        """Teardown GPU service clients after execution."""
        logger = get_dagster_logger()

        try:
            # Close gRPC channels
            if hasattr(self, "gpu_client"):
                self.gpu_client.close()
            if hasattr(self, "embedding_client"):
                self.embedding_client.close()
            if hasattr(self, "reranking_client"):
                self.reranking_client.close()

            logger.info(f"GPU service clients closed for execution {context.run_id}")

        except Exception as e:
            logger.warning(f"Error closing GPU service clients: {e}")

    def get_gpu_client(self) -> GPUServiceClient:
        """Get GPU service client."""
        if not hasattr(self, "gpu_client"):
            raise RuntimeError(
                "GPU service client not initialized. Call setup_for_execution first."
            )
        return self.gpu_client

    def get_embedding_client(self) -> EmbeddingServiceClient:
        """Get embedding service client."""
        if not hasattr(self, "embedding_client"):
            raise RuntimeError(
                "Embedding service client not initialized. Call setup_for_execution first."
            )
        return self.embedding_client

    def get_reranking_client(self) -> RerankingServiceClient:
        """Get reranking service client."""
        if not hasattr(self, "reranking_client"):
            raise RuntimeError(
                "Reranking service client not initialized. Call setup_for_execution first."
            )
        return self.reranking_client

    def health_check(self) -> dict[str, Any]:
        """Check health of all GPU services."""
        health_status = {
            "gpu_service": {"status": "unknown", "error": None},
            "embedding_service": {"status": "unknown", "error": None},
            "reranking_service": {"status": "unknown", "error": None},
        }

        # Check GPU service health
        try:
            if hasattr(self, "gpu_client"):
                gpu_health = self.gpu_client.health_check()
                health_status["gpu_service"]["status"] = (
                    "healthy" if gpu_health.status == "SERVING" else "unhealthy"
                )
            else:
                health_status["gpu_service"]["status"] = "not_initialized"
        except Exception as e:
            health_status["gpu_service"]["status"] = "error"
            health_status["gpu_service"]["error"] = str(e)

        # Check embedding service health
        try:
            if hasattr(self, "embedding_client"):
                embedding_health = self.embedding_client.health_check()
                health_status["embedding_service"]["status"] = (
                    "healthy" if embedding_health.status == "SERVING" else "unhealthy"
                )
            else:
                health_status["embedding_service"]["status"] = "not_initialized"
        except Exception as e:
            health_status["embedding_service"]["status"] = "error"
            health_status["embedding_service"]["error"] = str(e)

        # Check reranking service health
        try:
            if hasattr(self, "reranking_client"):
                reranking_health = self.reranking_client.health_check()
                health_status["reranking_service"]["status"] = (
                    "healthy" if reranking_health.status == "SERVING" else "unhealthy"
                )
            else:
                health_status["reranking_service"]["status"] = "not_initialized"
        except Exception as e:
            health_status["reranking_service"]["status"] = "error"
            health_status["reranking_service"]["error"] = str(e)

        return health_status


class GPUServiceResourceConfig:
    """Configuration for GPU service resource."""

    def __init__(
        self,
        gpu_service_address: str = "gpu-services:50051",
        embedding_service_address: str = "embedding-services:50051",
        reranking_service_address: str = "reranking-services:50051",
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """Initialize GPU service resource configuration."""
        self.gpu_service_address = gpu_service_address
        self.embedding_service_address = embedding_service_address
        self.reranking_service_address = reranking_service_address
        self.timeout = timeout
        self.max_retries = max_retries

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "gpu_service_address": self.gpu_service_address,
            "embedding_service_address": self.embedding_service_address,
            "reranking_service_address": self.reranking_service_address,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> GPUServiceResourceConfig:
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def for_environment(cls, environment: str) -> GPUServiceResourceConfig:
        """Create configuration for specific environment."""
        if environment == "development":
            return cls(
                gpu_service_address="localhost:50051",
                embedding_service_address="localhost:50052",
                reranking_service_address="localhost:50053",
                timeout=60,
                max_retries=5,
            )
        elif environment == "staging":
            return cls(
                gpu_service_address="gpu-services.staging:50051",
                embedding_service_address="embedding-services.staging:50051",
                reranking_service_address="reranking-services.staging:50051",
                timeout=45,
                max_retries=3,
            )
        elif environment == "production":
            return cls(
                gpu_service_address="gpu-services.production:50051",
                embedding_service_address="embedding-services.production:50051",
                reranking_service_address="reranking-services.production:50051",
                timeout=30,
                max_retries=3,
            )
        else:
            raise ValueError(f"Unknown environment: {environment}")


# Global resource instance for reuse
gpu_service_resource = GPUServiceResource()


def get_gpu_service_resource() -> GPUServiceResource:
    """Get the global GPU service resource."""
    return gpu_service_resource


def configure_gpu_service_resource(
    gpu_service_address: str = "gpu-services:50051",
    embedding_service_address: str = "embedding-services:50051",
    reranking_service_address: str = "reranking-services:50051",
    timeout: int = 30,
    max_retries: int = 3,
) -> GPUServiceResource:
    """Configure and return GPU service resource."""
    return GPUServiceResource(
        gpu_service_address=gpu_service_address,
        embedding_service_address=embedding_service_address,
        reranking_service_address=reranking_service_address,
        timeout=timeout,
        max_retries=max_retries,
    )
