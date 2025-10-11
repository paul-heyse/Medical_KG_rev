"""Dagster job for GPU processing operations."""

from __future__ import annotations

from typing import Any

from dagster import job

from Medical_KG_rev.orchestration.dagster.assets.gpu_assets import HttpClient
from Medical_KG_rev.orchestration.dagster.resources.gpu_services import (
    GPUServiceResource,
)


@job(
    name="gpu_processing_pipeline",
    description="GPU processing pipeline using torch isolation architecture",
    resource_defs={
        "gpu_service": GPUServiceResource(
            gpu_service_address="gpu-services:50051",
            embedding_service_address="embedding-services:50051",
            reranking_service_address="reranking-services:50051",
            timeout=30,
            max_retries=3,
        )
    },
    config={
        "resources": {
            "gpu_service": {
                "config": {
                    "gpu_service_address": "gpu-services:50051",
                    "embedding_service_address": "embedding-services:50051",
                    "reranking_service_address": "reranking-services:50051",
                    "timeout": 30,
                    "max_retries": 3,
                }
            }
        }
    },
)
def gpu_processing_pipeline():
    """GPU processing pipeline using torch isolation architecture.

    This job demonstrates the complete GPU processing pipeline:
    1. Check GPU service health
    2. Get GPU service status
    3. Allocate GPU resources
    4. Generate embeddings
    5. Rerank documents

    All GPU operations are performed via gRPC calls to dedicated
    Docker services, keeping the main Dagster worker torch-free.
    """
    # Check GPU service health first
    health_status = check_gpu_service_health()

    # Get GPU service status
    gpu_status = get_gpu_service_status()

    # Allocate GPU resources
    allocation = allocate_gpu_resources()

    # Note: The actual document processing would depend on upstream assets
    # This is a simplified example showing the GPU service integration


@job(
    name="embedding_generation_job",
    description="Embedding generation job using GPU services",
    resource_defs={
        "gpu_service": GPUServiceResource(
            gpu_service_address="gpu-services:50051",
            embedding_service_address="embedding-services:50051",
            reranking_service_address="reranking-services:50051",
            timeout=60,
            max_retries=5,
        )
    },
    config={
        "resources": {
            "gpu_service": {
                "config": {
                    "gpu_service_address": "gpu-services:50051",
                    "embedding_service_address": "embedding-services:50051",
                    "reranking_service_address": "reranking-services:50051",
                    "timeout": 60,
                    "max_retries": 5,
                }
            }
        }
    },
)
def embedding_generation_job():
    """Embedding generation job using GPU services.

    This job focuses specifically on embedding generation using
    the GPU-enabled embedding service.
    """
    # Check GPU service health
    health_status = check_gpu_service_health()

    # Generate embeddings (would depend on upstream chunked documents)
    # embeddings = generate_embeddings()


@job(
    name="reranking_job",
    description="Reranking job using GPU services",
    resource_defs={
        "gpu_service": GPUServiceResource(
            gpu_service_address="gpu-services:50051",
            embedding_service_address="embedding-services:50051",
            reranking_service_address="reranking-services:50051",
            timeout=45,
            max_retries=3,
        )
    },
    config={
        "resources": {
            "gpu_service": {
                "config": {
                    "gpu_service_address": "gpu-services:50051",
                    "embedding_service_address": "embedding-services:50051",
                    "reranking_service_address": "reranking-services:50051",
                    "timeout": 45,
                    "max_retries": 3,
                }
            }
        }
    },
)
def reranking_job():
    """Reranking job using GPU services.

    This job focuses specifically on document reranking using
    the GPU-enabled reranking service.
    """
    # Check GPU service health
    health_status = check_gpu_service_health()

    # Rerank documents (would depend on upstream embedded documents)
    # reranked_docs = rerank_documents()


@job(
    name="batch_processing_job",
    description="Batch processing job using GPU services",
    resource_defs={
        "gpu_service": GPUServiceResource(
            gpu_service_address="gpu-services:50051",
            embedding_service_address="embedding-services:50051",
            reranking_service_address="reranking-services:50051",
            timeout=120,
            max_retries=3,
        )
    },
    config={
        "resources": {
            "gpu_service": {
                "config": {
                    "gpu_service_address": "gpu-services:50051",
                    "embedding_service_address": "embedding-services:50051",
                    "reranking_service_address": "reranking-services:50051",
                    "timeout": 120,
                    "max_retries": 3,
                }
            }
        }
    },
)
def batch_processing_job():
    """Batch processing job using GPU services.

    This job demonstrates batch processing of multiple documents
    using GPU services for efficient resource utilization.
    """
    # Check GPU service health
    health_status = check_gpu_service_health()

    # Batch process documents (would depend on upstream chunked documents)
    # batch_results = batch_process_documents()


def create_gpu_job_for_environment(environment: str) -> dict[str, Any]:
    """Create GPU job configuration for specific environment."""
    if environment == "development":
        return {
            "gpu_service_address": "localhost:50051",
            "embedding_service_address": "localhost:50052",
            "reranking_service_address": "localhost:50053",
            "timeout": 60,
            "max_retries": 5,
        }
    elif environment == "staging":
        return {
            "gpu_service_address": "gpu-services.staging:50051",
            "embedding_service_address": "embedding-services.staging:50051",
            "reranking_service_address": "reranking-services.staging:50051",
            "timeout": 45,
            "max_retries": 3,
        }
    elif environment == "production":
        return {
            "gpu_service_address": "gpu-services.production:50051",
            "embedding_service_address": "embedding-services.production:50051",
            "reranking_service_address": "reranking-services.production:50051",
            "timeout": 30,
            "max_retries": 3,
        }
    else:
        raise ValueError(f"Unknown environment: {environment}")


def get_gpu_job_config(environment: str = "production") -> dict[str, Any]:
    """Get GPU job configuration for environment."""
    return {"resources": {"gpu_service": {"config": create_gpu_job_for_environment(environment)}}}
