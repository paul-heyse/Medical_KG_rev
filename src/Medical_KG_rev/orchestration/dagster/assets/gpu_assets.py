"""Dagster assets for GPU service operations."""

from __future__ import annotations

from typing import Any

from dagster import asset, get_dagster_logger
from pydantic import BaseModel

from Medical_KG_rev.observability.service_metrics import collect_service_metrics
from Medical_KG_rev.orchestration.dagster.resources.gpu_services import (
    GPUServiceResource,
)


class ChunkedDocument(BaseModel):
    """Chunked document for embedding generation."""

    document_id: str
    chunks: list[dict[str, Any]]


class EmbeddedDocument(BaseModel):
    """Document with embeddings."""

    document_id: str
    embeddings: list[list[float]]
    metadata: dict[str, Any]


class RerankedDocument(BaseModel):
    """Document with reranking scores."""

    document_id: str
    query: str
    scores: list[float]
    metadata: dict[str, Any]


@asset(
    compute_kind="gpu",
    required_resource_keys={"gpu_service"},
    description="Generate embeddings using GPU service via gRPC",
)
async def generate_embeddings(
    context,
    chunked_documents: list[ChunkedDocument],
) -> list[EmbeddedDocument]:
    """Generate embeddings using GPU service via gRPC.

    This asset calls the embedding service which runs in a separate
    torch-enabled Docker container. The main Dagster worker remains
    torch-free.
    """
    logger = get_dagster_logger()

    # Get gRPC client from Dagster resources
    gpu_service: GPUServiceResource = context.resources.gpu_service
    embedding_client = gpu_service.get_embedding_client()

    embedded_docs = []

    for doc in chunked_documents:
        try:
            # Collect metrics for this operation
            with collect_service_metrics("embedding_service", "generate_embeddings"):
                # Call embedding service via gRPC
                embeddings = await embedding_client.generate_embeddings(
                    texts=[chunk["text"] for chunk in doc.chunks],
                    model="Qwen/Qwen3-Embedding-8B",
                )

            embedded_doc = EmbeddedDocument(
                document_id=doc.document_id,
                embeddings=embeddings,
                metadata={
                    "chunk_count": len(doc.chunks),
                    "embedding_model": "Qwen/Qwen3-Embedding-8B",
                    "generated_at": context.run_id,
                },
            )
            embedded_docs.append(embedded_doc)

            logger.info(f"Generated embeddings for document {doc.document_id}")

        except Exception as e:
            # GPU unavailable - fail fast, don't continue
            logger.error(f"GPU unavailable for document {doc.document_id}: {e}")
            raise  # Fail the entire Dagster run

    logger.info(f"Generated embeddings for {len(embedded_docs)} documents")
    return embedded_docs


@asset(
    compute_kind="gpu",
    required_resource_keys={"gpu_service"},
    description="Rerank documents using GPU service via gRPC",
)
async def rerank_documents(
    context,
    embedded_documents: list[EmbeddedDocument],
    query: str,
) -> list[RerankedDocument]:
    """Rerank documents using GPU service via gRPC.

    This asset calls the reranking service which runs in a separate
    torch-enabled Docker container. The main Dagster worker remains
    torch-free.
    """
    logger = get_dagster_logger()

    # Get gRPC client from Dagster resources
    gpu_service: GPUServiceResource = context.resources.gpu_service
    reranking_client = gpu_service.get_reranking_client()

    reranked_docs = []

    for doc in embedded_documents:
        try:
            # Collect metrics for this operation
            with collect_service_metrics("reranking_service", "rerank_documents"):
                # Call reranking service via gRPC
                scores = await reranking_client.rerank(
                    query=query,
                    documents=[f"Document {doc.document_id}"],  # Simplified for example
                    model="default",
                )

            reranked_doc = RerankedDocument(
                document_id=doc.document_id,
                query=query,
                scores=scores,
                metadata={
                    "embedding_count": len(doc.embeddings),
                    "reranking_model": "default",
                    "generated_at": context.run_id,
                },
            )
            reranked_docs.append(reranked_doc)

            logger.info(f"Reranked document {doc.document_id}")

        except Exception as e:
            # GPU unavailable - fail fast, don't continue
            logger.error(f"GPU unavailable for document {doc.document_id}: {e}")
            raise  # Fail the entire Dagster run

    logger.info(f"Reranked {len(reranked_docs)} documents")
    return reranked_docs


@asset(
    compute_kind="gpu",
    required_resource_keys={"gpu_service"},
    description="Check GPU service health and availability",
)
async def check_gpu_service_health(
    context,
) -> dict[str, Any]:
    """Check GPU service health and availability.

    This asset verifies that GPU services are available and healthy
    before proceeding with GPU-intensive operations.
    """
    logger = get_dagster_logger()

    # Get gRPC client from Dagster resources
    gpu_service: GPUServiceResource = context.resources.gpu_service

    try:
        # Check health of all GPU services
        health_status = gpu_service.health_check()

        # Check if all services are healthy
        all_healthy = all(service["status"] == "healthy" for service in health_status.values())

        if not all_healthy:
            unhealthy_services = [
                name for name, status in health_status.items() if status["status"] != "healthy"
            ]
            error_msg = f"Unhealthy GPU services: {unhealthy_services}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info("All GPU services are healthy")

        return {"status": "healthy", "services": health_status, "checked_at": context.run_id}

    except Exception as e:
        logger.error(f"GPU service health check failed: {e}")
        raise


@asset(
    compute_kind="gpu",
    required_resource_keys={"gpu_service"},
    description="Get GPU service status and metrics",
)
async def get_gpu_service_status(
    context,
) -> dict[str, Any]:
    """Get GPU service status and metrics.

    This asset retrieves detailed status information from GPU services
    including utilization, memory usage, and device information.
    """
    logger = get_dagster_logger()

    # Get gRPC client from Dagster resources
    gpu_service: GPUServiceResource = context.resources.gpu_service
    gpu_client = gpu_service.get_gpu_client()

    try:
        # Collect metrics for this operation
        with collect_service_metrics("gpu_service", "get_status"):
            # Get GPU status
            gpu_status = await gpu_client.get_gpu_status()

            # Get GPU devices
            devices = await gpu_client.list_devices()

        status_info = {
            "gpu_status": {
                "available": gpu_status.available,
                "device_count": len(devices),
                "devices": [
                    {
                        "id": device.id,
                        "name": device.name,
                        "memory_total": device.memory_total,
                        "memory_used": device.memory_used,
                        "utilization": device.utilization,
                    }
                    for device in devices
                ],
            },
            "checked_at": context.run_id,
        }

        logger.info(f"GPU service status: {status_info}")
        return status_info

    except Exception as e:
        logger.error(f"Failed to get GPU service status: {e}")
        raise


@asset(
    compute_kind="gpu",
    required_resource_keys={"gpu_service"},
    description="Allocate GPU resources for processing",
)
async def allocate_gpu_resources(
    context,
    memory_mb: int = 1024,
) -> dict[str, Any]:
    """Allocate GPU resources for processing.

    This asset allocates GPU memory for subsequent processing operations.
    """
    logger = get_dagster_logger()

    # Get gRPC client from Dagster resources
    gpu_service: GPUServiceResource = context.resources.gpu_service
    gpu_client = gpu_service.get_gpu_client()

    try:
        # Collect metrics for this operation
        with collect_service_metrics("gpu_service", "allocate_gpu"):
            # Allocate GPU memory
            allocation = await gpu_client.allocate_gpu(memory_mb=memory_mb)

        if not allocation.success:
            error_msg = f"Failed to allocate {memory_mb}MB GPU memory"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        allocation_info = {
            "allocation_id": allocation.allocation_id,
            "memory_mb": memory_mb,
            "device_id": allocation.device_id,
            "allocated_at": context.run_id,
        }

        logger.info(f"Allocated {memory_mb}MB GPU memory: {allocation_info}")
        return allocation_info

    except Exception as e:
        logger.error(f"Failed to allocate GPU resources: {e}")
        raise


@asset(
    compute_kind="gpu",
    required_resource_keys={"gpu_service"},
    description="Batch process multiple documents with GPU services",
)
async def batch_process_documents(
    context,
    chunked_documents: list[ChunkedDocument],
    query: str,
) -> list[RerankedDocument]:
    """Batch process multiple documents with GPU services.

    This asset demonstrates batch processing using multiple GPU services
    in sequence: embedding generation followed by reranking.
    """
    logger = get_dagster_logger()

    # Get gRPC clients from Dagster resources
    gpu_service: GPUServiceResource = context.resources.gpu_service
    embedding_client = gpu_service.get_embedding_client()
    reranking_client = gpu_service.get_reranking_client()

    try:
        # Step 1: Generate embeddings for all documents
        logger.info(f"Generating embeddings for {len(chunked_documents)} documents")

        all_embeddings = []
        for doc in chunked_documents:
            with collect_service_metrics("embedding_service", "batch_generate_embeddings"):
                embeddings = await embedding_client.generate_embeddings(
                    texts=[chunk["text"] for chunk in doc.chunks],
                    model="Qwen/Qwen3-Embedding-8B",
                )
            all_embeddings.extend(embeddings)

        # Step 2: Rerank all documents
        logger.info(f"Reranking {len(chunked_documents)} documents")

        with collect_service_metrics("reranking_service", "batch_rerank"):
            scores = await reranking_client.rerank(
                query=query,
                documents=[f"Document {doc.document_id}" for doc in chunked_documents],
                model="default",
            )

        # Step 3: Create reranked documents
        reranked_docs = []
        for i, doc in enumerate(chunked_documents):
            reranked_doc = RerankedDocument(
                document_id=doc.document_id,
                query=query,
                scores=[scores[i]] if i < len(scores) else [0.0],
                metadata={
                    "chunk_count": len(doc.chunks),
                    "embedding_count": len(doc.chunks),
                    "batch_processed": True,
                    "processed_at": context.run_id,
                },
            )
            reranked_docs.append(reranked_doc)

        logger.info(f"Batch processed {len(reranked_docs)} documents")
        return reranked_docs

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise
