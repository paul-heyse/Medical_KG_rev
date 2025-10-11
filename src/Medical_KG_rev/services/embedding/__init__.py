"""Universal embedding service for vectorizing text and documents.

Key Responsibilities:
    - Provide unified embedding service interface across different backends
    - Support multiple embedding models and namespaces
    - Handle batch processing and caching of embeddings
    - Manage embedding service lifecycle and configuration

Collaborators:
    - Upstream: Gateway services and orchestration layers request embeddings
    - Downstream: Various embedding backends (gRPC services, local models)

Side Effects:
    - May trigger network calls to embedding services
    - Updates embedding caches and metrics
    - Emits telemetry for embedding operations

Thread Safety:
    - Thread-safe: Embedding operations can be called concurrently

Performance Characteristics:
    - Batch processing reduces per-request overhead
    - Caching minimizes redundant embedding computations
    - Namespace isolation prevents model conflicts

Example:
    >>> from Medical_KG_rev.services.embedding import EmbeddingWorker
    >>> worker = EmbeddingWorker()
    >>> result = worker.run(request)
    >>> print(f"Generated {len(result.vectors)} embeddings")
"""

from importlib import import_module
from typing import Any

from .registry import EmbeddingModelRegistry

__all__ = [
    "EmbeddingGrpcService",
    "EmbeddingModelRegistry",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingVector",
    "EmbeddingWorker",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - lazy import hook
    if name in {
        "EmbeddingGrpcService",
        "EmbeddingRequest",
        "EmbeddingResponse",
        "EmbeddingVector",
        "EmbeddingWorker",
    }:
        module = import_module(".service", __name__)
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
