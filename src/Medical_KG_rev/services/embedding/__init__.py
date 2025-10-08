"""Universal embedding service implementation."""

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
