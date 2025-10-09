"""Coordinator infrastructure for the API gateway.

This package extracts the heavy orchestration logic out of
``GatewayService`` into narrow coordinator classes.  Each coordinator
focuses on a single business capability (chunking, embedding, etc.)
while sharing lifecycle and resilience helpers.
"""

from .base import (
    BaseCoordinator,
    CoordinatorConfig,
    CoordinatorError,
    CoordinatorMetrics,
    CoordinatorRequest,
    CoordinatorResult,
)
from .chunking import ChunkingCoordinator, ChunkingRequest, ChunkingResult
from .embedding import EmbeddingCoordinator, EmbeddingRequest, EmbeddingResult
from .job_lifecycle import JobLifecycleManager

__all__ = [
    "BaseCoordinator",
    "ChunkingCoordinator",
    "ChunkingRequest",
    "ChunkingResult",
    "CoordinatorConfig",
    "CoordinatorError",
    "CoordinatorMetrics",
    "CoordinatorRequest",
    "CoordinatorResult",
    "EmbeddingCoordinator",
    "EmbeddingRequest",
    "EmbeddingResult",
    "JobLifecycleManager",
]
