"""Batch size controller for embedding operations.

This module provides batch size management for embedding operations,
replacing the torch-dependent BatchController with a simplified version
that works with the current gRPC-based architecture.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Final

import structlog


logger = structlog.get_logger(__name__)

_DEFAULT_WINDOW: Final[int] = 10


class BatchController:
    """Simplified batch size controller for embedding operations.

    This replaces the torch-dependent BatchController with a lightweight
    version that manages batch sizes for embedding operations without
    requiring GPU memory monitoring.
    """

    def __init__(self, window: int = _DEFAULT_WINDOW) -> None:
        """Initialize the batch controller.

        Args:
        ----
            window: Number of recent operations to track per namespace
        """
        self.window = window
        self.history: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.overrides: dict[str, int] = {}

    def choose(
        self,
        namespace: str,
        *,
        default: int,
        pending: int,
        candidates: list[int],
    ) -> int:
        """Choose optimal batch size for embedding operations.

        Args:
        ----
            namespace: Namespace identifier
            default: Default batch size
            pending: Number of pending operations
            candidates: Available batch size candidates

        Returns:
        -------
            Selected batch size
        """
        # Check for override first
        if namespace in self.overrides:
            return self.overrides[namespace]

        # Cap by pending operations
        max_size = min(pending, max(candidates))

        # Use default if no history
        if not self.history[namespace]:
            return min(default, max_size)

        # Choose based on recent performance
        recent_sizes = [size for size, _ in self.history[namespace][-3:]]
        if recent_sizes:
            # Prefer sizes that have been successful recently
            for candidate in sorted(candidates, reverse=True):
                if candidate <= max_size and candidate in recent_sizes:
                    return candidate

        return min(default, max_size)

    def reduce(self, namespace: str, size: int) -> None:
        """Set a reduced batch size override for a namespace.

        Args:
        ----
            namespace: Namespace identifier
            size: Override batch size
        """
        self.overrides[namespace] = size
        logger.debug("batch_controller.override", namespace=namespace, size=size)

    def record_success(self, namespace: str, size: int, duration: float) -> None:
        """Record a successful batch operation.

        Args:
        ----
            namespace: Namespace identifier
            size: Batch size used
            duration: Operation duration in seconds
        """
        self.history[namespace].append((size, duration))

        # Keep only recent history
        if len(self.history[namespace]) > self.window:
            self.history[namespace] = self.history[namespace][-self.window :]

        logger.debug(
            "batch_controller.success",
            namespace=namespace,
            size=size,
            duration=duration,
        )

    def record_failure(self, namespace: str, size: int, error: str) -> None:
        """Record a failed batch operation.

        Args:
        ----
            namespace: Namespace identifier
            size: Batch size that failed
            error: Error description
        """
        logger.warning(
            "batch_controller.failure",
            namespace=namespace,
            size=size,
            error=error,
        )

        # Reduce batch size on failure
        if namespace in self.overrides:
            self.overrides[namespace] = max(1, self.overrides[namespace] // 2)
        else:
            self.overrides[namespace] = max(1, size // 2)

    def clear_override(self, namespace: str) -> None:
        """Clear batch size override for a namespace.

        Args:
        ----
            namespace: Namespace identifier
        """
        if namespace in self.overrides:
            del self.overrides[namespace]
            logger.debug("batch_controller.clear_override", namespace=namespace)

    def get_stats(self, namespace: str) -> dict[str, float]:
        """Get statistics for a namespace.

        Args:
        ----
            namespace: Namespace identifier

        Returns:
        -------
            Statistics dictionary
        """
        if not self.history[namespace]:
            return {}

        sizes = [size for size, _ in self.history[namespace]]
        durations = [duration for _, duration in self.history[namespace]]

        return {
            "avg_size": sum(sizes) / len(sizes),
            "avg_duration": sum(durations) / len(durations),
            "total_operations": len(sizes),
            "current_override": self.overrides.get(namespace),
        }
