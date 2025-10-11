"""Simplified gRPC server bootstrap."""

from __future__ import annotations

from typing import Any


def start_grpc_server(*args: Any, **kwargs: Any) -> None:
    """Placeholder that logs the invocation."""
    # Real server start-up removed during refactor; this keeps call-sites safe.
    return None


__all__ = ["start_grpc_server"]
