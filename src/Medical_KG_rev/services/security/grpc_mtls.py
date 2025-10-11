"""gRPC mTLS helper placeholders."""

from __future__ import annotations

from typing import Any


def create_mtls_channel(endpoint: str, manager: Any | None = None) -> Any:
    """Return insecure channel placeholder."""
    return endpoint


__all__ = ["create_mtls_channel"]
