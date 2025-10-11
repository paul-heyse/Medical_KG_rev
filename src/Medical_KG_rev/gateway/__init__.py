"""Multi-protocol API gateway package with lazy exports to avoid circular imports."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .app import create_app as _create_app
from .services import GatewayService
from .services import get_gateway_service as _get_gateway_service

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from fastapi import FastAPI


def create_app(*args: Any, **kwargs: Any):
    """Create and configure the FastAPI application."""
    return _create_app(*args, **kwargs)


def get_gateway_service() -> GatewayService:
    """Return the singleton ``GatewayService`` instance."""
    return _get_gateway_service()


__all__ = ["GatewayService", "create_app", "get_gateway_service"]
