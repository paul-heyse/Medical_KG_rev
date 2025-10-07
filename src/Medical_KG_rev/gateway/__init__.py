"""Multi-protocol API gateway package with lazy exports to avoid circular imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .services import GatewayService


def create_app():
    from .app import create_app as _create_app

    return _create_app()


def get_gateway_service():
    from .services import get_gateway_service as _get_gateway_service

    return _get_gateway_service()


__all__ = ["GatewayService", "create_app", "get_gateway_service"]
