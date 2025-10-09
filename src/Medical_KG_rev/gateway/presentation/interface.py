"""Presentation layer interfaces for HTTP payload shaping.

This module defines protocols for the presentation layer, providing
abstractions for request parsing and response formatting across different
transport protocols (REST, GraphQL, gRPC). These interfaces ensure
consistent payload handling while allowing protocol-specific implementations.

Key Responsibilities:
    - Define response presentation protocols
    - Define request parsing protocols
    - Provide type-safe interfaces for payload handling
    - Enable protocol-specific implementations

Collaborators:
    - Upstream: Protocol handlers (REST router, GraphQL resolvers)
    - Downstream: Concrete presentation implementations

Side Effects:
    - None: Pure interface definitions

Thread Safety:
    - Thread-safe: Protocols are stateless interfaces

Performance Characteristics:
    - O(1) interface operations
    - No runtime overhead for protocol compliance

Example:
    >>> from Medical_KG_rev.gateway.presentation.interface import ResponsePresenter
    >>> class JSONPresenter:
    ...     def success(self, data, *, status_code=200, meta=None):
    ...         return JSONResponse(data, status_code=status_code)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

from typing import Any, Mapping, Protocol

from fastapi import Response

# ==============================================================================
# PRESENTATION PROTOCOLS
# ==============================================================================


class ResponsePresenter(Protocol):
    """Protocol describing presentation responsibilities for route handlers."""

    def success(
        self,
        data: Any,
        *,
        status_code: int = 200,
        meta: Mapping[str, Any] | None = None,
    ) -> Response:
        """Render a successful response with the given payload."""

    def error(
        self,
        detail: Any,
        *,
        status_code: int = 400,
    ) -> Response:
        """Render an error payload in the transport format."""


class RequestParser(Protocol):
    """Protocol for request parsers extracting structured information."""

    def parse(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        """Convert the raw mapping into a structured payload."""


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "ResponsePresenter",
    "RequestParser",
]
