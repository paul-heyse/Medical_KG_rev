"""Presentation layer interfaces for HTTP payload shaping."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from fastapi import Response


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
