"""Error payload helpers for presentation formatting.

This module provides utilities for formatting error information into
JSON:API compliant responses. It defines structured error representations
that can be serialized to JSON for API responses.

The module defines:
- ErrorDetail: Structured error metadata for JSON:API responses

Architecture:
- Error details follow JSON:API error object specification
- Supports optional detail messages and metadata
- Provides clean serialization to dictionary format

Thread Safety:
- ErrorDetail instances are immutable and thread-safe
- Serialization methods are stateless

Performance:
- Lightweight serialization with minimal overhead
- No external dependencies

Examples
--------
    error = ErrorDetail(
        status=400,
        code="INVALID_REQUEST",
        title="Invalid request format",
        detail="The request body is malformed",
        meta={"field": "body"}
    )
    json_payload = error.as_json()

"""

from __future__ import annotations

# IMPORTS
from dataclasses import dataclass, field
from typing import Any


# DATA MODELS
@dataclass(slots=True)
class ErrorDetail:
    """Structured error metadata for JSON:API responses.

    This dataclass represents error information in a format compatible
    with the JSON:API specification. It provides structured error
    details suitable for API responses.

    Attributes
    ----------
        status: HTTP status code
        code: Error code identifier
        title: Human-readable error title
        detail: Optional detailed error message
        meta: Optional additional metadata

    Thread Safety:
        Immutable dataclass, thread-safe.

    Examples
    --------
        error = ErrorDetail(
            status=400,
            code="INVALID_REQUEST",
            title="Invalid request format",
            detail="The request body is malformed"
        )

    """

    status: int
    code: str
    title: str
    detail: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        """Serialize the error detail to JSON-compatible dictionary.

        Returns
        -------
            Dictionary representation suitable for JSON serialization

        Raises
        ------
            None: This method never raises exceptions.

        """
        payload = {
            "status": str(self.status),
            "code": self.code,
            "title": self.title,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload


# EXPORTS
__all__ = ["ErrorDetail"]
