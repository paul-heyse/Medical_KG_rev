"""Problem detail helpers for consistent error reporting across services.

Key Responsibilities:
    - Provide RFC 7807 compliant data structures and helpers used by gateway and
      service layers when returning errors
    - Supply a base exception that carries problem details for translation to
      API responses

Collaborators:
    - Upstream: Service implementations raise ``FoundationError`` when they need
      structured error payloads
    - Downstream: Gateway response mappers serialise :class:`ProblemDetail`
      instances into HTTP responses

Side Effects:
    - None; helpers are pure data containers

Thread Safety:
    - Thread-safe; dataclasses are immutable aside from standard attribute
      mutation semantics

Performance Characteristics:
    - O(1) operations that only touch small dictionaries
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

__all__ = ["ProblemDetail", "FoundationError"]


@dataclass(slots=True)
class ProblemDetail:
    """Lightweight problem details object compliant with RFC 7807."""

    title: str
    status: int
    detail: str | None = None
    type: str = "about:blank"
    instance: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        """Return a dictionary representation with optional fields dropped."""
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        if not payload.get("extra"):
            payload.pop("extra", None)
        return payload

    def to_response(self) -> dict[str, Any]:
        """Alias for model_dump used by existing call-sites."""
        return self.model_dump()


class FoundationError(RuntimeError):
    """Base exception that carries a :class:`ProblemDetail` instance."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 500,
        detail: str | None = None,
        type: str = "about:blank",
        instance: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialise the exception with structured problem detail attributes.

        Args:
            message: Human readable error summary.
            status: HTTP status code associated with the problem.
            detail: Optional detailed description of the failure.
            type: Problem type URI, defaults to ``about:blank``.
            instance: Optional URI reference identifying the specific occurrence.
            extra: Additional attributes included in the serialized payload.
        """
        super().__init__(message)
        self.problem = ProblemDetail(
            title=message,
            status=status,
            detail=detail,
            type=type,
            instance=instance,
            extra=extra or {},
        )
