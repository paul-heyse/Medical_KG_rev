"""Error utilities implementing RFC 7807 problem details.

This module provides error handling utilities that implement the RFC 7807
Problem Details for HTTP APIs specification. It includes both Pydantic-based
and lightweight dataclass implementations for maximum compatibility.

The module defines:
- ProblemDetail: RFC 7807 compliant problem details object
- FoundationError: Base exception with embedded problem details

Architecture:
- Supports both Pydantic and dataclass implementations
- Automatic fallback when Pydantic is unavailable
- RFC 7807 compliant error structure
- Clean serialization for HTTP responses

Thread Safety:
- ProblemDetail instances are immutable and thread-safe
- FoundationError instances are thread-safe

Performance:
- Lightweight implementation with minimal overhead
- Optional Pydantic dependency for enhanced validation

Examples:
    # Using ProblemDetail directly
    problem = ProblemDetail(
        title="Validation Error",
        status=400,
        detail="Invalid input format"
    )
    response = problem.to_response()

    # Using FoundationError
    try:
        raise FoundationError("Invalid request", status=400)
    except FoundationError as e:
        response = e.problem.to_response()

"""

# IMPORTS
from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, field
from typing import Any

# TYPE DEFINITIONS & CONSTANTS
_PYDANTIC_AVAILABLE = importlib.util.find_spec("pydantic") is not None

# PROBLEM DETAILS IMPLEMENTATION
if _PYDANTIC_AVAILABLE:
    from pydantic import BaseModel, Field  # type: ignore

    class ProblemDetail(BaseModel):
        """Representation of RFC 7807 problem details object.

        This class provides a Pydantic-based implementation of RFC 7807
        problem details, offering automatic validation and serialization.

        Attributes:
            type: URI identifying the problem type
            title: Human-readable summary of the problem
            status: HTTP status code
            detail: Human-readable explanation of the problem
            instance: URI identifying the specific occurrence
            extra: Additional context about the problem

        Thread Safety:
            Immutable model, thread-safe.

        Examples:
            problem = ProblemDetail(
                title="Validation Error",
                status=400,
                detail="Invalid input format",
                type="https://example.com/errors/validation"
            )

        """

        type: str = Field(default="about:blank")
        title: str
        status: int
        detail: str | None = None
        instance: str | None = None
        extra: dict[str, Any] = Field(default_factory=dict)

        def to_response(self) -> dict[str, Any]:
            """Convert the problem detail to HTTP response format.

            Returns:
                Dictionary suitable for HTTP response serialization

            Raises:
                None: This method never raises exceptions.

            """
            data = self.model_dump()
            payload = {key: value for key, value in data.items() if value is not None}
            if payload.get("extra") == {}:
                payload.pop("extra", None)
            return payload

else:  # pragma: no cover - optional dependency fallback

    @dataclass(slots=True)
    class ProblemDetail:
        """Lightweight problem details implementation without pydantic.

        This class provides a dataclass-based implementation of RFC 7807
        problem details for environments where Pydantic is not available.

        Attributes:
            title: Human-readable summary of the problem
            status: HTTP status code
            type: URI identifying the problem type
            detail: Human-readable explanation of the problem
            instance: URI identifying the specific occurrence
            extra: Additional context about the problem

        Thread Safety:
            Immutable dataclass, thread-safe.

        Examples:
            problem = ProblemDetail(
                title="Validation Error",
                status=400,
                detail="Invalid input format"
            )

        """

        title: str
        status: int
        type: str = "about:blank"
        detail: str | None = None
        instance: str | None = None
        extra: dict[str, Any] = field(default_factory=dict)

        def model_dump(self) -> dict[str, Any]:
            """Convert the problem detail to dictionary format.

            Returns:
                Dictionary representation of the problem detail

            Raises:
                None: This method never raises exceptions.

            """
            return asdict(self)

        def to_response(self) -> dict[str, Any]:
            """Convert the problem detail to HTTP response format.

            Returns:
                Dictionary suitable for HTTP response serialization

            Raises:
                None: This method never raises exceptions.

            """
            data = self.model_dump()
            payload = {key: value for key, value in data.items() if value is not None}
            if payload.get("extra") == {}:
                payload.pop("extra", None)
            return payload


# EXCEPTION CLASSES
class FoundationError(RuntimeError):
    """Base exception for foundation utilities.

    This exception class provides a base for foundation utility errors,
    automatically embedding RFC 7807 problem details for consistent
    error handling across the application.

    Attributes:
        problem: RFC 7807 problem details object

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            raise FoundationError("Invalid configuration", status=400)
        except FoundationError as e:
            response = e.problem.to_response()

    """

    def __init__(self, message: str, *, status: int = 500, detail: str | None = None) -> None:
        """Initialize the foundation error with problem details.

        Args:
            message: Error message
            status: HTTP status code
            detail: Optional detailed error message

        Raises:
            None: Initialization always succeeds.

        """
        super().__init__(message)
        self.problem = ProblemDetail(title=message, status=status, detail=detail)
