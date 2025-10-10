"""Domain specific exceptions for the reranking system.

This module defines domain-specific exception classes for the reranking
system, providing structured error handling with RFC 7807 problem details
for API responses.

The module defines:
- RerankingError: Base exception with RFC 7807 problem details
- InvalidPairFormatError: Error for invalid query/document pairs
- UnknownRerankerError: Error for unknown reranker IDs
- GPUUnavailableError: Error for GPU resource unavailability
- CircuitBreakerOpenError: Error for circuit breaker states

Architecture:
- Exception hierarchy with base RerankingError class
- Automatic RFC 7807 problem detail generation
- Specific error types for different failure scenarios
- Extensible design for additional error types

Thread Safety:
- Exception classes are thread-safe.

Performance:
- Lightweight exception definitions with minimal overhead.
- Problem detail generation is fast and stateless.

Examples:
    try:
        reranker.rank(pairs)
    except InvalidPairFormatError as e:
        return e.to_problem().to_response()

"""

# IMPORTS
from __future__ import annotations

from typing import Any

from Medical_KG_rev.utils.errors import ProblemDetail


# EXCEPTION CLASSES
class RerankingError(RuntimeError):
    """Base exception translating to an RFC 7807 problem detail.

    This exception class provides the base for all reranking-related
    errors, automatically generating RFC 7807 problem details for
    consistent API error responses.

    Attributes:
        title: Error title
        status: HTTP status code
        detail: Optional error detail
        type: Error type URI
        extra: Additional error context

    Thread Safety:
        Thread-safe exception class.

    Examples:
        error = RerankingError(
            title="Reranking failed",
            status=500,
            detail="Internal reranking error"
        )
        problem = error.to_problem()

    """

    def __init__(
        self,
        title: str,
        *,
        status: int,
        detail: str | None = None,
        type: str = "https://docs.medical-kg/reranking/errors",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the reranking error.

        Args:
            title: Error title
            status: HTTP status code
            detail: Optional error detail
            type: Error type URI
            extra: Additional error context

        Raises:
            None: Initialization always succeeds.

        """
        super().__init__(title)
        self.title = title
        self.status = status
        self.detail = detail
        self.type = type
        self.extra = extra or {}

    def to_problem(self) -> ProblemDetail:
        """Convert the error to an RFC 7807 problem detail.

        Returns:
            ProblemDetail object suitable for API responses

        Raises:
            None: This method never raises exceptions.

        """
        return ProblemDetail(
            title=self.title,
            status=self.status,
            detail=self.detail,
            type=self.type,
            extra=dict(self.extra),
        )


class InvalidPairFormatError(RerankingError):
    """Error raised when query/document pairs have invalid format.

    This exception is raised when the input pairs for reranking
    do not conform to the expected format or structure.

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            validate_pairs(pairs)
        except InvalidPairFormatError as e:
            return e.to_problem()

    """

    def __init__(self, detail: str) -> None:
        """Initialize the invalid pair format error.

        Args:
            detail: Detailed error message

        Raises:
            None: Initialization always succeeds.

        """
        super().__init__(title="Invalid query/document pair", status=400, detail=detail)


class UnknownRerankerError(RerankingError):
    """Error raised when a reranker ID is not found.

    This exception is raised when attempting to use a reranker
    that is not registered or available in the system.

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            get_reranker("unknown-id")
        except UnknownRerankerError as e:
            return e.to_problem()

    """

    def __init__(self, reranker_id: str, available: list[str]) -> None:
        """Initialize the unknown reranker error.

        Args:
            reranker_id: The unknown reranker ID
            available: List of available reranker IDs

        Raises:
            None: Initialization always succeeds.

        """
        super().__init__(
            title="Reranker not found",
            status=422,
            detail=f"Reranker '{reranker_id}' is not registered",
            extra={"available": available},
        )


class GPUUnavailableError(RerankingError):
    """Error raised when GPU resources are unavailable.

    This exception is raised when a reranker requires GPU
    acceleration but GPU resources are not available.

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            gpu_reranker.rank(pairs)
        except GPUUnavailableError as e:
            return e.to_problem()

    """

    def __init__(self, reranker_id: str) -> None:
        """Initialize the GPU unavailable error.

        Args:
            reranker_id: The reranker ID requiring GPU

        Raises:
            None: Initialization always succeeds.

        """
        super().__init__(
            title="GPU unavailable",
            status=503,
            detail=f"Reranker '{reranker_id}' requires GPU acceleration",
            type="https://docs.medical-kg/reranking/errors/gpu-unavailable",
        )


class CircuitBreakerOpenError(RerankingError):
    """Error raised when circuit breaker is open.

    This exception is raised when a reranker's circuit breaker
    is in the open state, indicating temporary unavailability.

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            reranker.rank(pairs)
        except CircuitBreakerOpenError as e:
            return e.to_problem()

    """

    def __init__(self, reranker_id: str) -> None:
        """Initialize the circuit breaker open error.

        Args:
            reranker_id: The reranker ID with open circuit breaker

        Raises:
            None: Initialization always succeeds.

        """
        super().__init__(
            title="Reranker temporarily unavailable",
            status=503,
            detail=f"Circuit breaker open for reranker '{reranker_id}'",
            type="https://docs.medical-kg/reranking/errors/circuit-open",
        )
