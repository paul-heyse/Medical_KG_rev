"""Custom exceptions for the vector store subsystem.

This module defines custom exception classes for the vector store subsystem,
providing structured error handling with RFC 7807 problem details for
API responses.

The module defines:
- VectorStoreError: Base exception with RFC 7807 problem details
- NamespaceNotFoundError: Error for missing vector namespaces
- DimensionMismatchError: Error for vector dimension mismatches
- ResourceExhaustedError: Error for vector store capacity limits
- BackendUnavailableError: Error for vector backend unavailability
- ScopeError: Error for missing required scopes
- InvalidNamespaceConfigError: Error for invalid namespace configuration

Architecture:
- Exception hierarchy with base VectorStoreError class
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
        vector_store.store(namespace, vectors)
    except NamespaceNotFoundError as e:
        return e.problem.to_response()
"""

# IMPORTS
from __future__ import annotations

from typing import Any

from Medical_KG_rev.utils.errors import FoundationError


# EXCEPTION CLASSES
class VectorStoreError(FoundationError):
    """Base class for vector store errors with RFC 7807 payloads.

    This exception class provides the base for all vector store-related
    errors, automatically generating RFC 7807 problem details for
    consistent API error responses.

    Attributes:
        problem: RFC 7807 problem details object

    Thread Safety:
        Thread-safe exception class.

    Examples:
        error = VectorStoreError(
            "Vector operation failed",
            status=500,
            detail="Internal vector store error"
        )
        response = error.problem.to_response()
    """

    def __init__(self, message: str, *, status: int, detail: str | None = None, extra: dict[str, Any] | None = None) -> None:
        """Initialize the vector store error.

        Args:
            message: Error message
            status: HTTP status code
            detail: Optional detailed error message
            extra: Additional error context

        Raises:
            None: Initialization always succeeds.
        """
        super().__init__(message, status=status, detail=detail)
        if extra:
            self.problem.extra = extra


class NamespaceNotFoundError(VectorStoreError):
    """Error raised when a vector namespace is not found.

    This exception is raised when attempting to access a vector
    namespace that is not registered for the specified tenant.

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            vector_store.get_namespace("unknown", tenant_id="tenant-1")
        except NamespaceNotFoundError as e:
            return e.problem.to_response()
    """

    def __init__(self, namespace: str, *, tenant_id: str) -> None:
        """Initialize the namespace not found error.

        Args:
            namespace: The unknown namespace name
            tenant_id: The tenant ID

        Raises:
            None: Initialization always succeeds.
        """
        super().__init__(
            "Vector namespace not found",
            status=404,
            detail=f"Namespace '{namespace}' is not registered for tenant '{tenant_id}'.",
            extra={"namespace": namespace, "tenant_id": tenant_id},
        )


class DimensionMismatchError(VectorStoreError):
    """Error raised when vector dimensions do not match.

    This exception is raised when attempting to store vectors
    with dimensions that do not match the namespace configuration.

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            vector_store.store(namespace, vectors)
        except DimensionMismatchError as e:
            return e.problem.to_response()
    """

    def __init__(self, expected: int, actual: int, *, namespace: str) -> None:
        """Initialize the dimension mismatch error.

        Args:
            expected: Expected vector dimension
            actual: Actual vector dimension
            namespace: The namespace name

        Raises:
            None: Initialization always succeeds.
        """
        super().__init__(
            "Vector dimension mismatch",
            status=422,
            detail=f"Expected dimension {expected}, received {actual}.",
            extra={"expected": expected, "actual": actual, "namespace": namespace},
        )


class ResourceExhaustedError(VectorStoreError):
    """Error raised when vector store capacity is exceeded.

    This exception is raised when the vector store backend
    cannot accept more data due to capacity limits.

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            vector_store.store(namespace, large_vectors)
        except ResourceExhaustedError as e:
            return e.problem.to_response()
    """

    def __init__(self, namespace: str, *, detail: str | None = None) -> None:
        """Initialize the resource exhausted error.

        Args:
            namespace: The namespace name
            detail: Optional detailed error message

        Raises:
            None: Initialization always succeeds.
        """
        super().__init__(
            "Vector store capacity exceeded",
            status=507,
            detail=detail or "The vector backend cannot accept more data.",
            extra={"namespace": namespace},
        )


class BackendUnavailableError(VectorStoreError):
    """Error raised when vector backend is unavailable.

    This exception is raised when the vector store backend
    is temporarily unavailable or unreachable.

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            vector_store.query(namespace, query_vector)
        except BackendUnavailableError as e:
            return e.problem.to_response()
    """

    def __init__(self, message: str = "Vector backend unavailable", *, retry_after: float | None = None) -> None:
        """Initialize the backend unavailable error.

        Args:
            message: Error message
            retry_after: Optional retry delay in seconds

        Raises:
            None: Initialization always succeeds.
        """
        super().__init__(
            message,
            status=503,
            detail="Vector store backend is temporarily unavailable.",
            extra={"retry_after": retry_after} if retry_after else None,
        )


class ScopeError(VectorStoreError):
    """Error raised when required scope is missing.

    This exception is raised when the caller lacks the
    required scope for vector store operations.

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            vector_store.store(namespace, vectors)
        except ScopeError as e:
            return e.problem.to_response()
    """

    def __init__(self, *, required_scope: str) -> None:
        """Initialize the scope error.

        Args:
            required_scope: The required scope name

        Raises:
            None: Initialization always succeeds.
        """
        super().__init__(
            "Missing required scope",
            status=403,
            detail=f"The caller lacks the '{required_scope}' scope.",
            extra={"required_scope": required_scope},
        )


class InvalidNamespaceConfigError(VectorStoreError):
    """Error raised when namespace configuration is invalid.

    This exception is raised when the namespace configuration
    does not meet the required specifications.

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            vector_store.create_namespace(config)
        except InvalidNamespaceConfigError as e:
            return e.problem.to_response()
    """

    def __init__(self, namespace: str, *, detail: str) -> None:
        """Initialize the invalid namespace config error.

        Args:
            namespace: The namespace name
            detail: Detailed error message

        Raises:
            None: Initialization always succeeds.
        """
        super().__init__(
            "Invalid namespace configuration",
            status=422,
            detail=detail,
            extra={"namespace": namespace},
        )
