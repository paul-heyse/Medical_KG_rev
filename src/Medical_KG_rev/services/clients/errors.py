"""Custom exception classes for service communication errors.
"""

from typing import Any


class ServiceError(Exception):
    """Base exception for service communication errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Initialize service error.

        Args:
            message: Error message
            details: Additional error details

        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ServiceTimeoutError(ServiceError):
    """Raised when service call times out."""

    pass


class ServiceUnavailableError(ServiceError):
    """Raised when service is unavailable."""

    pass


class ServiceOverloadedError(ServiceError):
    """Raised when service is overloaded."""

    pass


class ServiceAuthenticationError(ServiceError):
    """Raised when authentication fails."""

    pass


class ServiceAuthorizationError(ServiceError):
    """Raised when authorization fails."""

    pass


class ServiceValidationError(ServiceError):
    """Raised when request validation fails."""

    pass


class ServiceInternalError(ServiceError):
    """Raised when service has internal error."""

    pass
