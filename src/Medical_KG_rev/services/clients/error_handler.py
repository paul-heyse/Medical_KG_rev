"""Error handling utilities for gRPC service communication.

Provides centralized error handling, retry logic, and error classification.
"""

from collections.abc import Callable
from enum import Enum
from typing import Any
import asyncio
import logging

import grpc

from .errors import (
    ServiceAuthenticationError,
    ServiceAuthorizationError,
    ServiceError,
    ServiceInternalError,
    ServiceOverloadedError,
    ServiceTimeoutError,
    ServiceUnavailableError,
    ServiceValidationError,
)

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    SERVICE_UNAVAILABLE = "service_unavailable"
    SERVICE_OVERLOADED = "service_overloaded"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategies for different error types."""

    NO_RETRY = "no_retry"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"


class ErrorClassification:
    """Classifies gRPC errors and determines appropriate handling."""

    # Error classification mapping
    ERROR_CLASSIFICATIONS: dict[grpc.StatusCode, dict[str, Any]] = {
        grpc.StatusCode.UNAVAILABLE: {
            "category": ErrorCategory.SERVICE_UNAVAILABLE,
            "severity": ErrorSeverity.HIGH,
            "retry_strategy": RetryStrategy.EXPONENTIAL_BACKOFF,
            "max_retries": 3,
            "exception_type": ServiceUnavailableError,
        },
        grpc.StatusCode.DEADLINE_EXCEEDED: {
            "category": ErrorCategory.TIMEOUT,
            "severity": ErrorSeverity.MEDIUM,
            "retry_strategy": RetryStrategy.LINEAR_BACKOFF,
            "max_retries": 2,
            "exception_type": ServiceTimeoutError,
        },
        grpc.StatusCode.RESOURCE_EXHAUSTED: {
            "category": ErrorCategory.SERVICE_OVERLOADED,
            "severity": ErrorSeverity.HIGH,
            "retry_strategy": RetryStrategy.EXPONENTIAL_BACKOFF,
            "max_retries": 5,
            "exception_type": ServiceOverloadedError,
        },
        grpc.StatusCode.UNAUTHENTICATED: {
            "category": ErrorCategory.AUTHENTICATION,
            "severity": ErrorSeverity.CRITICAL,
            "retry_strategy": RetryStrategy.NO_RETRY,
            "max_retries": 0,
            "exception_type": ServiceAuthenticationError,
        },
        grpc.StatusCode.PERMISSION_DENIED: {
            "category": ErrorCategory.AUTHORIZATION,
            "severity": ErrorSeverity.CRITICAL,
            "retry_strategy": RetryStrategy.NO_RETRY,
            "max_retries": 0,
            "exception_type": ServiceAuthorizationError,
        },
        grpc.StatusCode.INVALID_ARGUMENT: {
            "category": ErrorCategory.VALIDATION,
            "severity": ErrorSeverity.MEDIUM,
            "retry_strategy": RetryStrategy.NO_RETRY,
            "max_retries": 0,
            "exception_type": ServiceValidationError,
        },
        grpc.StatusCode.INTERNAL: {
            "category": ErrorCategory.INTERNAL_ERROR,
            "severity": ErrorSeverity.HIGH,
            "retry_strategy": RetryStrategy.EXPONENTIAL_BACKOFF,
            "max_retries": 2,
            "exception_type": ServiceInternalError,
        },
    }

    @classmethod
    def classify_error(cls, error: Exception) -> dict[str, Any]:
        """Classify an error and return handling information.

        Args:
        ----
            error: The error to classify

        Returns:
        -------
            Dictionary containing error classification information

        """
        if isinstance(error, grpc.RpcError):
            status_code = error.code()
            classification = cls.ERROR_CLASSIFICATIONS.get(
                status_code,
                {
                    "category": ErrorCategory.UNKNOWN,
                    "severity": ErrorSeverity.MEDIUM,
                    "retry_strategy": RetryStrategy.NO_RETRY,
                    "max_retries": 0,
                    "exception_type": ServiceError,
                },
            )

            return {
                "status_code": status_code,
                "category": classification["category"],
                "severity": classification["severity"],
                "retry_strategy": classification["retry_strategy"],
                "max_retries": classification["max_retries"],
                "exception_type": classification["exception_type"],
                "details": error.details(),
                "is_retryable": classification["max_retries"] > 0,
            }
        else:
            return {
                "status_code": None,
                "category": ErrorCategory.UNKNOWN,
                "severity": ErrorSeverity.MEDIUM,
                "retry_strategy": RetryStrategy.NO_RETRY,
                "max_retries": 0,
                "exception_type": ServiceError,
                "details": str(error),
                "is_retryable": False,
            }


class RetryManager:
    """Manages retry logic for service calls."""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0):
        """Initialize retry manager.

        Args:
        ----
            base_delay: Base delay in seconds for retries
            max_delay: Maximum delay in seconds for retries

        """
        self.base_delay = base_delay
        self.max_delay = max_delay

    def calculate_delay(
        self,
        attempt: int,
        strategy: RetryStrategy,
        base_delay: float | None = None,
    ) -> float:
        """Calculate delay for retry attempt.

        Args:
        ----
            attempt: Current attempt number (0-based)
            strategy: Retry strategy to use
            base_delay: Override base delay

        Returns:
        -------
            Delay in seconds

        """
        delay = base_delay or self.base_delay

        if strategy == RetryStrategy.NO_RETRY:
            return 0.0
        elif strategy == RetryStrategy.FIXED_DELAY:
            return delay
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            return min(delay * (attempt + 1), self.max_delay)
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return min(delay * (2**attempt), self.max_delay)
        else:
            return delay

    async def execute_with_retry(
        self,
        func: Callable,
        error_classification: dict[str, Any],
        *args,
        **kwargs,
    ) -> Any:
        """Execute function with retry logic.

        Args:
        ----
            func: Function to execute
            error_classification: Error classification information
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
        -------
            Function result

        Raises:
        ------
            ServiceError: If all retries fail

        """
        max_retries = error_classification["max_retries"]
        retry_strategy = error_classification["retry_strategy"]

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e

                # Classify the error
                error_info = ErrorClassification.classify_error(e)

                # Check if we should retry
                if attempt >= max_retries or not error_info["is_retryable"]:
                    break

                # Calculate delay and wait
                delay = self.calculate_delay(attempt, retry_strategy)
                if delay > 0:
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay. "
                        f"Error: {error_info['details']}"
                    )
                    await asyncio.sleep(delay)

        # All retries failed, raise appropriate exception
        if isinstance(last_error, grpc.RpcError):
            error_info = ErrorClassification.classify_error(last_error)
            exception_type = error_info["exception_type"]
            raise exception_type(
                f"Service call failed after {max_retries} retries: {error_info['details']}"
            ) from last_error
        else:
            raise ServiceError(f"Service call failed: {last_error!s}") from last_error


class CircuitBreakerErrorHandler:
    """Error handler that integrates with circuit breaker patterns."""

    def __init__(self, circuit_breaker: Any) -> None:
        """Initialize circuit breaker error handler.

        Args:
        ----
            circuit_breaker: Circuit breaker instance

        """
        self.circuit_breaker = circuit_breaker
        self.retry_manager = RetryManager()

    async def handle_service_call(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Handle service call with circuit breaker and retry logic.

        Args:
        ----
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
        -------
            Function result

        Raises:
        ------
            ServiceError: If service call fails

        """
        try:
            # Execute with circuit breaker
            result = await self.circuit_breaker.call(func, *args, **kwargs)
            return result
        except Exception as e:
            # Classify error
            error_info = ErrorClassification.classify_error(e)

            # Log error with appropriate level
            log_level = self._get_log_level(error_info["severity"])
            logger.log(
                log_level,
                f"Service call failed: {error_info['category'].value} - {error_info['details']}",
            )

            # Handle based on error type
            if error_info["is_retryable"]:
                try:
                    return await self.retry_manager.execute_with_retry(
                        func, error_info, *args, **kwargs
                    )
                except Exception as retry_error:
                    # Retry failed, raise original error
                    error_info = ErrorClassification.classify_error(retry_error)
                    exception_type = error_info["exception_type"]
                    raise exception_type(
                        f"Service call failed after retries: {error_info['details']}"
                    ) from retry_error
            else:
                # Non-retryable error, raise immediately
                exception_type = error_info["exception_type"]
                raise exception_type(f"Service call failed: {error_info['details']}") from e

    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Get log level for error severity."""
        severity_levels = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        return severity_levels.get(severity, logging.ERROR)


class ServiceErrorMetrics:
    """Collects and reports error metrics for services."""

    def __init__(self) -> None:
        """Initialize error metrics collector."""
        self._metrics = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "errors_by_service": {},
            "retry_attempts": 0,
            "circuit_breaker_trips": 0,
        }

    def record_error(
        self,
        service_name: str,
        error_info: dict[str, Any],
        retry_attempts: int = 0,
    ) -> None:
        """Record an error occurrence.

        Args:
        ----
            service_name: Name of the service
            error_info: Error classification information
            retry_attempts: Number of retry attempts made

        """
        self._metrics["total_errors"] += 1
        self._metrics["retry_attempts"] += retry_attempts

        # Record by category
        category = error_info["category"].value
        self._metrics["errors_by_category"][category] = (
            self._metrics["errors_by_category"].get(category, 0) + 1
        )

        # Record by severity
        severity = error_info["severity"].value
        self._metrics["errors_by_severity"][severity] = (
            self._metrics["errors_by_severity"].get(severity, 0) + 1
        )

        # Record by service
        self._metrics["errors_by_service"][service_name] = (
            self._metrics["errors_by_service"].get(service_name, 0) + 1
        )

    def record_circuit_breaker_trip(self) -> None:
        """Record a circuit breaker trip."""
        self._metrics["circuit_breaker_trips"] += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get current error metrics."""
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Reset error metrics."""
        self._metrics = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "errors_by_service": {},
            "retry_attempts": 0,
            "circuit_breaker_trips": 0,
        }


class ServiceErrorHandler:
    """Main error handler for service communication."""

    def __init__(self, service_name: str) -> None:
        """Initialize service error handler.

        Args:
        ----
            service_name: Name of the service

        """
        self.service_name = service_name
        self.metrics = ServiceErrorMetrics()
        self.retry_manager = RetryManager()

    async def handle_call(
        self,
        func: Callable,
        circuit_breaker: Any = None,
        *args,
        **kwargs,
    ) -> Any:
        """Handle service call with comprehensive error handling.

        Args:
        ----
            func: Function to execute
            circuit_breaker: Optional circuit breaker instance
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
        -------
            Function result

        Raises:
        ------
            ServiceError: If service call fails

        """
        # start_time = time.time()  # Unused variable

        try:
            if circuit_breaker:
                # Use circuit breaker error handler
                error_handler = CircuitBreakerErrorHandler(circuit_breaker)
                result = await error_handler.handle_service_call(func, *args, **kwargs)
            else:
                # Direct execution
                result = await func(*args, **kwargs)

            return result

        except Exception as e:
            # Classify error
            error_info = ErrorClassification.classify_error(e)

            # Record error metrics
            self.metrics.record_error(self.service_name, error_info)

            # Log error
            log_level = self._get_log_level(error_info["severity"])
            logger.log(
                log_level,
                f"Service call failed for {self.service_name}: "
                f"{error_info['category'].value} - {error_info['details']}",
            )

            # Handle based on error type
            if error_info["is_retryable"] and not circuit_breaker:
                try:
                    return await self.retry_manager.execute_with_retry(
                        func, error_info, *args, **kwargs
                    )
                except Exception as retry_error:
                    # Retry failed, raise original error
                    error_info = ErrorClassification.classify_error(retry_error)
                    exception_type = error_info["exception_type"]
                    raise exception_type(
                        f"Service call failed after retries: {error_info['details']}"
                    ) from retry_error
            else:
                # Non-retryable error or circuit breaker is handling retries
                exception_type = error_info["exception_type"]
                raise exception_type(f"Service call failed: {error_info['details']}") from e

    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Get log level for error severity."""
        severity_levels = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        return severity_levels.get(severity, logging.ERROR)

    def get_error_summary(self) -> dict[str, Any]:
        """Get error summary for the service."""
        metrics = self.metrics.get_metrics()
        return {
            "service_name": self.service_name,
            "total_errors": metrics["total_errors"],
            "error_rate": metrics["total_errors"] / max(1, metrics["total_errors"]),
            "top_error_categories": sorted(
                metrics["errors_by_category"].items(), key=lambda x: x[1], reverse=True
            )[:5],
            "error_severity_distribution": metrics["errors_by_severity"],
            "retry_attempts": metrics["retry_attempts"],
            "circuit_breaker_trips": metrics["circuit_breaker_trips"],
        }


def create_error_handler(service_name: str) -> ServiceErrorHandler:
    """Create service error handler instance.

    Args:
    ----
        service_name: Name of the service

    Returns:
    -------
        ServiceErrorHandler instance

    """
    return ServiceErrorHandler(service_name)
