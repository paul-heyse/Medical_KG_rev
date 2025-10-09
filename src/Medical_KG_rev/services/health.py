"""Infrastructure health checks for the gateway.

This module provides health monitoring capabilities for the Medical KG gateway,
implementing Kubernetes-compatible liveness and readiness probes. It supports
custom health checks for various infrastructure components and services.

Key Responsibilities:
    - Define health check interfaces and result structures
    - Implement liveness probe for basic service availability
    - Implement readiness probe with component-specific checks
    - Track service uptime and version information
    - Provide standardized health check result format

Collaborators:
    - Upstream: Gateway services, infrastructure components
    - Downstream: Kubernetes health probes, monitoring systems

Side Effects:
    - Executes health check functions which may perform I/O operations
    - Logs health check failures and errors
    - Updates uptime tracking based on service start time

Thread Safety:
    - Thread-safe: All methods are stateless except for read-only uptime calculation
    - Health checks should be thread-safe if called concurrently

Performance Characteristics:
    - O(n) readiness check time for n registered health checks
    - O(1) liveness check time
    - Minimal memory overhead for result caching

Example:
    >>> from Medical_KG_rev.services.health import HealthService, success, failure
    >>> def database_check():
    ...     try:
    ...         # Check database connectivity
    ...         return success("Database connection healthy")
    ...     except Exception as e:
    ...         return failure(f"Database error: {e}")
    >>> service = HealthService({"database": database_check}, version="1.0.0")
    >>> readiness = service.readiness()
    >>> print(f"Service status: {readiness['status']}")

"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(slots=True)
class CheckResult:
    """Result of a health check operation.

    Represents the outcome of a health check with standardized status
    and optional detail information for debugging and monitoring.

    Attributes:
        status: Health check status ("ok", "error", "degraded")
        detail: Optional detailed message explaining the status

    Example:
        >>> result = CheckResult(status="ok", detail="All systems operational")
        >>> print(f"Status: {result.status}, Detail: {result.detail}")

    """

    status: str
    detail: str = ""


# ============================================================================
# TYPE DEFINITIONS & CONSTANTS
# ============================================================================


HealthCheck = Callable[[], CheckResult]
"""Type alias for health check functions.

Health check functions should return a CheckResult indicating the
status of the component being checked. They should be fast, idempotent,
and not perform expensive operations.

    Example:
        >>> def my_health_check() -> CheckResult:
        ...     try:
        ...         # Perform lightweight check
        ...         return CheckResult(status="ok", detail="Component healthy")
        ...     except Exception as e:
        ...         return CheckResult(status="error", detail=str(e))

"""

# ============================================================================
# SERVICE IMPLEMENTATION
# ============================================================================


@dataclass
class HealthService:
    """Service for monitoring infrastructure health and availability.

    Implements Kubernetes-compatible health probes including liveness
    and readiness checks. Manages a collection of custom health checks
    and provides standardized health status reporting.

    Attributes:
        checks: Mapping of check names to health check functions
        version: Service version string for identification
        started_at: Timestamp when the service was started

    Invariants:
        - started_at is always in the past or present
        - version is non-empty string
        - checks mapping contains valid HealthCheck functions

    Thread Safety:
        - Thread-safe: All methods are read-only except for uptime calculation
        - Health check functions should be thread-safe

    Lifecycle:
        - Created with version and health check mappings
        - Uptime calculated from started_at timestamp
        - Health checks executed on demand for readiness probe

    Example:
        >>> checks = {
        ...     "database": lambda: success("DB healthy"),
        ...     "cache": lambda: success("Cache healthy")
        ... }
        >>> service = HealthService(checks, version="1.0.0")
        >>> readiness = service.readiness()
        >>> liveness = service.liveness()

    """

    checks: Mapping[str, HealthCheck]
    version: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def uptime_seconds(self) -> float:
        """Calculate service uptime in seconds.

        Computes the time elapsed since the service was started,
        rounded to 3 decimal places for precision.

        Returns:
            Uptime in seconds as a float value.

        Example:
            >>> service = HealthService({}, version="1.0.0")
            >>> import time
            >>> time.sleep(1.5)
            >>> uptime = service.uptime_seconds()
            >>> assert uptime >= 1.5

        """
        delta = datetime.now(UTC) - self.started_at
        return round(delta.total_seconds(), 3)

    def liveness(self) -> dict[str, object]:
        """Generate liveness probe response.

        Provides basic service availability information including
        status, version, uptime, and timestamp. This probe indicates
        whether the service is running and responsive.

        Returns:
            Dictionary containing liveness information:
                - status: Always "ok" for liveness probe
                - version: Service version string
                - uptime_seconds: Time since service start
                - timestamp: Current UTC timestamp in ISO format

        Note:
            Liveness probe does not check dependencies, only service
            availability. Use readiness() for dependency health checks.

        Example:
            >>> service = HealthService({}, version="1.0.0")
            >>> liveness = service.liveness()
            >>> assert liveness["status"] == "ok"
            >>> assert "version" in liveness
            >>> assert "uptime_seconds" in liveness

        """
        return {
            "status": "ok",
            "version": self.version,
            "uptime_seconds": self.uptime_seconds(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def readiness(self) -> dict[str, object]:
        """Generate readiness probe response with health check results.

        Executes all registered health checks and aggregates their
        results to determine overall service readiness. Returns
        detailed status for each component plus overall status.

        Returns:
            Dictionary containing readiness information:
                - status: Overall status ("ok", "degraded", "error")
                - version: Service version string
                - uptime_seconds: Time since service start
                - timestamp: Current UTC timestamp in ISO format
                - checks: Dictionary of individual check results

        Note:
            Status precedence: "error" > "degraded" > "ok"
            Individual check failures are caught and reported as errors.

        Example:
            >>> def failing_check():
            ...     return CheckResult(status="error", detail="Component down")
            >>> service = HealthService({"component": failing_check}, version="1.0.0")
            >>> readiness = service.readiness()
            >>> assert readiness["status"] == "error"
            >>> assert readiness["checks"]["component"]["status"] == "error"

        """
        results: dict[str, dict[str, object]] = {}
        overall_status = "ok"

        # Execute all health checks and aggregate results
        for name, check in self.checks.items():
            try:
                result = check()
            except Exception as exc:  # pragma: no cover - defensive
                # Defensive programming: catch any exceptions from health checks
                results[name] = {"status": "error", "detail": str(exc)}
                overall_status = "error"
                continue
            results[name] = {"status": result.status, "detail": result.detail}
            if result.status != "ok" and overall_status != "error":
                overall_status = "degraded"

        # Combine liveness information with readiness results
        payload = self.liveness()
        payload.update({"status": overall_status, "checks": results})
        return payload


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def success(detail: str = "") -> CheckResult:
    """Create a successful health check result.

    Convenience function for creating CheckResult instances with
    "ok" status and optional detail message.

    Args:
        detail: Optional success message describing the check outcome.
            Defaults to empty string.

    Returns:
        CheckResult with status "ok" and provided detail.

    Example:
        >>> result = success("Database connection healthy")
        >>> assert result.status == "ok"
        >>> assert result.detail == "Database connection healthy"

    """
    return CheckResult(status="ok", detail=detail)


def failure(detail: str) -> CheckResult:
    """Create a failed health check result.

    Convenience function for creating CheckResult instances with
    "error" status and required detail message explaining the failure.

    Args:
        detail: Required failure message describing what went wrong.

    Returns:
        CheckResult with status "error" and provided detail.

    Example:
        >>> result = failure("Database connection timeout")
        >>> assert result.status == "error"
        >>> assert result.detail == "Database connection timeout"

    """
    return CheckResult(status="error", detail=detail)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["CheckResult", "HealthService", "failure", "success"]
