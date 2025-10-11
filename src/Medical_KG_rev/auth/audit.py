"""Audit logging utilities for security-sensitive actions.

Audit logging helpers for authentication events.

The authentication subsystem relies on lightweight audit logging to track
security-sensitive actions such as key usage, scope violations, and rate-limit
denials. This module provides an in-memory implementation suitable for unit
tests and local development environments.

Key Responsibilities:
    - Capture structured audit entries for security events.
    - Provide query helpers for retrieving recent events per tenant.
    - Emit structured logs via ``structlog`` for external aggregation.

Collaborators:
    - Upstream: Authentication dependencies in ``dependencies.py``.
    - Downstream: Structured logging pipeline via ``structlog``.

Side Effects:
    - Emits log messages using the ``security.audit`` event name.

Thread Safety:
    - ``AuditTrail`` is not thread-safe; each worker should use a dedicated
      instance or guard access appropriately.

Performance Characteristics:
    - Operations are O(n) for retrieval based on tenant filtering. The in-memory
      list retains entries for the lifetime of the process.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import builtins
import structlog

from .context import SecurityContext


logger = structlog.get_logger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class AuditEntry:
    """Immutable representation of a security audit event.

    Attributes
    ----------
        timestamp: When the event occurred in UTC.
        tenant_id: Identifier of the tenant associated with the event.
        subject: Authenticated principal performing the action.
        action: Name of the action (e.g., ``"authenticate"``).
        resource: Resource acted upon.
        metadata: Additional contextual metadata for downstream consumers.

    """

    timestamp: datetime
    tenant_id: str
    subject: str
    action: str
    resource: str
    metadata: dict[str, object]


# ============================================================================
# AUDIT TRAIL IMPLEMENTATION
# ============================================================================


class AuditTrail:
    """Capture and query audit events for authentication flows.

    The default implementation stores audit entries in-memory, making it ideal
    for local development and automated tests. Production deployments should
    replace this implementation with a persistent backend via dependency
    injection.
    """

    def __init__(self) -> None:
        """Initialize the in-memory audit trail."""
        self._entries: list[AuditEntry] = []

    def record(
        self,
        *,
        context: SecurityContext,
        action: str,
        resource: str,
        metadata: dict[str, object] | None = None,
    ) -> AuditEntry:
        """Record a new audit event.

        Args:
        ----
            context: Security context associated with the request.
            action: Name of the performed action.
            resource: Resource identifier describing what was acted upon.
            metadata: Optional structured metadata.

        Returns:
        -------
            The stored :class:`AuditEntry` instance for further inspection.

        """
        entry = AuditEntry(
            timestamp=datetime.now(UTC),
            tenant_id=context.tenant_id,
            subject=context.identity,
            action=action,
            resource=resource,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        logger.info(
            "security.audit",
            tenant_id=entry.tenant_id,
            subject=entry.subject,
            action=entry.action,
            resource=entry.resource,
        )
        return entry

    def list(self, *, tenant_id: str, limit: int = 100) -> builtins.list[AuditEntry]:
        """Return the most recent audit entries for the provided tenant.

        Args:
        ----
            tenant_id: Tenant identifier to filter events.
            limit: Maximum number of entries to return.

        Returns:
        -------
            Sorted list of :class:`AuditEntry` instances ordered newest first.

        """
        items = [entry for entry in self._entries if entry.tenant_id == tenant_id]
        return sorted(items, key=lambda item: item.timestamp, reverse=True)[:limit]


# ============================================================================
# SINGLETONS
# ============================================================================


_audit_trail = AuditTrail()


def get_audit_trail() -> AuditTrail:
    """Return the global audit trail instance used by authentication flows."""
    return _audit_trail


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["AuditEntry", "AuditTrail", "get_audit_trail"]
