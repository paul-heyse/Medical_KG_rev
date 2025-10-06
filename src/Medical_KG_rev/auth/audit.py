"""Audit logging utilities for security-sensitive actions."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from .context import SecurityContext

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AuditEntry:
    timestamp: datetime
    tenant_id: str
    subject: str
    action: str
    resource: str
    metadata: dict[str, object]


class AuditTrail:
    """In-memory audit log used for testing and local development."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(
        self,
        *,
        context: SecurityContext,
        action: str,
        resource: str,
        metadata: dict[str, object] | None = None,
    ) -> AuditEntry:
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
        items = [entry for entry in self._entries if entry.tenant_id == tenant_id]
        return sorted(items, key=lambda item: item.timestamp, reverse=True)[:limit]


_audit_trail = AuditTrail()


def get_audit_trail() -> AuditTrail:
    return _audit_trail
