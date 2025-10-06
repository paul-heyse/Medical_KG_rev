"""Security context primitives shared across FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class SecurityContext:
    """Represents the authenticated principal for the current request."""

    subject: str
    tenant_id: str
    scopes: set[str] = field(default_factory=set)
    expires_at: datetime | None = None
    claims: Mapping[str, object] = field(default_factory=dict)
    auth_type: str = "oauth"
    token: str | None = None
    key_id: str | None = None

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "*" in self.scopes

    @property
    def identity(self) -> str:
        """Return stable identifier used for rate limiting/logging."""

        return self.key_id or self.subject

    def with_scope(self, *extra_scopes: str) -> SecurityContext:
        merged_scopes: set[str] = set(self.scopes).union(extra_scopes)
        data: MutableMapping[str, object] = {
            "subject": self.subject,
            "tenant_id": self.tenant_id,
            "scopes": merged_scopes,
            "expires_at": self.expires_at,
            "claims": self.claims,
            "auth_type": self.auth_type,
            "token": self.token,
            "key_id": self.key_id,
        }
        return SecurityContext(**data)  # type: ignore[arg-type]
