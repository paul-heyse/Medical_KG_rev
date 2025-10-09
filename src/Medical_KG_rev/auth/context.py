"""Security context primitives shared across FastAPI dependencies."""

"""Security context utilities shared across authentication flows.

The context encapsulates authentication metadata derived from API keys or JWT
tokens. Downstream services use the context to perform authorization checks,
emit telemetry, and scope rate limiting decisions.
"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class SecurityContext:
    """Represents the authenticated principal for the current request.

    Attributes:
        subject: Subject identifier from the authentication token.
        tenant_id: Tenant associated with the request.
        scopes: Granted scope set used for authorization decisions.
        expires_at: Optional token expiry timestamp.
        claims: Raw token claims useful for auditing and downstream services.
        auth_type: Authentication mechanism (``"oauth"`` or ``"api_key"``).
        token: Raw bearer token when OAuth authentication is used.
        key_id: Identifier for API-key based authentication.

    Example:
        >>> context = SecurityContext(
        ...     subject="user:42",
        ...     tenant_id="tenant-a",
        ...     scopes={"ingest:write"},
        ... )
        >>> context.has_scope("ingest:write")
        True
    """

    subject: str
    tenant_id: str
    scopes: set[str] = field(default_factory=set)
    expires_at: datetime | None = None
    claims: Mapping[str, object] = field(default_factory=dict)
    auth_type: str = "oauth"
    token: str | None = None
    key_id: str | None = None

    def has_scope(self, scope: str) -> bool:
        """Return ``True`` when the provided scope is authorized.

        Args:
            scope: Scope string to evaluate.

        Returns:
            ``True`` if the context includes the scope or a wildcard.
        """

        return scope in self.scopes or "*" in self.scopes

    @property
    def identity(self) -> str:
        """Return the canonical identifier used for logging and rate limiting.

        Returns:
            Stable identifier string combining subject or key ID.
        """

        return self.key_id or self.subject

    def with_scope(self, *extra_scopes: str) -> SecurityContext:
        """Return a new context with the provided scopes merged in.

        Args:
            *extra_scopes: Additional scopes to grant on the derived context.

        Returns:
            A new :class:`SecurityContext` instance containing the merged scope
            set while preserving other metadata.
        """

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


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["SecurityContext"]
