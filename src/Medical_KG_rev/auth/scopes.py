"""Scope definitions shared across authentication and authorization flows."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================


# (No additional imports required; section present for consistency.)


# ============================================================================
# SCOPE CONSTANTS
# ============================================================================


class Scopes:
    """Canonical scope names used across the platform."""

    INGEST_WRITE = "ingest:write"
    JOBS_READ = "jobs:read"
    JOBS_WRITE = "jobs:write"
    EMBED_READ = "embed:read"
    EMBED_WRITE = "embed:write"
    EMBED_ADMIN = "embed:admin"
    RETRIEVE_READ = "kg:read"
    KG_WRITE = "kg:write"
    PROCESS_WRITE = "process:write"
    AUDIT_READ = "audit:read"
    ADAPTERS_READ = "adapters:read"
    EVALUATE_WRITE = "evaluate:write"


# ============================================================================
# DESCRIPTIONS
# ============================================================================


SCOPE_DESCRIPTIONS: dict[str, str] = {
    Scopes.INGEST_WRITE: "Submit ingestion jobs",
    Scopes.JOBS_READ: "Read job status",
    Scopes.JOBS_WRITE: "Cancel or mutate jobs",
    Scopes.EMBED_READ: "Read embedding metadata and namespace catalogs",
    Scopes.EMBED_WRITE: "Generate embeddings",
    Scopes.EMBED_ADMIN: "Administer embedding namespaces and storage",
    Scopes.RETRIEVE_READ: "Search the knowledge graph",
    Scopes.KG_WRITE: "Write to the knowledge graph",
    Scopes.PROCESS_WRITE: "Execute processing utilities (chunking, extraction)",
    Scopes.AUDIT_READ: "Read audit logs",
    Scopes.ADAPTERS_READ: "List and inspect adapter plugins",
    Scopes.EVALUATE_WRITE: "Run retrieval evaluation jobs",
}
"""Human-readable descriptions for each defined scope."""


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["SCOPE_DESCRIPTIONS", "Scopes"]
