"""Scope definitions and helpers."""

from __future__ import annotations

from typing import Dict


class Scopes:
    """Canonical scope names used across the platform."""

    INGEST_WRITE = "ingest:write"
    JOBS_READ = "jobs:read"
    JOBS_WRITE = "jobs:write"
    EMBED_WRITE = "embed:write"
    RETRIEVE_READ = "kg:read"
    KG_WRITE = "kg:write"
    PROCESS_WRITE = "process:write"
    AUDIT_READ = "audit:read"


SCOPE_DESCRIPTIONS: Dict[str, str] = {
    Scopes.INGEST_WRITE: "Submit ingestion jobs",
    Scopes.JOBS_READ: "Read job status",
    Scopes.JOBS_WRITE: "Cancel or mutate jobs",
    Scopes.EMBED_WRITE: "Generate embeddings",
    Scopes.RETRIEVE_READ: "Search the knowledge graph",
    Scopes.KG_WRITE: "Write to the knowledge graph",
    Scopes.PROCESS_WRITE: "Execute processing utilities (chunking, extraction)",
    Scopes.AUDIT_READ: "Read audit logs",
}
