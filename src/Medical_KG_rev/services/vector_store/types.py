"""Protocol definitions for vector store adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .models import (
    CompressionPolicy,
    HealthStatus,
    IndexParams,
    RebuildReport,
    SnapshotInfo,
    VectorMatch,
    VectorQuery,
    VectorRecord,
)


class VectorStorePort(Protocol):
    """Protocol all vector store adapters must implement."""

    def create_or_update_collection(
        self,
        *,
        tenant_id: str,
        namespace: str,
        params: IndexParams,
        compression: CompressionPolicy,
        metadata: Mapping[str, object] | None = None,
        named_vectors: Mapping[str, IndexParams] | None = None,
    ) -> None:
        """Ensure the target namespace exists with the provided parameters."""

    def list_collections(self, *, tenant_id: str) -> Sequence[str]:
        """Return namespaces available for the tenant."""

    def upsert(
        self,
        *,
        tenant_id: str,
        namespace: str,
        records: Sequence[VectorRecord],
    ) -> None:
        """Insert or update the supplied vector records."""

    def query(
        self,
        *,
        tenant_id: str,
        namespace: str,
        query: VectorQuery,
    ) -> Sequence[VectorMatch]:
        """Execute a vector similarity query."""

    def delete(
        self,
        *,
        tenant_id: str,
        namespace: str,
        vector_ids: Sequence[str],
    ) -> int:
        """Delete the specified vector identifiers and return the removed count."""

    def create_snapshot(
        self,
        *,
        tenant_id: str,
        namespace: str,
        destination: str,
        include_payloads: bool = True,
    ) -> SnapshotInfo:
        """Persist a point-in-time snapshot of the namespace to the destination path."""

    def restore_snapshot(
        self,
        *,
        tenant_id: str,
        namespace: str,
        source: str,
        overwrite: bool = False,
    ) -> RebuildReport:
        """Restore a namespace from a previously created snapshot archive."""

    def rebuild_index(
        self,
        *,
        tenant_id: str,
        namespace: str,
        force: bool = False,
    ) -> RebuildReport:
        """Trigger index retraining or full rebuild for the namespace."""

    def check_health(
        self,
        *,
        tenant_id: str,
        namespace: str | None = None,
    ) -> Mapping[str, HealthStatus]:
        """Return health details for the backend or specific namespace."""
