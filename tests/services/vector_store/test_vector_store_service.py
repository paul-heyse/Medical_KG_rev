from __future__ import annotations

from pathlib import Path

import pytest

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.vector_store import (
    CompressionPolicy,
    DimensionMismatchError,
    HealthStatus,
    IndexParams,
    NamespaceConfig,
    NamespaceRegistry,
    ScopeError,
    SnapshotInfo,
    VectorQuery,
    VectorRecord,
    VectorStoreService,
)
from Medical_KG_rev.services.vector_store.stores.memory import InMemoryVectorStore


@pytest.fixture()
def service() -> VectorStoreService:
    registry = NamespaceRegistry()
    store = InMemoryVectorStore()
    svc = VectorStoreService(store, registry)
    context = SecurityContext(
        subject="tester", tenant_id="tenant-a", scopes={"index:write", "index:read"}
    )
    config = NamespaceConfig(
        name="dense.test.128.v1",
        params=IndexParams(dimension=128, kind="hnsw"),
        compression=CompressionPolicy(kind="none"),
        version="v1",
    )
    svc.ensure_namespace(context=context, config=config)
    return svc


@pytest.fixture()
def context() -> SecurityContext:
    return SecurityContext(
        subject="tester", tenant_id="tenant-a", scopes={"index:write", "index:read"}
    )


def test_upsert_and_query_returns_results(
    service: VectorStoreService, context: SecurityContext
) -> None:
    base = [0.0] * 128
    vec1 = base.copy()
    vec1[0] = 1.0
    vec2 = base.copy()
    vec2[1] = 1.0
    records = [
        VectorRecord(vector_id="vec-1", values=vec1, metadata={"text": "alpha"}),
        VectorRecord(vector_id="vec-2", values=vec2, metadata={"text": "beta"}),
    ]
    service.upsert(context=context, namespace="dense.test.128.v1", records=records)

    matches = service.query(
        context=context,
        namespace="dense.test.128.v1",
        query=VectorQuery(values=[1.0] + [0.0] * 127, top_k=2),
    )

    assert len(matches) == 2
    assert matches[0].metadata["text"] == "alpha"


def test_dimension_mismatch_raises(service: VectorStoreService, context: SecurityContext) -> None:
    records = [VectorRecord(vector_id="vec-1", values=[1.0, 0.0], metadata={})]
    with pytest.raises(DimensionMismatchError):
        service.upsert(context=context, namespace="dense.test.128.v1", records=records)


def test_scope_enforced(service: VectorStoreService, context: SecurityContext) -> None:
    no_scope = SecurityContext(subject="tester", tenant_id="tenant-a", scopes={"index:read"})
    records = [VectorRecord(vector_id="vec-1", values=[1.0] + [0.0] * 127, metadata={})]
    with pytest.raises(ScopeError):
        service.upsert(context=no_scope, namespace="dense.test.128.v1", records=records)


def test_snapshot_restore_round_trip(
    service: VectorStoreService, context: SecurityContext, tmp_path: Path
) -> None:
    base = [0.0] * 128
    record = VectorRecord(vector_id="vec-1", values=[1.0] + base[1:], metadata={"text": "alpha"})
    service.upsert(context=context, namespace="dense.test.128.v1", records=[record])

    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "snapshot.json"
    snapshot = service.create_snapshot(
        context=context,
        namespace="dense.test.128.v1",
        destination=str(snapshot_path),
    )
    assert isinstance(snapshot, SnapshotInfo)
    service.delete(context=context, namespace="dense.test.128.v1", vector_ids=["vec-1"])
    matches = service.query(
        context=context,
        namespace="dense.test.128.v1",
        query=VectorQuery(values=[1.0] + base[1:], top_k=1),
    )
    assert not matches
    report = service.restore_snapshot(
        context=context,
        namespace="dense.test.128.v1",
        source=snapshot.path,
        overwrite=True,
    )
    assert report.rebuilt
    restored = service.query(
        context=context,
        namespace="dense.test.128.v1",
        query=VectorQuery(values=[1.0] + base[1:], top_k=1),
    )
    assert restored and restored[0].metadata["text"] == "alpha"


def test_rebuild_namespace_returns_report(
    service: VectorStoreService, context: SecurityContext
) -> None:
    report = service.rebuild_namespace(
        context=context,
        namespace="dense.test.128.v1",
        force=True,
    )
    assert report.rebuilt


def test_health_check_reports_namespaces(
    service: VectorStoreService, context: SecurityContext
) -> None:
    statuses = service.check_health(context=context)
    assert "dense.test.128.v1" in statuses
    status = statuses["dense.test.128.v1"]
    assert isinstance(status, HealthStatus)
    assert status.healthy
