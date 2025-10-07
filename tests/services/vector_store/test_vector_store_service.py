from __future__ import annotations

import pytest

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.vector_store import (
    CompressionPolicy,
    DimensionMismatchError,
    IndexParams,
    NamespaceConfig,
    NamespaceRegistry,
    ScopeError,
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
    context = SecurityContext(subject="tester", tenant_id="tenant-a", scopes={"index:write", "index:read"})
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
    return SecurityContext(subject="tester", tenant_id="tenant-a", scopes={"index:write", "index:read"})


def test_upsert_and_query_returns_results(service: VectorStoreService, context: SecurityContext) -> None:
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
