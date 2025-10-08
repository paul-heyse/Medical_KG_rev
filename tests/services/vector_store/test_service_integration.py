from dataclasses import dataclass
from typing import Any

import pytest

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.vector_store.errors import DimensionMismatchError, NamespaceNotFoundError, ScopeError
from Medical_KG_rev.services.vector_store.models import IndexParams, NamespaceConfig, VectorQuery, VectorRecord
from Medical_KG_rev.services.vector_store.registry import NamespaceRegistry
from Medical_KG_rev.services.vector_store.service import VectorStoreService
from Medical_KG_rev.services.vector_store.types import VectorStorePort


@dataclass
class RecordingStore(VectorStorePort):
    created: list[dict[str, Any]]
    upserts: list[dict[str, Any]]
    queries: list[dict[str, Any]]
    deletes: list[dict[str, Any]]

    def create_or_update_collection(self, **kwargs):  # type: ignore[override]
        self.created.append(kwargs)

    def list_collections(self, **kwargs):  # type: ignore[override]
        return []

    def upsert(self, **kwargs):  # type: ignore[override]
        self.upserts.append(kwargs)

    def query(self, **kwargs):  # type: ignore[override]
        self.queries.append(kwargs)
        return []

    def delete(self, **kwargs):  # type: ignore[override]
        self.deletes.append(kwargs)
        return 0


@pytest.fixture()
def vector_service() -> tuple[VectorStoreService, RecordingStore, NamespaceRegistry]:
    store = RecordingStore(created=[], upserts=[], queries=[], deletes=[])
    registry = NamespaceRegistry()
    service = VectorStoreService(store, registry)
    return service, store, registry


def context(scopes: set[str]) -> SecurityContext:
    return SecurityContext(subject="tester", tenant_id="tenant", scopes=scopes)


def register_namespace(service: VectorStoreService, registry: NamespaceRegistry) -> None:
    config = NamespaceConfig(name="demo", params=IndexParams(dimension=128))
    service.ensure_namespace(
        context=context({"index:write"}),
        config=config,
    )


def test_ensure_namespace_registers_collection(vector_service) -> None:
    service, store, registry = vector_service
    register_namespace(service, registry)
    assert store.created[0]["namespace"] == "demo"
    assert registry.get(tenant_id="tenant", namespace="demo").name == "demo"


def test_upsert_requires_scope(vector_service) -> None:
    service, store, registry = vector_service
    register_namespace(service, registry)
    with pytest.raises(ScopeError):
        service.upsert(context=context({"index:read"}), namespace="demo", records=[])


def test_upsert_validates_dimensions(vector_service) -> None:
    service, store, registry = vector_service
    register_namespace(service, registry)
    record = VectorRecord(vector_id="1", values=[0.1] * 128, metadata={})
    service.upsert(context=context({"index:write"}), namespace="demo", records=[record])
    with pytest.raises(DimensionMismatchError):
        bad_record = VectorRecord(vector_id="2", values=[0.1] * 129, metadata={})
        service.upsert(context=context({"index:write"}), namespace="demo", records=[bad_record])


def test_query_requires_namespace_registration(vector_service) -> None:
    service, store, registry = vector_service
    with pytest.raises(NamespaceNotFoundError):
        service.query(
            context=context({"index:read"}),
            namespace="missing",
            query=VectorQuery(values=[0.1, 0.2]),
        )


def test_query_validates_dimension(vector_service) -> None:
    service, store, registry = vector_service
    register_namespace(service, registry)
    with pytest.raises(DimensionMismatchError):
        service.query(
            context=context({"index:read"}),
            namespace="demo",
            query=VectorQuery(values=[0.1, 0.2, 0.3]),
        )


def test_delete_requires_scope(vector_service) -> None:
    service, store, registry = vector_service
    register_namespace(service, registry)
    with pytest.raises(ScopeError):
        service.delete(context=context({"index:read"}), namespace="demo", ids=["1"])


def test_delete_passthrough(vector_service) -> None:
    service, store, registry = vector_service
    register_namespace(service, registry)
    service.delete(context=context({"index:write"}), namespace="demo", ids=["1"])
    assert store.deletes[0]["namespace"] == "demo"


def test_upsert_multiple_records(vector_service) -> None:
    service, store, registry = vector_service
    register_namespace(service, registry)
    records = [VectorRecord(vector_id=str(i), values=[0.1] * 128, metadata={}) for i in range(3)]
    service.upsert(context=context({"index:write"}), namespace="demo", records=records)
    assert store.upserts[0]["records"] == records
