from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.ingestion.vector_ingestion import VectorIngestionService
from Medical_KG_rev.services.vector_store.models import (
    CompressionPolicy,
    IndexParams,
    NamespaceConfig,
    VectorRecord,
)
from Medical_KG_rev.services.vector_store.registry import NamespaceRegistry
from Medical_KG_rev.services.vector_store.service import VectorStoreService
from Medical_KG_rev.services.vector_store.stores.memory import InMemoryVectorStore


def _service() -> VectorIngestionService:
    store = InMemoryVectorStore()
    registry = NamespaceRegistry()
    service = VectorStoreService(store, registry)
    context = SecurityContext(subject="tester", tenant_id="tenant", scopes={"index:write"})
    namespace_config = NamespaceConfig(
        name="default",
        params=IndexParams(dimension=128),
        compression=CompressionPolicy(),
    )
    registry.register(tenant_id="tenant", config=namespace_config)
    service.ensure_namespace(context=context, config=namespace_config)
    return VectorIngestionService(vector_store=service)


def test_register_and_ingest_records() -> None:
    ingestion = _service()
    context = SecurityContext(subject="tester", tenant_id="tenant", scopes={"index:write"})
    result = ingestion.ingest_records(
        context=context,
        dataset="default",
        records=[VectorRecord(vector_id="1", values=[0.1] + [0.0] * 127)],
    )
    assert result.upserted == 1


def test_ingest_documents_maps_fields() -> None:
    ingestion = _service()
    context = SecurityContext(subject="tester", tenant_id="tenant", scopes={"index:write"})
    result = ingestion.ingest_documents(
        context=context,
        dataset="default",
        documents=[{"id": "2", "vector": [0.2] + [0.0] * 127, "metadata": {"source": "test"}}],
    )
    assert result.upserted == 1
