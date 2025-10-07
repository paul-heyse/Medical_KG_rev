from pathlib import Path

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.embedding.service import EmbeddingResponse, EmbeddingVector
from Medical_KG_rev.services.retrieval.faiss_index import FAISSIndex
from Medical_KG_rev.services.retrieval.opensearch_client import OpenSearchClient
from Medical_KG_rev.services.retrieval.retrieval_service import RetrievalService
from Medical_KG_rev.services.vector_store.models import IndexParams, NamespaceConfig, VectorRecord
from Medical_KG_rev.services.vector_store.registry import NamespaceRegistry
from Medical_KG_rev.services.vector_store.service import VectorStoreService
from Medical_KG_rev.services.vector_store.stores.memory import InMemoryVectorStore


class _StubEmbeddingWorker:
    def __init__(self, namespace: str, dimension: int) -> None:
        self.namespace = namespace
        self.dimension = dimension
        self.calls = 0

    @property
    def active_namespaces(self) -> list[str]:
        return [self.namespace]

    @property
    def namespace_weights(self) -> dict[str, float]:
        return {self.namespace: 1.0}

    def encode_queries(self, request):  # pragma: no cover - signature provided by EmbeddingWorker
        self.calls += 1
        vector = [0.1] * self.dimension
        return EmbeddingResponse(
            vectors=[
                EmbeddingVector(
                    id="query",
                    model="bge",
                    namespace=self.namespace,
                    kind="single_vector",
                    vectors=[vector],
                    terms=None,
                    dimension=self.dimension,
                    metadata={},
                )
            ]
        )


def test_retrieval_query_caching() -> None:
    namespace = "single_vector.bge_small_en.384.v1"
    registry = NamespaceRegistry()
    registry.register(tenant_id="tenant", config=NamespaceConfig(name=namespace, params=IndexParams(dimension=384)))
    store = InMemoryVectorStore()
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace=namespace,
        params=IndexParams(dimension=384),
        metadata={},
    )
    store.upsert(
        tenant_id="tenant",
        namespace=namespace,
        records=[VectorRecord(vector_id="doc-1", values=[0.1] * 384, metadata={"text": "foo"})],
    )
    vector_service = VectorStoreService(store=store, registry=registry)
    worker = _StubEmbeddingWorker(namespace=namespace, dimension=384)
    retrieval = RetrievalService(
        opensearch=OpenSearchClient(),
        faiss=FAISSIndex(384),
        vector_store=vector_service,
        vector_namespace=namespace,
        embedding_worker=worker,
        active_namespaces=[namespace],
    )
    context = SecurityContext(subject="user", tenant_id="tenant", scopes={"index:read", "index:write"})
    results_first = retrieval._vector_store_search("hypertension", 3, context)
    assert results_first
    calls_after_first = worker.calls
    results_second = retrieval._vector_store_search("hypertension", 3, context)
    assert results_second
    assert worker.calls == calls_after_first
