from pathlib import Path
from types import SimpleNamespace

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.embedding.service import EmbeddingWorker
from Medical_KG_rev.services.ingestion import IngestionOptions, IngestionService
from Medical_KG_rev.services.retrieval.faiss_index import FAISSIndex
from Medical_KG_rev.services.retrieval.opensearch_client import OpenSearchClient
from Medical_KG_rev.services.vector_store.models import IndexParams, NamespaceConfig
from Medical_KG_rev.services.vector_store.registry import NamespaceRegistry
from Medical_KG_rev.services.vector_store.service import VectorStoreService
from Medical_KG_rev.services.vector_store.stores.memory import InMemoryVectorStore


class _StubChunkingService:
    def chunk(self, tenant_id: str, document_id: str, text: str, options=None):  # noqa: D401 - simple stub
        return [
            SimpleNamespace(
                chunk_id=f"{document_id}:0",
                id=f"{document_id}:0",
                body=text,
                doc_id=document_id,
                granularity="paragraph",
                chunker="stub",
                meta={"text": text},
            )
        ]


def test_ingestion_pipeline_persists_embeddings() -> None:
    chunking = _StubChunkingService()
    config_path = Path(__file__).resolve().parents[2] / "config" / "embeddings.yaml"
    worker = EmbeddingWorker(config_path=str(config_path))
    registry = NamespaceRegistry()
    namespace = "single_vector.bge_small_en.384.v1"
    registry.register(
        tenant_id="tenant",
        config=NamespaceConfig(name=namespace, params=IndexParams(dimension=384)),
    )
    store = InMemoryVectorStore()
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace=namespace,
        params=IndexParams(dimension=384),
        metadata={},
    )
    vector_service = VectorStoreService(store=store, registry=registry)
    ingestion = IngestionService(
        chunking=chunking,
        embedding_worker=worker,
        vector_store=vector_service,
        opensearch=OpenSearchClient(),
        faiss=FAISSIndex(384),
    )
    context = SecurityContext(subject="user", tenant_id="tenant", scopes={"index:write"})
    options = IngestionOptions(namespaces=[namespace])
    result = ingestion.ingest(
        tenant_id="tenant",
        document_id="doc-1",
        text="Clinical trial data on hypertension treatment",
        context=context,
        options=options,
    )
    assert result.stored["single_vector"] >= 1
    assert result.metrics.total == 1
