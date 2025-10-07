from Medical_KG_rev.models.ir import Block, Document, Section
from Medical_KG_rev.services.ingestion import IngestionService


def _document() -> Document:
    blocks = [
        Block(id="b-1", text="Introduction with rationale."),
        Block(id="b-2", text="Methods describing design."),
        Block(id="b-3", text="Results summarising outcomes."),
    ]
    section = Section(id="s-1", title="Trial", blocks=blocks)
    return Document(
        id="doc-ingest",
        source="pmc",
        title="Ingestion Sample",
        sections=[section],
        metadata={"tei_xml": "<TEI><text><body><div type='intro'><head>Intro</head></div></body></text></TEI>"},
    )


def test_ingestion_service_chunking_run() -> None:
    service = IngestionService()
    result = service.chunk_document(_document(), tenant_id="tenant", source_hint="pmc")
    assert result.chunks
    assert result.granularity_counts
    stored = service.list_chunks("tenant", "doc-ingest")
    assert stored

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
    namespace_state = store._state["tenant"][namespace]
    assert namespace_state.metadata
    stored_meta = next(iter(namespace_state.metadata.values()))
    assert stored_meta.get("tenant_id") == "tenant"
