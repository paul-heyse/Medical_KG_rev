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
        metadata={
            "tei_xml": "<TEI><text><body><div type='intro'><head>Intro</head></div></body></text></TEI>"
        },
    )


def test_ingestion_service_chunking_run() -> None:
    service = IngestionService()
    result = service.chunk_document(_document(), tenant_id="tenant", source_hint="pmc")
    assert result.chunks
    assert result.granularity_counts
    stored = service.list_chunks("tenant", "doc-ingest")
    assert stored
