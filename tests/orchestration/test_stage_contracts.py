from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.adapters.plugins.models import AdapterDomain, AdapterRequest
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.stages import build_default_stage_factory
from Medical_KG_rev.orchestration.stages.contracts import (
    ChunkStage,
    EmbedStage,
    EmbeddingBatch,
    ExtractStage,
    GraphWriteReceipt,
    IngestStage,
    IndexReceipt,
    IndexStage,
    KGStage,
    ParseStage,
    StageContext,
)


class StubPluginManager:
    def __init__(self) -> None:
        self.invocations: list[tuple[str, AdapterRequest]] = []

    def invoke(self, adapter: str, request: AdapterRequest, *, strict: bool = False):
        self.invocations.append((adapter, request))
        payload = {"text": "Example abstract for testing", "title": "Test"}
        return SimpleNamespace(response=SimpleNamespace(items=[payload]))


@pytest.fixture()
def stage_context() -> StageContext:
    return StageContext(
        tenant_id="tenant-a",
        doc_id="doc-1",
        correlation_id="corr-1",
        metadata={"title": "Sample"},
        pipeline_name="auto",
        pipeline_version="2024-01-01",
    )


@pytest.fixture()
def adapter_request() -> AdapterRequest:
    return AdapterRequest(
        tenant_id="tenant-a",
        correlation_id="corr-1",
        domain=AdapterDomain.BIOMEDICAL,
        parameters={"adapter": "clinical-trials"},
    )


def _definition(stage_type: str, name: str, config: dict | None = None) -> StageDefinition:
    payload = {"name": name, "type": stage_type, "policy": "default"}
    if config:
        payload["config"] = config
    return StageDefinition.model_validate(payload)


def test_default_stage_factory_complies_with_protocols(stage_context, adapter_request):
    manager = StubPluginManager()
    registry = build_default_stage_factory(manager)

    ingest = registry.get_builder("ingest")(
        _definition("ingest", "ingest", {"adapter": "clinical-trials", "strict": False})
    )
    assert isinstance(ingest, IngestStage)
    payloads = ingest.execute(stage_context, adapter_request)
    assert payloads and isinstance(payloads[0], dict)

    parse = registry.get_builder("parse")(_definition("parse", "parse"))
    assert isinstance(parse, ParseStage)
    document = parse.execute(stage_context, payloads)

    validator = registry.get_builder("ir-validation")(_definition("ir-validation", "ir_validation"))
    assert isinstance(validator, ParseStage)
    validated = validator.execute(stage_context, document)
    assert validated is document

    chunker = registry.get_builder("chunk")(_definition("chunk", "chunk"))
    assert isinstance(chunker, ChunkStage)
    chunks = chunker.execute(stage_context, document)
    assert chunks and chunks[0].doc_id == document.id

    embedder = registry.get_builder("embed")(_definition("embed", "embed"))
    assert isinstance(embedder, EmbedStage)
    batch = embedder.execute(stage_context, chunks)
    assert isinstance(batch, EmbeddingBatch)
    assert batch.vectors

    indexer = registry.get_builder("index")(_definition("index", "index"))
    assert isinstance(indexer, IndexStage)
    receipt = indexer.execute(stage_context, batch)
    assert isinstance(receipt, IndexReceipt)
    assert receipt.chunks_indexed == len(batch.vectors)

    extractor = registry.get_builder("extract")(_definition("extract", "extract"))
    assert isinstance(extractor, ExtractStage)
    entities, claims = extractor.execute(stage_context, document)
    assert entities == [] and claims == []

    kg_stage = registry.get_builder("knowledge-graph")(_definition("knowledge-graph", "kg"))
    assert isinstance(kg_stage, KGStage)
    graph_receipt = kg_stage.execute(stage_context, entities, claims)
    assert isinstance(graph_receipt, GraphWriteReceipt)
    assert graph_receipt.nodes_written == 0

    assert manager.invocations and manager.invocations[0][0] == "clinical-trials"
