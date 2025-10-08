from types import SimpleNamespace

import pytest

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
    PipelineState,
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
    state = PipelineState.initialise(context=stage_context, adapter_request=adapter_request)

    ingest = registry["ingest"](
        _definition("ingest", "ingest", {"adapter": "clinical-trials", "strict": False})
    )
    assert isinstance(ingest, IngestStage)
    payloads = ingest.execute(stage_context, state)
    assert payloads and isinstance(payloads[0], dict)
    state.apply_stage_output("ingest", "ingest", payloads)

    parse = registry["parse"](_definition("parse", "parse"))
    assert isinstance(parse, ParseStage)
    document = parse.execute(stage_context, state)
    state.apply_stage_output("parse", "parse", document)

    validator = registry["ir-validation"](_definition("ir-validation", "ir_validation"))
    assert isinstance(validator, ParseStage)
    validated = validator.execute(stage_context, state)
    assert validated is document
    state.apply_stage_output("ir-validation", "ir_validation", validated)

    chunker = registry["chunk"](_definition("chunk", "chunk"))
    assert isinstance(chunker, ChunkStage)
    chunks = chunker.execute(stage_context, state)
    assert chunks and chunks[0].doc_id == document.id
    state.apply_stage_output("chunk", "chunk", chunks)

    embedder = registry["embed"](_definition("embed", "embed"))
    assert isinstance(embedder, EmbedStage)
    batch = embedder.execute(stage_context, state)
    assert isinstance(batch, EmbeddingBatch)
    assert batch.vectors
    state.apply_stage_output("embed", "embed", batch)

    indexer = registry["index"](_definition("index", "index"))
    assert isinstance(indexer, IndexStage)
    receipt = indexer.execute(stage_context, state)
    assert isinstance(receipt, IndexReceipt)
    assert receipt.chunks_indexed == len(batch.vectors)
    state.apply_stage_output("index", "index", receipt)

    extractor = registry["extract"](_definition("extract", "extract"))
    assert isinstance(extractor, ExtractStage)
    entities, claims = extractor.execute(stage_context, state)
    assert entities == [] and claims == []
    state.apply_stage_output("extract", "extract", (entities, claims))

    kg_stage = registry["knowledge-graph"](_definition("knowledge-graph", "kg"))
    assert isinstance(kg_stage, KGStage)
    graph_receipt = kg_stage.execute(stage_context, state)
    assert isinstance(graph_receipt, GraphWriteReceipt)
    assert graph_receipt.nodes_written == 0
    state.apply_stage_output("knowledge-graph", "kg", graph_receipt)

    assert manager.invocations and manager.invocations[0][0] == "clinical-trials"
    assert state.has_document() and state.has_embeddings()
    assert state.serialise()["chunk_count"] == len(chunks)
