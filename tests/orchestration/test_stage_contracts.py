from __future__ import annotations

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
    StageContext,
)


class StubManager:
    def __init__(self) -> None:
        self.invocations: list[str] = []

    def invoke(self, adapter: str, request: AdapterRequest, *, strict: bool = False):  # pragma: no cover - simple stub
        self.invocations.append(adapter)
        payload = {
            "doc_id": request.parameters.get("doc_id", "doc-001"),
            "title": "Example",
            "body": "Population: adults Intervention: treatment Outcome: success",
        }
        return SimpleNamespace(response=SimpleNamespace(items=[payload]))


@pytest.fixture()
def stage_context() -> StageContext:
    return StageContext(tenant_id="tenant-a", doc_id="doc-001", correlation_id="corr-123")


@pytest.fixture()
def adapter_request() -> AdapterRequest:
    return AdapterRequest(
        tenant_id="tenant-a",
        correlation_id="corr-123",
        domain=AdapterDomain.BIOMEDICAL,
        parameters={"doc_id": "doc-001", "adapter": "stub"},
    )


def _build_stage_definition(name: str, stage_type: str, config: dict[str, object] | None = None) -> StageDefinition:
    return StageDefinition(name=name, stage_type=stage_type, config=config or {})


def test_default_stage_factory_contracts(stage_context: StageContext, adapter_request: AdapterRequest) -> None:
    manager = StubManager()
    registry = build_default_stage_factory(manager)

    ingest = registry["ingest"](_build_stage_definition("ingest", "ingest", {"adapter": "stub"}))
    assert isinstance(ingest, IngestStage)
    payloads = ingest.execute(stage_context, adapter_request)
    assert isinstance(payloads, list) and isinstance(payloads[0], dict)

    parse = registry["parse"](_build_stage_definition("parse", "parse"))
    assert isinstance(parse, ParseStage)
    document = parse.execute(stage_context, payloads)

    validate = registry["ir-validation"](_build_stage_definition("validate", "ir-validation"))
    assert isinstance(validate, ParseStage)
    validated = validate.execute(stage_context, document)
    assert validated is document

    chunk = registry["chunk"](_build_stage_definition("chunk", "chunk"))
    assert isinstance(chunk, ChunkStage)
    chunks = chunk.execute(stage_context, document)
    assert chunks and chunks[0].doc_id == document.id

    embed = registry["embed"](_build_stage_definition("embed", "embed"))
    assert isinstance(embed, EmbedStage)
    batch = embed.execute(stage_context, chunks)
    assert isinstance(batch, EmbeddingBatch)
    assert batch.vectors

    index = registry["index"](_build_stage_definition("index", "index"))
    assert isinstance(index, IndexStage)
    receipt = index.execute(stage_context, batch)
    assert isinstance(receipt, IndexReceipt)
    assert receipt.chunks_indexed == len(batch.vectors)

    extract = registry["extract"](_build_stage_definition("extract", "extract"))
    assert isinstance(extract, ExtractStage)
    entities, claims = extract.execute(stage_context, document)
    assert isinstance(entities, list) and isinstance(claims, list)

    kg = registry["knowledge-graph"](_build_stage_definition("kg", "knowledge-graph"))
    assert isinstance(kg, KGStage)
    graph_receipt = kg.execute(stage_context, entities, claims)
    assert isinstance(graph_receipt, GraphWriteReceipt)
    assert graph_receipt.nodes_written == len(entities)

    assert manager.invocations == ["stub"]
