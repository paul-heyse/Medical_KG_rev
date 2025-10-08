from types import SimpleNamespace

import pytest

from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.adapters.plugins.models import AdapterDomain, AdapterRequest
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.runtime import build_stage_factory
from Medical_KG_rev.orchestration.dagster.stages import create_default_pipeline_resource
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.dagster.runtime import StageFactory
from Medical_KG_rev.orchestration.dagster.stages import create_stage_plugin_manager
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
from Medical_KG_rev.orchestration.stages.plugin_manager import (
    StagePlugin,
    StagePluginContext,
    StagePluginExecutionError,
    StagePluginManager,
    StagePluginMetadata,
    StagePluginNotAvailable,
)


class StubPluginManager(AdapterPluginManager):
    def __init__(self) -> None:
        super().__init__(project_name="test.adapters")
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
    pipeline_resource = create_default_pipeline_resource()
    factory = build_stage_factory(manager, pipeline_resource, JobLedger())
    state = PipelineState.initialise(context=stage_context, adapter_request=adapter_request)

    ingest = factory.resolve(
        "auto",
    factory = StageFactory(create_stage_plugin_manager(manager))
    state = PipelineState.initialise(context=stage_context, adapter_request=adapter_request)

    ingest = factory.resolve(
        "default",
        _definition("ingest", "ingest", {"adapter": "clinical-trials", "strict": False}),
    )
    assert isinstance(ingest, IngestStage)
    payloads = ingest.execute(stage_context, state)
    assert payloads and isinstance(payloads[0], dict)
    state.apply_stage_output("ingest", "ingest", payloads)

    parse = factory.resolve("auto", _definition("parse", "parse"))
    parse = factory.resolve("default", _definition("parse", "parse"))
    assert isinstance(parse, ParseStage)
    document = parse.execute(stage_context, state)
    state.apply_stage_output("parse", "parse", document)

    validator = factory.resolve("auto", _definition("ir-validation", "ir_validation"))
    validator = factory.resolve("default", _definition("ir-validation", "ir_validation"))
    assert isinstance(validator, ParseStage)
    validated = validator.execute(stage_context, state)
    assert validated is document
    state.apply_stage_output("ir-validation", "ir_validation", validated)

    chunker = factory.resolve("auto", _definition("chunk", "chunk"))
    chunker = factory.resolve("default", _definition("chunk", "chunk"))
    assert isinstance(chunker, ChunkStage)
    chunks = chunker.execute(stage_context, state)
    assert chunks and chunks[0].doc_id == document.id
    state.apply_stage_output("chunk", "chunk", chunks)

    embedder = factory.resolve("auto", _definition("embed", "embed"))
    embedder = factory.resolve("default", _definition("embed", "embed"))
    assert isinstance(embedder, EmbedStage)
    batch = embedder.execute(stage_context, state)
    assert isinstance(batch, EmbeddingBatch)
    assert batch.vectors
    state.apply_stage_output("embed", "embed", batch)

    indexer = factory.resolve("auto", _definition("index", "index"))
    indexer = factory.resolve("default", _definition("index", "index"))
    assert isinstance(indexer, IndexStage)
    receipt = indexer.execute(stage_context, state)
    assert isinstance(receipt, IndexReceipt)
    assert receipt.chunks_indexed == len(batch.vectors)
    state.apply_stage_output("index", "index", receipt)

    extractor = factory.resolve("auto", _definition("extract", "extract"))
    extractor = factory.resolve("default", _definition("extract", "extract"))
    assert isinstance(extractor, ExtractStage)
    entities, claims = extractor.execute(stage_context, state)
    assert entities == [] and claims == []
    state.apply_stage_output("extract", "extract", (entities, claims))

    kg_stage = factory.resolve("auto", _definition("knowledge-graph", "kg"))
    kg_stage = factory.resolve("default", _definition("knowledge-graph", "kg"))
    assert isinstance(kg_stage, KGStage)
    graph_receipt = kg_stage.execute(stage_context, state)
    assert isinstance(graph_receipt, GraphWriteReceipt)
    assert graph_receipt.nodes_written == 0
    state.apply_stage_output("knowledge-graph", "kg", graph_receipt)

    download_stage = factory.resolve(
        "default",
        _definition("download", "download", {"urls": ["s3://bucket/doc.pdf"]}),
    )
    assets = download_stage.execute(stage_context, state)
    assert isinstance(assets, list)
    state.apply_stage_output("download", "download", assets)

    gate_stage = factory.resolve("default", _definition("gate", "pdf_gate"))
    gate_ready = gate_stage.execute(stage_context, state)
    assert gate_ready in {True, False}
    state.apply_stage_output("gate", "pdf_gate", gate_ready)

    assert manager.invocations and manager.invocations[0][0] == "clinical-trials"
    assert state.has_document() and state.has_embeddings()
    assert state.serialise()["chunk_count"] == len(chunks)


class _FailingStagePlugin(StagePlugin):
    def __init__(self) -> None:
        super().__init__(
            StagePluginMetadata(
                name="failing",
                version="1.0.0",
                stage_types=("ingest",),
            )
        )

    def initialise(self, context: StagePluginContext) -> None:  # pragma: no cover - no-op
        pass

    def health_check(self, context: StagePluginContext) -> None:  # pragma: no cover - no-op
        pass

    def create_stage(self, definition: object, context: StagePluginContext) -> object:
        raise RuntimeError("boom")


class _DecliningStagePlugin(StagePlugin):
    def __init__(self) -> None:
        super().__init__(
            StagePluginMetadata(
                name="decline",
                version="1.0.0",
                stage_types=("ingest",),
            )
        )

    def initialise(self, context: StagePluginContext) -> None:  # pragma: no cover - no-op
        pass

    def health_check(self, context: StagePluginContext) -> None:  # pragma: no cover - no-op
        pass

    def create_stage(self, definition: object, context: StagePluginContext) -> object:
        return None


def test_stage_plugin_manager_retries_and_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = StagePluginManager(StagePluginContext(resources={}))
    manager.register(_FailingStagePlugin())
    manager.create_stage.retry.sleep = lambda _: None  # type: ignore[attr-defined]

    definition = SimpleNamespace(stage_type="ingest")
    with pytest.raises(StagePluginExecutionError):
        manager.create_stage(definition)


def test_stage_plugin_manager_reports_missing_stage_type() -> None:
    manager = StagePluginManager(StagePluginContext(resources={}))
    with pytest.raises(StagePluginNotAvailable):
        manager.create_stage(SimpleNamespace(stage_type="unknown"))


def test_stage_plugin_manager_reports_declined_stage() -> None:
    manager = StagePluginManager(StagePluginContext(resources={}))
    manager.register(_DecliningStagePlugin())
    with pytest.raises(StagePluginNotAvailable):
        manager.create_stage(SimpleNamespace(stage_type="ingest"))
