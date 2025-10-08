from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import importlib.util
import pathlib
import sys
import types

import pytest

_ORCH_PATH = pathlib.Path("src/Medical_KG_rev/orchestration").resolve()

import dagster

_RESULTS_MODULE = types.ModuleType("dagster._core.execution.results")
_RESULTS_MODULE.ExecuteInProcessResult = dagster.ExecuteInProcessResult
sys.modules["dagster._core.execution.results"] = _RESULTS_MODULE


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_CONFIG_MODULE = _load_module(
    "dagster_configuration", _ORCH_PATH / "dagster" / "configuration.py"
)
_STAGES_MODULE = _load_module(
    "stage_contracts", _ORCH_PATH / "stages" / "contracts.py"
)
orch_pkg = types.ModuleType("Medical_KG_rev.orchestration")
orch_pkg.__path__ = [_ORCH_PATH]
sys.modules["Medical_KG_rev.orchestration"] = orch_pkg
dagster_pkg = types.ModuleType("Medical_KG_rev.orchestration.dagster")
dagster_pkg.__path__ = [_ORCH_PATH / "dagster"]
dagster_pkg.configuration = _CONFIG_MODULE
sys.modules["Medical_KG_rev.orchestration.dagster"] = dagster_pkg
sys.modules["Medical_KG_rev.orchestration.dagster.configuration"] = _CONFIG_MODULE
stages_pkg = types.ModuleType("Medical_KG_rev.orchestration.stages")
stages_pkg.__path__ = [_ORCH_PATH / "stages"]
stages_pkg.contracts = _STAGES_MODULE
sys.modules["Medical_KG_rev.orchestration.stages"] = stages_pkg
sys.modules["Medical_KG_rev.orchestration.stages.contracts"] = _STAGES_MODULE
_RUNTIME_MODULE = _load_module("dagster_runtime", _ORCH_PATH / "dagster" / "runtime.py")
from Medical_KG_rev.models.entities import Claim, Entity, ExtractionActivity
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section, Span
from Medical_KG_rev.models.provenance import DataSource
from Medical_KG_rev.orchestration.events import StageEventEmitter
from Medical_KG_rev.orchestration.kafka import KafkaClient
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.openlineage import OpenLineageEmitter

PipelineConfigLoader = _CONFIG_MODULE.PipelineConfigLoader
ResiliencePolicyLoader = _CONFIG_MODULE.ResiliencePolicyLoader
StageFactory = _RUNTIME_MODULE.StageFactory
DagsterOrchestrator = _RUNTIME_MODULE.DagsterOrchestrator
dagster_runtime = _RUNTIME_MODULE
EmbeddingBatch = _STAGES_MODULE.EmbeddingBatch
EmbeddingVector = _STAGES_MODULE.EmbeddingVector
GraphWriteReceipt = _STAGES_MODULE.GraphWriteReceipt
IndexReceipt = _STAGES_MODULE.IndexReceipt
StageContext = _STAGES_MODULE.StageContext


class StubAdapterRequest:
    """Lightweight request model compatible with the Dagster runtime."""

    def __init__(
        self,
        tenant_id: str,
        correlation_id: str,
        domain: str,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.correlation_id = correlation_id
        self.domain = domain
        self.parameters = dict(parameters or {})

    def model_dump(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
            "domain": self.domain,
            "parameters": self.parameters,
        }

    @classmethod
    def model_validate(cls, payload: Mapping[str, object]) -> "StubAdapterRequest":
        return cls(
            tenant_id=str(payload["tenant_id"]),
            correlation_id=str(payload["correlation_id"]),
            domain=str(payload.get("domain", "biomedical")),
            parameters=payload.get("parameters", {}),
        )


dagster_runtime.AdapterRequest = StubAdapterRequest


class StubChunk:
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        tenant_id: str,
        body: str,
        title_path: tuple[str, ...],
        section: str,
    ) -> None:
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.tenant_id = tenant_id
        self.body = body
        self.title_path = title_path
        self.section = section



class _BaseStage:
    def __init__(self, stage_type: str, log: list[str]):
        self.stage_type = stage_type
        self.calls: list[Mapping[str, object]] = []
        self._log = log

    def record(self, **payload: object) -> None:
        payload.setdefault("stage", self.stage_type)
        self.calls.append(payload)
        self._log.append(self.stage_type)


class StubIngestStage(_BaseStage):
    def __init__(self, log: list[str]) -> None:
        super().__init__("ingest", log)

    def execute(self, ctx: StageContext, request: StubAdapterRequest) -> list[dict[str, object]]:
        self.record(tenant=ctx.tenant_id, adapter=request.parameters.get("adapter"))
        doc_id = ctx.doc_id or "doc-001"
        return [
            {
                "doc_id": doc_id,
                "title": "Example clinical trial",
                "body": "Population: adults Intervention: treatment Outcome: success",
            }
        ]


class StubParseStage(_BaseStage):
    def __init__(self, log: list[str], stage_type: str = "parse") -> None:
        super().__init__(stage_type, log)

    def execute(self, ctx: StageContext, payloads: list[dict[str, object]]) -> Document:
        self.record(payload_count=len(payloads))
        payload = payloads[0]
        block = Block(
            id="b1",
            type=BlockType.PARAGRAPH,
            text=payload["body"],
            spans=[Span(start=0, end=len(payload["body"]))],
        )
        section = Section(id="s1", title="Body", blocks=[block])
        return Document(
            id=payload["doc_id"],
            source="clinical-trials",
            title=payload["title"],
            sections=[section],
        )


class StubChunkStage(_BaseStage):
    def __init__(self, log: list[str]) -> None:
        super().__init__("chunk", log)
        self.fail_once = False
        self._failures = 0

    def execute(self, ctx: StageContext, document: Document) -> list[StubChunk]:
        self.record(document=document.id)
        if self.fail_once and self._failures == 0:
            self._failures += 1
            raise RuntimeError("chunk transient failure")
        chunk = StubChunk(
            chunk_id=f"{document.id}:chunk:0",
            doc_id=document.id,
            tenant_id=ctx.tenant_id,
            body=document.sections[0].blocks[0].text or "",
            title_path=(document.title or "Untitled",),
            section=document.sections[0].id,
        )
        return [chunk]


class StubEmbedStage(_BaseStage):
    def __init__(self, log: list[str]) -> None:
        super().__init__("embed", log)

    def execute(self, ctx: StageContext, chunks: list[Chunk]) -> EmbeddingBatch:
        self.record(chunks=len(chunks))
        vector = EmbeddingVector(id=chunks[0].chunk_id, values=(0.1, 0.2, 0.3))
        return EmbeddingBatch(vectors=(vector,), model="stub-model", tenant_id=ctx.tenant_id)


class StubIndexStage(_BaseStage):
    def __init__(self, log: list[str]) -> None:
        super().__init__("index", log)

    def execute(self, ctx: StageContext, batch: EmbeddingBatch) -> IndexReceipt:
        self.record(vectors=len(batch.vectors))
        return IndexReceipt(
            chunks_indexed=len(batch.vectors),
            opensearch_ok=True,
            faiss_ok=True,
            metadata={"tenant": ctx.tenant_id},
        )


class StubExtractStage(_BaseStage):
    def __init__(self, log: list[str]) -> None:
        super().__init__("extract", log)

    def execute(self, ctx: StageContext, document: Document) -> tuple[list[Entity], list[Claim]]:
        self.record(document=document.id)
        activity = ExtractionActivity(
            id="activity-1",
            actor="stub",
            data_source=DataSource(id="clinical-trials", name="ClinicalTrials.gov"),
            performed_at=datetime.now(UTC),
        )
        entity = Entity(
            id="entity-1",
            type="Condition",
            canonical_name="Hypertension",
            aliases=["High blood pressure"],
            spans=(Span(start=0, end=10),),
            metadata={"source": "stub"},
        )
        claim = Claim(
            id="claim-1",
            subject_id=entity.id,
            predicate="treats",
            object_id="drug-1",
            extraction=activity,
        )
        return [entity], [claim]


class StubKGStage(_BaseStage):
    def __init__(self, log: list[str]) -> None:
        super().__init__("knowledge-graph", log)

    def execute(
        self,
        ctx: StageContext,
        entities: list[Entity],
        claims: list[Claim],
    ) -> GraphWriteReceipt:
        self.record(entities=len(entities), claims=len(claims))
        return GraphWriteReceipt(
            nodes_written=len(entities),
            edges_written=len(claims),
            correlation_id=ctx.correlation_id or "corr-1",
        )


@pytest.fixture()
def orchestrator() -> DagsterOrchestrator:
    execution_log: list[str] = []

    ingest = StubIngestStage(execution_log)
    parse = StubParseStage(execution_log)
    validate = StubParseStage(execution_log, "ir-validation")
    chunk = StubChunkStage(execution_log)
    embed = StubEmbedStage(execution_log)
    index = StubIndexStage(execution_log)
    extract = StubExtractStage(execution_log)
    kg = StubKGStage(execution_log)

    stage_factory = StageFactory(
        {
            "ingest": lambda stage: ingest,
            "parse": lambda stage: parse,
            "ir-validation": lambda stage: validate,
            "chunk": lambda stage: chunk,
            "embed": lambda stage: embed,
            "index": lambda stage: index,
            "extract": lambda stage: extract,
            "knowledge-graph": lambda stage: kg,
        }
    )

    loader = PipelineConfigLoader("config/orchestration/pipelines")
    policies = ResiliencePolicyLoader("config/orchestration/resilience.yaml")
    policies.load()

    resilience_calls: list[str] = []
    original_apply = policies.apply

    def _tracking_apply(self, name: str, stage: str, func, *, hooks=None):  # type: ignore[override]
        wrapped = original_apply(name, stage, func, hooks=hooks)

        def _instrumented(*args, **kwargs):
            resilience_calls.append(stage)
            return wrapped(*args, **kwargs)

        return _instrumented

    policies.apply = _tracking_apply.__get__(policies, ResiliencePolicyLoader)  # type: ignore[attr-defined]
    ledger = JobLedger()
    kafka = KafkaClient()
    emitter = StageEventEmitter(kafka)
    lineage = OpenLineageEmitter(enabled=True)

    orchestrator = DagsterOrchestrator(
        loader,
        policies,
        stage_factory,
        openlineage_emitter=lineage,
        job_ledger=ledger,
        kafka_client=kafka,
        event_emitter=emitter,
    )

    # Attach stubs for introspection in the test
    orchestrator._stub_stages = {
        "ingest": ingest,
        "parse": parse,
        "ir-validation": validate,
        "chunk": chunk,
        "embed": embed,
        "index": index,
        "extract": extract,
        "knowledge-graph": kg,
    }
    orchestrator._execution_log = execution_log
    orchestrator._resilience_calls = resilience_calls
    orchestrator._kafka = kafka
    orchestrator._openlineage = lineage
    yield orchestrator
    policies.close()


def test_auto_pipeline_executes_all_stages(orchestrator: DagsterOrchestrator) -> None:
    context = StageContext(
        tenant_id="tenant-a",
        job_id="job-auto-1",
        doc_id="doc-001",
        correlation_id="corr-123",
        metadata={"source": "clinical-trials"},
    )
    request = StubAdapterRequest(
        tenant_id="tenant-a",
        correlation_id="corr-123",
        domain="biomedical",
        parameters={"adapter": "auto"},
    )

    orchestrator.job_ledger.create(
        job_id="job-auto-1",
        doc_key="doc-001",
        tenant_id="tenant-a",
        pipeline="auto",
        metadata={},
    )

    result = orchestrator.submit(
        pipeline="auto",
        context=context,
        adapter_request=request,
        payload={"seed": "value"},
    )

    assert result.success
    state = result.state

    assert state["payloads"]
    assert state["document"].id == "doc-001"
    assert state["chunks"][0].chunk_id.endswith(":chunk:0")
    assert state["embedding_batch"].model == "stub-model"
    assert state["index_receipt"].chunks_indexed == 1
    assert len(state["entities"]) == 1
    assert state["graph_receipt"].nodes_written == 1

    call_order = list(getattr(orchestrator, "_execution_log", ()))
    assert call_order[:3] == ["ingest", "parse", "ir-validation"]
    assert set(getattr(orchestrator, "_resilience_calls", ())) == {
        "ingest",
        "parse",
        "ir_validation",
        "chunk",
        "embed",
        "index",
        "extract",
        "kg",
    }

    entry = orchestrator.job_ledger.get("job-auto-1")
    assert entry is not None
    assert entry.retry_count_per_stage.get("chunk", 0) == 0
    assert entry.metadata["stage.ingest.output_count"] == 1
    assert entry.metadata["stage.kg.output_count"] == 1

    messages = list(orchestrator.kafka_client.consume("orchestration.events.v1"))
    event_types = [message.value["attributes"]["type"] for message in messages]
    assert event_types.count("stage.started") == len(call_order)
    assert event_types.count("stage.completed") == len(call_order)
    assert "stage.retrying" not in event_types


def test_stage_retry_emits_events(orchestrator: DagsterOrchestrator) -> None:
    chunk_stage = orchestrator._stub_stages["chunk"]
    chunk_stage.fail_once = True
    chunk_stage._failures = 0

    orchestrator.job_ledger.create(
        job_id="job-retry-1",
        doc_key="doc-retry",
        tenant_id="tenant-a",
        pipeline="auto",
        metadata={},
    )

    context = StageContext(
        tenant_id="tenant-a",
        job_id="job-retry-1",
        doc_id="doc-retry",
        correlation_id="corr-retry",
        metadata={"source": "clinical-trials"},
    )
    request = StubAdapterRequest(
        tenant_id="tenant-a",
        correlation_id="corr-retry",
        domain="biomedical",
        parameters={"adapter": "auto"},
    )

    result = orchestrator.submit(
        pipeline="auto",
        context=context,
        adapter_request=request,
        payload={"seed": "retry"},
    )

    assert result.success
    entry = orchestrator.job_ledger.get("job-retry-1")
    assert entry is not None
    assert entry.retry_count_per_stage.get("chunk") == 1

    messages = list(orchestrator.kafka_client.consume("orchestration.events.v1"))
    types = [message.value["attributes"]["type"] for message in messages]
    assert "stage.retrying" in types
    stage_count = len(orchestrator._stub_stages)
    assert types.count("stage.started") == stage_count
    assert types.count("stage.completed") == stage_count

    chunk_stage.fail_once = False
    chunk_stage._failures = 0


def test_openlineage_events_capture(orchestrator: DagsterOrchestrator) -> None:
    orchestrator.openlineage.clear()
    orchestrator.job_ledger.create(
        job_id="job-lineage-1",
        doc_key="doc-lineage",
        tenant_id="tenant-b",
        pipeline="auto",
        metadata={},
    )

    context = StageContext(
        tenant_id="tenant-b",
        job_id="job-lineage-1",
        doc_id="doc-lineage",
        correlation_id="corr-lineage",
        metadata={
            "source": "clinical-trials",
            "metrics": {"gpu_memory_mb": 2048, "gpu_utilization_percent": 72.5},
            "models": {"embed": {"name": "qwen", "version": "1.5"}},
        },
    )
    request = StubAdapterRequest(
        tenant_id="tenant-b",
        correlation_id="corr-lineage",
        domain="biomedical",
        parameters={"adapter": "auto"},
    )

    orchestrator.submit(
        pipeline="auto",
        context=context,
        adapter_request=request,
        payload={"seed": "lineage"},
    )

    events = orchestrator.openlineage.events
    assert events, "expected OpenLineage events to be emitted"
    assert events[0]["eventType"] == "START"
    assert events[-1]["eventType"] == "COMPLETE"
    run_facets = events[-1]["run"]["facets"]
    assert "retryAttempts" in run_facets
    job_facets = events[-1]["jobFacets"]
    assert "gpuUtilization" in job_facets
    assert "modelVersion" in job_facets
