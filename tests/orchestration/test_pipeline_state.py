from __future__ import annotations

import base64
import json
import zlib
from typing import Iterable

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.adapters.plugins.models import AdapterDomain, AdapterRequest
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.orchestration.stages.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
    GraphWriteReceipt,
    IndexReceipt,
    PipelineState,
    PipelineStateSnapshot,
    PipelineStateValidationError,
    StageContext,
    StageResultSnapshot,
)


def _build_document() -> Document:
    block = Block(id="b1", type=BlockType.PARAGRAPH, text="Hello world")
    section = Section(id="s1", title="Section", blocks=(block,))
    return Document(id="doc-1", source="test", sections=(section,), metadata={})


def _sample_state(payload: dict[str, object] | None = None) -> PipelineState:
    ctx = StageContext(tenant_id="tenant", correlation_id="corr", pipeline_name="unit")
    request = AdapterRequest(
        tenant_id="tenant",
        correlation_id="corr",
        domain=AdapterDomain.BIOMEDICAL,
        parameters={},
    )
    return PipelineState.initialise(context=ctx, adapter_request=request, payload=payload or {})


@pytest.fixture(autouse=True)
def reset_validators() -> Iterable[None]:
    PipelineState.clear_validators()
    yield
    PipelineState.clear_validators()


def test_pipeline_state_stage_flow_serialises() -> None:
    state = _sample_state()
    with pytest.raises(PipelineStateValidationError):
        state.validate_transition("parse")

    payloads = [{"id": "p1"}]
    state.apply_stage_output("ingest", "ingest", payloads)
    assert state.require_payloads() == tuple(payloads)

    document = _build_document()
    state.apply_stage_output("parse", "parse", document)
    assert state.has_document()
    state.validate_transition("chunk")

    chunk = Chunk(
        chunk_id="c1",
        doc_id=document.id,
        tenant_id="tenant",
        body="chunk text",
        title_path=("Section",),
        section="s1",
        start_char=0,
        end_char=10,
        granularity="paragraph",
        chunker="stub",
        chunker_version="1",
    )
    state.apply_stage_output("chunk", "chunk", [chunk])
    assert state.require_chunks()[0].chunk_id == "c1"

    batch = EmbeddingBatch(
        vectors=(
            EmbeddingVector(id="c1", values=(0.1, 0.2), metadata={"chunk_id": "c1"}),
        ),
        model="stub",
        tenant_id="tenant",
    )
    state.apply_stage_output("embed", "embed", batch)
    assert state.has_embeddings()

    receipt = IndexReceipt(chunks_indexed=1, opensearch_ok=True, faiss_ok=True)
    state.apply_stage_output("index", "index", receipt)
    state.record_stage_metrics("index", stage_type="index", output_count=1)
    assert state.index_receipt == receipt

    state.apply_stage_output("extract", "extract", ([], []))
    state.apply_stage_output(
        "knowledge-graph",
        "kg",
        GraphWriteReceipt(nodes_written=1, edges_written=0, correlation_id="corr", metadata={}),
    )

    snapshot = state.serialise()
    assert snapshot["payload_count"] == 1
    assert snapshot["chunk_count"] == 1
    assert snapshot["embedding_count"] == 1
    assert snapshot["stage_results"]["index"]["output_count"] == 1


def test_serialise_caches_until_mutation() -> None:
    state = _sample_state()
    assert state.is_dirty() is True
    first = state.serialise()
    assert first["payload_count"] == 0
    assert state.is_dirty() is False
    second = state.serialise()
    assert second == first
    state.set_payloads([{"id": "p1"}])
    assert state.is_dirty() is True


def test_serialise_json_uses_cache() -> None:
    state = _sample_state()
    snapshot = state.serialise_json()
    assert snapshot
    cached = state.serialise_json()
    assert cached == snapshot


def test_pipeline_state_recover_handles_compressed_payload() -> None:
    state = _sample_state({"foo": "bar"})
    state.apply_stage_output("ingest", "ingest", [{"foo": 1}])
    state.record_stage_metrics("ingest", stage_type="ingest", attempts=1)
    payload = state.serialise()
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode("utf-8"))).decode("ascii")

    recovered = PipelineState.recover(
        encoded,
        context=state.context,
        adapter_request=state.adapter_request,
    )
    assert recovered.schema_version == payload["version"]
    assert recovered.metadata == state.metadata
    assert "ingest" in recovered.stage_results


def test_dependencies_require_completed_stage() -> None:
    state = _sample_state()
    state.stage_results["parse"] = StageResultSnapshot(stage="parse", stage_type="parse")
    with pytest.raises(ValueError):
        state.ensure_dependencies("chunk", ["ingest"])
    state.stage_results["ingest"] = StageResultSnapshot(stage="ingest", stage_type="ingest")
    assert state.dependencies_satisfied(["ingest"]) is True
    state.ensure_dependencies("chunk", ["ingest"])


def test_to_model_generates_valid_payload() -> None:
    state = _sample_state()
    model = state.to_model()
    assert model.context.tenant_id == "tenant"
    assert model.payload_count == 0


def test_diff_reports_changes_between_states() -> None:
    baseline = _sample_state()
    mutated = _sample_state()
    mutated.apply_stage_output("ingest", "ingest", [{"id": 1}])
    mutated.apply_stage_output(
        "chunk",
        "chunk",
        [
            Chunk(
                chunk_id="c1",
                doc_id="d1",
                tenant_id="tenant",
                body="text",
                title_path=("t",),
                section="s",
                start_char=0,
                end_char=1,
                granularity="sentence",
                chunker="stub",
                chunker_version="1",
            )
        ],
    )
    diff = mutated.diff(baseline)
    assert diff["payload_count"] == (1, 0)
    assert diff["chunk_count"] == (1, 0)
    assert "pipeline_version" not in diff


def test_pdf_gate_serialises_state() -> None:
    state = _sample_state()
    state.apply_stage_output("pdf-download", "download", {"url": "http://example"})
    assert state.pdf_gate.downloaded is True
    state.apply_stage_output("pdf-ir-gate", "gate", {"status": "ready"})
    assert state.pdf_gate.ir_ready is True
    payload = state.serialise()
    assert payload["pdf_gate"]["downloaded"] is True
    assert payload["pdf_gate"]["ir_ready"] is True


def test_custom_validation_rules_raise_errors() -> None:
    state = _sample_state()

    def _rule(current: PipelineState) -> None:
        if not current.payload:
            raise ValueError("payload missing")

    PipelineState.register_validator(_rule, name="payload-check")
    with pytest.raises(PipelineStateValidationError) as excinfo:
        state.validate()
    assert excinfo.value.rule == "payload-check"


def test_cleanup_stage_releases_outputs() -> None:
    state = _sample_state()
    state.apply_stage_output("ingest", "ingest", [{"id": 1}])
    state.cleanup_stage("ingest")
    assert not state.get_payloads()


def test_snapshot_and_restore_rollback_changes() -> None:
    state = _sample_state()
    state.apply_stage_output("ingest", "ingest", [{"id": "a"}])
    state.apply_stage_output("parse", "parse", _build_document())
    original = state.snapshot()

    state.set_payloads([{"id": "b"}])
    state.metadata["flag"] = True
    assert state.require_payloads()[0]["id"] == "b"

    state.restore(original)
    assert isinstance(original, PipelineStateSnapshot)
    assert state.require_payloads()[0]["id"] == "a"
    assert "flag" not in state.metadata


def test_tenant_scope_enforcement() -> None:
    state = _sample_state()
    state.ensure_tenant_scope("tenant")
    with pytest.raises(PipelineStateValidationError):
        state.ensure_tenant_scope("other")


def test_checkpoint_and_rollback_restore_prior_state() -> None:
    state = _sample_state()
    state.apply_stage_output("ingest", "ingest", [{"id": "one"}])
    state.create_checkpoint("before-parse")
    state.apply_stage_output("parse", "parse", _build_document())
    state.set_payloads([{"id": "two"}])

    state.rollback_to("before-parse")

    payloads = state.require_payloads()
    assert payloads[0]["id"] == "one"
    assert state.document is None

    state.clear_checkpoint("before-parse")
    assert state.get_checkpoint("before-parse") is None


def test_legacy_round_trip_preserves_core_fields() -> None:
    state = _sample_state({"foo": "bar"})
    state.apply_stage_output("ingest", "ingest", [{"id": "legacy"}])
    document = _build_document()
    state.apply_stage_output("parse", "parse", document)
    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id=document.id,
        tenant_id="tenant",
        body="hello",
        title_path=("Section",),
        section="s1",
        start_char=0,
        end_char=5,
        granularity="paragraph",
        chunker="stub",
        chunker_version="1",
    )
    state.apply_stage_output("chunk", "chunk", [chunk])
    batch = EmbeddingBatch(
        vectors=(EmbeddingVector(id="chunk-1", values=(0.1, 0.2), metadata={}),),
        model="stub",
        tenant_id="tenant",
    )
    state.apply_stage_output("embed", "embed", batch)

    legacy = state.to_legacy_dict()
    restored = PipelineState.from_legacy(
        legacy,
        context=StageContext.from_dict(state.context.to_dict()),
        adapter_request=state.adapter_request.model_copy(deep=True),
    )

    assert restored.require_payloads() == state.require_payloads()
    assert restored.document is not None
    assert restored.document.id == document.id
    assert restored.embedding_batch is not None
    assert restored.embedding_batch.model == batch.model
    assert restored.metadata == state.metadata
    assert restored.stage_results.keys() == state.stage_results.keys()
