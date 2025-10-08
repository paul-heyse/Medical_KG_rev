from __future__ import annotations

import base64
import orjson
import zlib
from typing import Iterable

import pytest
from Medical_KG_rev.orchestration.stages import contracts as state_contracts

pytest.importorskip("pydantic")

from Medical_KG_rev.adapters.plugins.models import AdapterDomain, AdapterRequest
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.orchestration.stages.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
    DownloadArtifact,
    GateDecision,
    GraphWriteReceipt,
    IndexReceipt,
    PipelineState,
    PipelineStateSnapshot,
    PipelineStateValidationError,
    StageContext,
    StageResultSnapshot,
)

AdapterRequest = state_contracts.AdapterRequest
Chunk = state_contracts.Chunk
Document = state_contracts.Document


def _build_document() -> Document:
    return Document(id="doc-1", source="test", sections=(), metadata={})


def _sample_state(payload: dict[str, object] | None = None) -> PipelineState:
    ctx = StageContext(tenant_id="tenant", correlation_id="corr", pipeline_name="unit")
    request = AdapterRequest(
        tenant_id="tenant",
        correlation_id="corr",
        domain="biomedical",
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

    artifacts = [
        DownloadArtifact(document_id="doc-1", tenant_id="tenant", uri="s3://bucket/a.pdf")
    ]
    state.apply_stage_output("download", "download", artifacts)
    assert state.require_downloads()[0].uri.endswith("a.pdf")

    gate_decision = GateDecision(name="pdf_gate", ready=True)
    state.apply_stage_output("gate", "pdf_gate", gate_decision)
    assert state.get_gate_decision("pdf_gate").ready is True

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
    state.apply_stage_output(
        "download",
        "download",
        [{"asset_id": "pdf-1", "uri": "s3://bucket/file.pdf"}],
    )
    state.apply_stage_output("gate", "pdf_gate", True)
    assert state.has_pdf_assets()
    assert state.is_pdf_ready is True

    snapshot = state.serialise()
    assert snapshot["payload_count"] == 1
    assert snapshot["chunk_count"] == 1
    assert snapshot["embedding_count"] == 1
    assert snapshot["download_count"] == 1
    assert snapshot["gate_status"]["pdf_gate"] is True
    assert snapshot["stage_results"]["index"]["output_count"] == 1
    assert snapshot["pdf_asset_count"] == 1
    assert snapshot["gate_status"]["pdf_gate"] is True


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
    state.apply_stage_output(
        "download",
        "download",
        [{"asset_id": "pdf-1", "uri": "s3://bucket/file.pdf"}],
    )
    state.apply_stage_output("gate", "pdf_gate", False)
    payload = state.serialise()
    encoded = base64.b64encode(zlib.compress(state_contracts.orjson.dumps(payload))).decode("ascii")
    encoded = base64.b64encode(zlib.compress(orjson.dumps(payload))).decode("ascii")

    recovered = PipelineState.recover(
        encoded,
        context=state.context,
        adapter_request=state.adapter_request,
    )
    assert recovered.schema_version == payload["version"]
    assert recovered.metadata == state.metadata
    assert "ingest" in recovered.stage_results
    assert recovered.has_pdf_assets()
    assert recovered.gate_status["pdf_gate"] is False
    assert recovered.serialise()["pdf"] == payload["pdf"]
    assert recovered.has_pdf_assets()
    assert recovered.gate_status["pdf_gate"] is False


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
    mutated.apply_stage_output(
        "download",
        "download",
        [
            DownloadArtifact(
                document_id="d1",
                tenant_id="tenant",
                uri="s3://bucket/doc.pdf",
            )
        ],
    )

    diff = mutated.diff(baseline)
    assert diff["payload_count"] == (1, 0)
    assert diff["chunk_count"] == (1, 0)
    assert diff["download_count"] == (1, 0)
    assert "gate_status" not in diff

    mutated.record_gate_decision(GateDecision(name="pdf_gate", ready=False))
    diff = mutated.diff(baseline)
    assert diff["gate_status"] == ({"pdf_gate": False}, {})
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


def test_pdf_gate_dependencies_and_serialisation() -> None:
    state = _sample_state()
    with pytest.raises(PipelineStateValidationError):
        state.validate_transition("pdf-gate")

    state.apply_stage_output("pdf-download", "pdf-download", ["asset-1", "asset-2"])
    state.validate_transition("pdf-gate")

    class _GateDecision:
        allowed = True
        reason = "manual-override"
        ledger_reference = "ledger-123"

    state.apply_stage_output("pdf-gate", "pdf-gate", _GateDecision())
    snapshot = state.serialise()
    assert snapshot["pdf"]["downloads"] == ["asset-1", "asset-2"]
    assert snapshot["pdf"]["gate_open"] is True
    assert snapshot["pdf"]["ledger_reference"] == "ledger-123"


def test_persist_with_retry_invokes_callback_until_success() -> None:
    state = _sample_state()
    attempts: list[int] = []

    def _persist(payload: dict[str, object]) -> None:
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("transient failure")
        assert "pdf" in payload

    state.persist_with_retry(_persist)
    assert len(attempts) == 2


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


def test_persist_with_retry_retries_on_failure() -> None:
    state = _sample_state({"foo": "bar"})
    attempts = {"count": 0}

    def writer(data: bytes) -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("transient failure")
        return "ok"

    result = state.persist_with_retry(writer)
    assert result == "ok"
    assert attempts["count"] == 2


def test_validate_transition_requires_pdf_assets_for_gate() -> None:
    state = _sample_state()
    with pytest.raises(PipelineStateValidationError):
        state.validate_transition("gate")
    state.apply_stage_output("parse", "parse", _build_document())
    state.apply_stage_output(
        "download",
        "download",
        [{"asset_id": "pdf-1", "uri": "s3://bucket/file.pdf"}],
    )
    state.validate_transition("gate")
