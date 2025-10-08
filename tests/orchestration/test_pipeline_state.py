import pytest

from Medical_KG_rev.adapters.plugins.models import AdapterDomain, AdapterRequest
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.orchestration.stages.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
    GraphWriteReceipt,
    IndexReceipt,
    PipelineState,
    StageContext,
)


def _build_document() -> Document:
    block = Block(id="b1", type=BlockType.PARAGRAPH, text="Hello world")
    section = Section(id="s1", title="Section", blocks=(block,))
    return Document(id="doc-1", source="test", sections=(section,), metadata={})


def _sample_state() -> PipelineState:
    ctx = StageContext(tenant_id="tenant", correlation_id="corr", pipeline_name="unit")
    request = AdapterRequest(
        tenant_id="tenant",
        correlation_id="corr",
        domain=AdapterDomain.BIOMEDICAL,
        parameters={},
    )
    return PipelineState.initialise(context=ctx, adapter_request=request, payload={"foo": "bar"})


def test_pipeline_state_stage_flow_serialises() -> None:
    state = _sample_state()
    with pytest.raises(ValueError):
        state.ensure_ready_for("parse")

    payloads = [{"id": "p1"}]
    state.apply_stage_output("ingest", "ingest", payloads)
    assert state.require_payloads() == tuple(payloads)

    document = _build_document()
    state.apply_stage_output("parse", "parse", document)
    assert state.has_document()
    state.ensure_ready_for("chunk")

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


def test_pipeline_state_recover_handles_partial_payload() -> None:
    state = _sample_state()
    recovered = PipelineState.recover(
        {"version": "v2", "metadata": {"note": "ok"}},
        context=state.context,
        adapter_request=state.adapter_request,
    )
    assert recovered.schema_version == "v2"
    assert recovered.metadata["note"] == "ok"
