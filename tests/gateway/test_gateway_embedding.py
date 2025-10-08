from __future__ import annotations

from types import SimpleNamespace

import pytest

from Medical_KG_rev.gateway.models import EmbedRequest
from Medical_KG_rev.gateway.services import GatewayService
from Medical_KG_rev.gateway.sse.manager import EventStreamManager
from Medical_KG_rev.orchestration.dagster.runtime import StageFactory
from Medical_KG_rev.orchestration.stages.contracts import EmbeddingBatch, EmbeddingVector, StageContext
from Medical_KG_rev.services.retrieval.chunking import ChunkingService


class _StubLedger:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.processing: list[tuple[str, str]] = []
        self.metadata: list[tuple[str, dict[str, object]]] = []
        self.completed: list[tuple[str, dict[str, object]]] = []
        self.failed: list[tuple[str, str, str]] = []

    def create(self, **payload: object) -> None:
        self.created.append(payload)

    def mark_processing(self, job_id: str, *, stage: str) -> None:
        self.processing.append((job_id, stage))

    def mark_completed(self, job_id: str, *, metadata: dict[str, object] | None = None) -> None:
        self.completed.append((job_id, metadata or {}))

    def mark_failed(self, job_id: str, *, stage: str, reason: str) -> None:
        self.failed.append((job_id, stage, reason))

    def update_metadata(self, job_id: str, metadata: dict[str, object]) -> None:
        self.metadata.append((job_id, metadata))


class _StubChunkStage:
    def execute(self, ctx: StageContext, document) -> list[object]:  # noqa: ANN001 - unused
        return []


class _StubEmbedStage:
    def __init__(self) -> None:
        self.calls: list[tuple[StageContext, list[object]]] = []

    def execute(self, ctx: StageContext, chunks):  # noqa: ANN001 - interface contract
        self.calls.append((ctx, list(chunks)))
        vectors = [
            EmbeddingVector(id=chunk.chunk_id, values=(1.0, 2.0), metadata={"chunk_id": chunk.chunk_id})
            for chunk in chunks
        ]
        return EmbeddingBatch(vectors=tuple(vectors), model="stub-model", tenant_id=ctx.tenant_id)


@pytest.fixture()
def gateway_service() -> GatewayService:
    embed_stage = _StubEmbedStage()
    chunk_stage = _StubChunkStage()
    stage_factory = StageFactory({
        "chunk": lambda _: chunk_stage,
        "embed": lambda _: embed_stage,
    })
    service = GatewayService(
        events=EventStreamManager(),
        orchestrator=SimpleNamespace(),
        ledger=_StubLedger(),
        stage_factory=stage_factory,
        chunker=ChunkingService(stage_factory=stage_factory, chunk_stage=chunk_stage),
    )
    # attach stub for assertion access
    service._embed_stage_stub = embed_stage  # type: ignore[attr-defined]
    service._ledger_stub = service.ledger  # type: ignore[attr-defined]
    return service


def test_embed_normalizes_vectors(gateway_service: GatewayService) -> None:
    request = EmbedRequest(tenant_id="tenant", inputs=["hello world"], model="stub-model", normalize=True)
    embeddings = gateway_service.embed(request)
    assert len(embeddings) == 1
    vector = embeddings[0]
    expected_norm = 1.0
    actual_norm = sum(value * value for value in vector.vector) ** 0.5
    assert pytest.approx(expected_norm, rel=1e-6) == actual_norm
    embed_stage = gateway_service._embed_stage_stub  # type: ignore[attr-defined]
    ctx, chunks = embed_stage.calls[0]
    assert ctx.pipeline_name == gateway_service._PIPELINE_NAME
    assert len(chunks) == 1
    ledger: _StubLedger = gateway_service._ledger_stub  # type: ignore[attr-defined]
    assert ledger.metadata, "expected metadata update for embeddings"
