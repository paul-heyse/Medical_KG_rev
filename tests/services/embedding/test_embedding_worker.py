from Medical_KG_rev.orchestration.stages.contracts import (
    EmbeddingBatch,
    StageContext,
)
from Medical_KG_rev.orchestration.stages.contracts import (
    EmbeddingVector as StageEmbeddingVector,
)
from Medical_KG_rev.services.embedding.service import (
    EmbeddingRequest,
    EmbeddingWorker,
)


class _StubEmbedStage:
    def __init__(self) -> None:
        self.calls: list[tuple[StageContext, list[object]]] = []

    def execute(self, ctx: StageContext, chunks):
        self.calls.append((ctx, list(chunks)))
        vectors = [
            StageEmbeddingVector(
                id=chunk.chunk_id,
                values=(1.0, 2.0, 3.0),
                metadata={"chunk_id": chunk.chunk_id},
            )
            for chunk in chunks
        ]
        return EmbeddingBatch(vectors=tuple(vectors), model="stub", tenant_id=ctx.tenant_id)


def test_worker_runs_embed_stage_and_normalizes() -> None:
    stage = _StubEmbedStage()
    worker = EmbeddingWorker(embed_stage=stage)
    request = EmbeddingRequest(tenant_id="tenant", texts=["Hello world"], normalize=True)
    response = worker.run(request)
    assert len(response.vectors) == 1
    vector = response.vectors[0]
    assert vector.model == "stub"
    assert vector.dimension == 3
    magnitude = sum(value * value for value in vector.values) ** 0.5
    assert magnitude == 1.0
    ctx, chunks = stage.calls[0]
    assert ctx.tenant_id == "tenant"
    assert chunks[0].tenant_id == "tenant"


def test_worker_generates_chunk_ids_when_missing() -> None:
    stage = _StubEmbedStage()
    worker = EmbeddingWorker(embed_stage=stage)
    request = EmbeddingRequest(tenant_id="tenant", texts=["one", "two"], normalize=False)
    response = worker.run(request)
    ids = [vector.id for vector in response.vectors]
    assert len(ids) == 2
    assert all(id_.startswith("tenant:") for id_ in ids)
