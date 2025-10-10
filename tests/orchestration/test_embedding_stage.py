from dataclasses import dataclass

import pytest

pytest.importorskip("yaml")

from Medical_KG_rev.orchestration.ingestion_pipeline import EmbeddingStage
from Medical_KG_rev.orchestration.pipeline import PipelineContext
from Medical_KG_rev.orchestration.stages import StageFailure
from Medical_KG_rev.services import GpuNotAvailableError
from Medical_KG_rev.services.embedding.service import EmbeddingResponse, EmbeddingVector


@dataclass
class StubWorker:
    responses: list[EmbeddingResponse]
    calls: list[dict]

    def run(self, request):  # type: ignore[override]
        self.calls.append(
            {
                "texts": list(request.texts),
                "namespaces": request.namespaces,
                "models": request.models,
            }
        )
        if not self.responses:
            raise RuntimeError("No response configured")
        return self.responses.pop(0)

    def encode_queries(self, request):  # pragma: no cover - not used in tests
        return self.run(request)


def _vector(namespace: str, text: str) -> EmbeddingVector:
    return EmbeddingVector(
        id=f"chunk-{text}",
        model="model",
        namespace=namespace,
        kind="single_vector",
        vectors=[[1.0, 0.0]],
        dimension=2,
        metadata={"storage_target": "faiss", "provider": "test"},
    )


def _context(chunks: list[dict]) -> PipelineContext:
    return PipelineContext(tenant_id="tenant", operation="ingest", data={"chunks": chunks})


def test_embedding_stage_populates_context(monkeypatch: pytest.MonkeyPatch) -> None:
    response = EmbeddingResponse(
        vectors=[_vector("single_vector.ns", "1"), _vector("single_vector.ns", "2")]
    )
    worker = StubWorker(responses=[response], calls=[])
    stage = EmbeddingStage(worker=worker, namespaces=["single_vector.ns"])
    context = stage.execute(
        _context(
            [
                {"chunk_id": "1", "body": "alpha"},
                {"chunk_id": "2", "body": "beta"},
            ]
        )
    )
    assert worker.calls[0]["texts"] == ["alpha", "beta"]
    assert len(context.data["embeddings"]) == 2
    assert context.data["metrics"]["embedding"]["vectors"] == 2


def test_embedding_stage_raises_on_gpu_error() -> None:
    worker = StubWorker(responses=[], calls=[])

    def failing_run(request):  # type: ignore[override]
        raise GpuNotAvailableError("GPU unavailable")

    worker.run = failing_run  # type: ignore[assignment]
    stage = EmbeddingStage(worker=worker)
    with pytest.raises(StageFailure) as exc:
        stage.execute(_context([{"chunk_id": "1", "body": "text"}]))
    assert exc.value.error_type == "gpu_unavailable"


def test_embedding_stage_records_namespace_rollup(monkeypatch: pytest.MonkeyPatch) -> None:
    vectors = [
        _vector("single_vector.ns", "1"),
        _vector("single_vector.ns", "2"),
        _vector("single_vector.other", "3"),
    ]
    worker = StubWorker(responses=[EmbeddingResponse(vectors=vectors)], calls=[])
    stage = EmbeddingStage(worker=worker)
    context = stage.execute(
        _context(
            [
                {"chunk_id": "1", "body": "a"},
                {"chunk_id": "2", "body": "b"},
                {"chunk_id": "3", "body": "c"},
            ]
        )
    )
    summary = context.data["embedding_summary"]
    assert summary["vectors"] == 3
    assert set(summary["per_namespace"].keys()) == {"single_vector.ns", "single_vector.other"}


def test_embedding_stage_passes_models_to_worker() -> None:
    response = EmbeddingResponse(vectors=[_vector("single_vector.ns", "1")])
    worker = StubWorker(responses=[response], calls=[])
    stage = EmbeddingStage(worker=worker, models=["dense-model"])
    stage.execute(_context([{"chunk_id": "1", "body": "text"}]))
    assert worker.calls[0]["models"] == ["dense-model"]


def test_embedding_stage_requires_chunks() -> None:
    worker = StubWorker(responses=[], calls=[])
    stage = EmbeddingStage(worker=worker)
    with pytest.raises(StageFailure) as exc:
        stage.execute(PipelineContext(tenant_id="tenant", operation="ingest", data={"chunks": []}))
    assert exc.value.error_type == "validation"
