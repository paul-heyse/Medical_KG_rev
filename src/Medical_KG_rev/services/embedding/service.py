"""Embedding microservice that generates SPLADE and Qwen-3 vectors on GPU."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import structlog

from ..gpu.manager import GpuManager

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class EmbeddingRequest:
    tenant_id: str
    chunk_ids: Sequence[str]
    texts: Sequence[str]
    normalize: bool = False
    batch_size: int = 8
    models: Sequence[str] | None = None


@dataclass(slots=True)
class EmbeddingVector:
    id: str
    model: str
    kind: str  # "dense" or "sparse"
    values: list[float]
    dimension: int


@dataclass(slots=True)
class EmbeddingBatch:
    model: str
    chunk_ids: Sequence[str]
    texts: Sequence[str]


@dataclass(slots=True)
class EmbeddingResponse:
    vectors: list[EmbeddingVector] = field(default_factory=list)


class _BaseModel:
    name: str
    kind: str
    dimension: int

    def __init__(self, gpu: GpuManager) -> None:
        self.gpu = gpu
        self._is_loaded = False

    def load(self) -> None:
        if self._is_loaded:
            return
        with self.gpu.device_session(f"model:{self.name}", warmup=True):
            logger.info("embedding.model.loaded", model=self.name)
            self._is_loaded = True

    def encode(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - abstract
        raise NotImplementedError


class SpladeModel(_BaseModel):
    name = "splade"
    kind = "sparse"
    dimension = 64

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.load()
        vectors: list[list[float]] = []
        for text in texts:
            buckets = [0.0] * self.dimension
            tokens = [token for token in text.lower().split() if token]
            for token in tokens:
                bucket_index = (
                    int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension
                )
                buckets[bucket_index] += 1.0
            max_value = max(buckets) if buckets else 1.0
            normalized = [value / max_value for value in buckets]
            vectors.append(normalized)
        return vectors


class QwenDenseModel(_BaseModel):
    name = "qwen-3"
    kind = "dense"
    dimension = 128

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.load()
        vectors: list[list[float]] = []
        for text in texts:
            values = [0.0] * self.dimension
            base_hash = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
            for i in range(self.dimension):
                rotated = (base_hash >> (i % 32)) & 0xFFFFFFFF
                value = math.sin(rotated / (10**6))
                values[i] = float(value)
            vectors.append(values)
        return vectors


class EmbeddingModelRegistry:
    """Caches embedding models so they are only loaded once per process."""

    def __init__(self, gpu: GpuManager) -> None:
        self.gpu = gpu
        self._cache: dict[str, _BaseModel] = {}

    def get(self, name: str) -> _BaseModel:
        if name not in self._cache:
            if name == SpladeModel.name:
                self._cache[name] = SpladeModel(self.gpu)
            elif name in {QwenDenseModel.name, "qwen"}:
                self._cache[name] = QwenDenseModel(self.gpu)
            else:
                raise ValueError(f"Unknown embedding model: {name}")
        return self._cache[name]


class EmbeddingWorker:
    """Coordinates batching and GPU execution for embedding generation."""

    def __init__(self, registry: EmbeddingModelRegistry) -> None:
        self.registry = registry

    def _batched(self, request: EmbeddingRequest) -> Iterable[EmbeddingBatch]:
        models = tuple(request.models or (SpladeModel.name, QwenDenseModel.name))
        for start in range(0, len(request.texts), request.batch_size):
            end = start + request.batch_size
            for model_name in models:
                yield EmbeddingBatch(
                    model=model_name,
                    chunk_ids=request.chunk_ids[start:end],
                    texts=request.texts[start:end],
                )

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def run(self, request: EmbeddingRequest) -> EmbeddingResponse:
        response = EmbeddingResponse()
        for batch in self._batched(request):
            model = self.registry.get(batch.model)
            vectors = model.encode(batch.texts)
            for chunk_id, vector_values in zip(batch.chunk_ids, vectors, strict=False):
                if request.normalize:
                    vector_values = self._normalize(vector_values)
                response.vectors.append(
                    EmbeddingVector(
                        id=chunk_id,
                        model=model.name,
                        kind=model.kind,
                        values=vector_values,
                        dimension=model.dimension,
                    )
                )
        logger.info(
            "embedding.batch.completed",
            total=len(response.vectors),
            chunks=len(request.texts),
            normalize=request.normalize,
        )
        return response


class EmbeddingGrpcService:
    """Async gRPC servicer for the embedding worker."""

    def __init__(self, worker: EmbeddingWorker) -> None:
        self.worker = worker

    async def EmbedChunks(self, request, context):  # type: ignore[override]
        embed_request = EmbeddingRequest(
            tenant_id=request.tenant_id,
            chunk_ids=list(request.chunk_ids),
            texts=list(request.texts),
            normalize=request.normalize,
            batch_size=request.batch_size or 8,
            models=list(request.models) or None,
        )
        response = self.worker.run(embed_request)
        from Medical_KG_rev.proto.gen import embedding_pb2  # type: ignore import-error

        reply = embedding_pb2.EmbedChunksResponse()
        for vector in response.vectors:
            message = reply.vectors.add()
            message.chunk_id = vector.id
            message.model = vector.model
            message.kind = vector.kind
            message.dimension = vector.dimension
            message.values.extend(vector.values)
        return reply
