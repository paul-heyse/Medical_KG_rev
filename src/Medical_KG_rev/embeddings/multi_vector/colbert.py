"""ColBERT late interaction embedder using deterministic pseudo vectors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import structlog

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry


logger = structlog.get_logger(__name__)


def _token_vectors(tokens: list[str], dim: int) -> tuple[list[list[float]], list[int]]:
    vectors: list[list[float]] = []
    positions: list[int] = []
    for index, token in enumerate(tokens):
        digest = hashlib.sha1(f"{token}:{index}".encode("utf-8")).digest()
        repeats = (dim * 4 + len(digest) - 1) // len(digest)
        tiled = (digest * repeats)[: dim * 4]
        array = np.frombuffer(tiled, dtype=np.uint32)
        values = (array.astype(np.float64) / np.iinfo(np.uint32).max) * 2 - 1
        vectors.append(values.astype(float).tolist()[:dim])
        positions.append(index)
    return vectors, positions


@dataclass(slots=True)
class ColbertShard:
    name: str
    dimension: int
    capacity: int
    documents: dict[str, list[list[float]]] = field(default_factory=dict)

    def add(self, doc_id: str, vectors: list[list[float]]) -> None:
        if len(self.documents) >= self.capacity:
            # Remove the oldest document to keep the shard bounded.
            key, _value = next(iter(self.documents.items()))
            self.documents.pop(key, None)
        self.documents[doc_id] = vectors


@dataclass(slots=True)
class ColbertShardManager:
    shards: dict[str, ColbertShard] = field(default_factory=dict)

    def register(self, name: str, *, dimension: int, capacity: int = 1024) -> None:
        if name not in self.shards:
            self.shards[name] = ColbertShard(name=name, dimension=dimension, capacity=capacity)

    def assign(self, doc_id: str) -> ColbertShard:
        if not self.shards:
            raise RuntimeError("No shards registered for ColBERT index")
        shard_names = sorted(self.shards)
        index = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16) % len(shard_names)
        return self.shards[shard_names[index]]

    def store(self, doc_id: str, vectors: list[list[float]]) -> str:
        shard = self.assign(doc_id)
        shard.add(doc_id, vectors)
        return shard.name

    def get(self, name: str) -> ColbertShard | None:
        return self.shards.get(name)


def maxsim_score(query: Iterable[list[float]], document: Iterable[list[float]]) -> float:
    """Compute MaxSim score between query and document vectors."""

    doc_vectors = list(document)
    if not doc_vectors:
        return 0.0
    score = 0.0
    for q in query:
        best = max(
            sum(qi * di for qi, di in zip(q, d, strict=False))
            for d in doc_vectors
        )
        score += best
    return score


@dataclass(slots=True)
class QdrantMultiVectorAdapter:
    collection: str
    payloads: dict[str, Mapping[str, object]] = field(default_factory=dict)

    def upsert(self, doc_id: str, vectors: list[list[float]], metadata: Mapping[str, object]) -> None:
        self.payloads[doc_id] = {"vectors": vectors, **metadata}


@dataclass(slots=True)
class ColBERTRagatouilleEmbedder:
    config: EmbedderConfig
    _dim: int = 0
    _max_tokens: int = 0
    _shards: ColbertShardManager = field(default_factory=ColbertShardManager)
    _qdrant: QdrantMultiVectorAdapter | None = None
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        params = self.config.parameters
        self._dim = int(self.config.dim or params.get("dim", 128))
        self._max_tokens = int(params.get("max_doc_tokens", 180))
        shard_count = int(params.get("shards", 4))
        shard_capacity = int(params.get("shard_capacity", 2048))
        for index in range(shard_count):
            self._shards.register(
                f"shard-{index}", dimension=self._dim, capacity=shard_capacity
            )
        if "qdrant_collection" in params:
            self._qdrant = QdrantMultiVectorAdapter(collection=str(params["qdrant_collection"]))
        self.name = self.config.name
        self.kind = self.config.kind

    def _records(self, request: EmbeddingRequest, texts: list[str]) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []
        ids = list(request.ids or [f"{request.namespace}:{index}" for index in range(len(texts))])
        for chunk_id, text in zip(ids, texts, strict=False):
            tokens = text.split()
            tokens = tokens[: self._max_tokens]
            vectors, positions = _token_vectors(tokens, self._dim)
            shard_name = self._shards.store(chunk_id, vectors)
            metadata = {
                "provider": self.config.provider,
                "token_positions": positions,
                "shard": shard_name,
            }
            if self._qdrant is not None:
                metadata["qdrant_collection"] = self._qdrant.collection
                self._qdrant.upsert(chunk_id, vectors, {"positions": positions})
            logger.debug(
                "colbert.embedder.generated",
                chunk_id=chunk_id,
                shard=shard_name,
                tokens=len(tokens),
            )
            records.append(
                EmbeddingRecord(
                    id=chunk_id,
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model_id=self.config.model_id,
                    model_version=self.config.model_version,
                    kind=self.config.kind,
                    dim=self._dim,
                    vectors=vectors,
                    metadata=metadata,
                    correlation_id=request.correlation_id,
                )
            )
        return records

    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self._records(request, list(request.texts))

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self._records(request, list(request.texts))


def register_colbert(registry: EmbedderRegistry) -> None:
    registry.register("colbert", lambda config: ColBERTRagatouilleEmbedder(config=config))
