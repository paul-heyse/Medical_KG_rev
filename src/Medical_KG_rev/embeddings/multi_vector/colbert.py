"""ColBERT late interaction embedder using deterministic pseudo vectors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry


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
class ColBERTRagatouilleEmbedder:
    config: EmbedderConfig
    _dim: int = 0
    _max_tokens: int = 0
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        params = self.config.parameters
        self._dim = int(self.config.dim or params.get("dim", 128))
        self._max_tokens = int(params.get("max_doc_tokens", 180))
        self.name = self.config.name
        self.kind = self.config.kind

    def _records(self, request: EmbeddingRequest, texts: list[str]) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []
        ids = list(request.ids or [f"{request.namespace}:{index}" for index in range(len(texts))])
        for chunk_id, text in zip(ids, texts, strict=False):
            tokens = text.split()
            tokens = tokens[: self._max_tokens]
            vectors, positions = _token_vectors(tokens, self._dim)
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
                    metadata={
                        "provider": self.config.provider,
                        "token_positions": positions,
                    },
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
