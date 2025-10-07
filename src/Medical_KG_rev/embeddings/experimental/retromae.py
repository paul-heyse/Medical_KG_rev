"""RetroMAE experimental embedder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry
from ..utils.normalization import normalize_batch


@dataclass(slots=True)
class RetroMAEEmbedder:
    config: EmbedderConfig
    _dim: int = 0
    _normalize: bool = False
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        self._dim = int(self.config.dim or 768)
        self._normalize = bool(self.config.normalize)
        self.name = self.config.name
        self.kind = self.config.kind

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        repeats = (self._dim * 4 + len(digest) - 1) // len(digest)
        tiled = (digest * repeats)[: self._dim * 4]
        ints = [int.from_bytes(tiled[i : i + 4], "big") for i in range(0, len(tiled), 4)]
        scale = float(2**32)
        return [(value / scale) * 2 - 1 for value in ints][: self._dim]

    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        vectors = [self._vector(text) for text in request.texts]
        if self._normalize:
            vectors = normalize_batch(vectors)
        ids = list(request.ids or [f"{request.namespace}:{index}" for index in range(len(vectors))])
        records: list[EmbeddingRecord] = []
        for chunk_id, vector in zip(ids, vectors, strict=False):
            records.append(
                EmbeddingRecord(
                    id=chunk_id,
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model_id=self.config.model_id,
                    model_version=self.config.model_version,
                    kind=self.config.kind,
                    dim=len(vector),
                    vectors=[vector],
                    normalized=self._normalize,
                    metadata={"provider": self.config.provider, "experimental": True},
                    correlation_id=request.correlation_id,
                )
            )
        return records

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self.embed_documents(request)


def register_retromae(registry: EmbedderRegistry) -> None:
    registry.register("retromae", lambda config: RetroMAEEmbedder(config=config))
