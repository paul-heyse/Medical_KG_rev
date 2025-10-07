"""Learned sparse embedder approximations for SPLADE style models."""

from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass, field
from typing import Counter, Mapping

import structlog

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry


logger = structlog.get_logger(__name__)


def build_rank_features_mapping(namespace: str) -> Mapping[str, object]:
    """Generate OpenSearch rank_features mapping for a namespace."""

    field_name = namespace.replace(".", "_")
    return {
        "properties": {
            field_name: {
                "type": "rank_features",
                "positive_score_impact": True,
            }
        }
    }


@dataclass(slots=True)
class SPLADEDocEmbedder:
    config: EmbedderConfig
    _top_k: int = 0
    _normalization: str = "none"
    _vocabulary: Counter[str] = field(default_factory=collections.Counter)
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        params = self.config.parameters
        self._top_k = int(params.get("top_k", 400))
        self._normalization = str(params.get("normalization", "l2"))
        self.name = self.config.name
        self.kind = self.config.kind

    def _term_weights(self, text: str) -> dict[str, float]:
        tokens = [token.strip().lower() for token in text.split() if token.strip()]
        weights: Counter[str] = collections.Counter()
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            value = int(digest[:8], 16) / 0xFFFFFFFF
            weights[token] += float(value)
        most_common = weights.most_common(self._top_k)
        ranked = {token: float(weight) for token, weight in most_common}
        self._vocabulary.update(ranked)
        return self._normalize_weights(ranked)

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        if not weights:
            return {}
        if self._normalization == "l1":
            total = sum(weights.values()) or 1.0
            return {token: value / total for token, value in weights.items()}
        if self._normalization == "l2":
            norm = sum(value * value for value in weights.values()) ** 0.5 or 1.0
            return {token: value / norm for token, value in weights.items()}
        if self._normalization == "max":
            maximum = max(weights.values()) or 1.0
            return {token: value / maximum for token, value in weights.items()}
        return weights

    def vocabulary_snapshot(self, top_n: int = 50) -> Mapping[str, float]:
        return dict(self._vocabulary.most_common(top_n))

    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        ids = list(request.ids or [f"{request.namespace}:{index}" for index in range(len(request.texts))])
        records: list[EmbeddingRecord] = []
        for chunk_id, text in zip(ids, request.texts, strict=False):
            weights = self._term_weights(text)
            logger.debug(
                "splade.embedding.generated",
                chunk_id=chunk_id,
                terms=len(weights),
                normalization=self._normalization,
            )
            records.append(
                EmbeddingRecord(
                    id=chunk_id,
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model_id=self.config.model_id,
                    model_version=self.config.model_version,
                    kind=self.config.kind,
                    dim=len(weights),
                    terms=weights,
                    metadata={"provider": self.config.provider},
                    correlation_id=request.correlation_id,
                )
            )
        return records

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self.embed_documents(request)


@dataclass(slots=True)
class SPLADEQueryEmbedder(SPLADEDocEmbedder):
    pass


@dataclass(slots=True)
class PyseriniSparseEmbedder:
    config: EmbedderConfig
    _weighting: str = "bm25"
    _normalization: str = "none"
    _vocabulary: Counter[str] = field(default_factory=collections.Counter)
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        params = self.config.parameters
        self._weighting = params.get("weighting", "bm25")
        self._normalization = str(params.get("normalization", "none"))
        self.name = self.config.name
        self.kind = self.config.kind

    def _term_weights(self, text: str) -> dict[str, float]:
        tokens = [token.strip().lower() for token in text.split() if token.strip()]
        weights: dict[str, float] = {}
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            magnitude = int(digest[:8], 16) / 0xFFFFFFFF
            if self._weighting == "bm25":
                weight = 1.2 + 1.5 * magnitude
            else:
                weight = magnitude
            weights[token] = weights.get(token, 0.0) + float(weight)
        self._vocabulary.update(weights)
        return self._normalize(weights)

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        if not weights:
            return {}
        if self._normalization == "max":
            maximum = max(weights.values()) or 1.0
            return {token: value / maximum for token, value in weights.items()}
        if self._normalization == "l2":
            norm = sum(value * value for value in weights.values()) ** 0.5 or 1.0
            return {token: value / norm for token, value in weights.items()}
        return weights

    def vocabulary_snapshot(self, top_n: int = 50) -> Mapping[str, float]:
        return dict(self._vocabulary.most_common(top_n))

    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        ids = list(request.ids or [f"{request.namespace}:{index}" for index in range(len(request.texts))])
        records: list[EmbeddingRecord] = []
        for chunk_id, text in zip(ids, request.texts, strict=False):
            weights = self._term_weights(text)
            records.append(
                EmbeddingRecord(
                    id=chunk_id,
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model_id=self.config.model_id,
                    model_version=self.config.model_version,
                    kind=self.config.kind,
                    dim=len(weights),
                    terms=weights,
                    metadata={"provider": self.config.provider, "weighting": self._weighting},
                    correlation_id=request.correlation_id,
                )
            )
        return records

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self.embed_documents(request)


def register_sparse(registry: EmbedderRegistry) -> None:
    registry.register("splade-doc", lambda config: SPLADEDocEmbedder(config=config))
    registry.register("splade-query", lambda config: SPLADEQueryEmbedder(config=config))
    registry.register("pyserini", lambda config: PyseriniSparseEmbedder(config=config))
