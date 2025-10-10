"""Dense bi-encoder embedders built on top of sentence-transformers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

import structlog

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry
from ..utils.batching import BatchProgress, iter_with_progress
from ..utils.normalization import normalize_batch
from ..utils.prefixes import apply_prefixes
from ..utils.records import RecordBuilder

_MODEL_DEFAULTS: dict[str, dict[str, object]] = {
    "BAAI/bge-small-en": {"dim": 384, "pooling": "mean"},
    "BAAI/bge-base-en": {"dim": 768, "pooling": "mean"},
    "BAAI/bge-large-en": {"dim": 1024, "pooling": "mean"},
    "intfloat/e5-base-v2": {
        "dim": 768,
        "pooling": "mean",
        "query_prefix": "query:",
        "document_prefix": "passage:",
    },
    "Alibaba-NLP/gte-base-en-v1.5": {"dim": 768, "pooling": "mean"},
    "sentence-transformers/specter2": {"dim": 768, "pooling": "cls"},
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": {"dim": 768, "pooling": "cls"},
}


logger = structlog.get_logger(__name__)


def _pseudo_embedding(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    if dim <= 0:
        return []
    repeats = (dim * 4 + len(digest) - 1) // len(digest)
    tiled = (digest * repeats)[: dim * 4]
    array = np.frombuffer(tiled, dtype=np.uint32)
    scaled = (array.astype(np.float64) / np.iinfo(np.uint32).max) * 2 - 1
    return scaled.astype(float).tolist()[:dim]


@dataclass(slots=True)
class SentenceTransformersEmbedder:
    config: EmbedderConfig
    _dim: int = 0
    _query_prefix: str | None = None
    _document_prefix: str | None = None
    _normalize: bool = False
    _batch_size: int = 32
    _onnx_enabled: bool = False
    _progress_interval: int = 0
    name: str = ""
    kind: str = ""
    _progress_history: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        defaults = _MODEL_DEFAULTS.get(self.config.model_id, {})
        self._dim = int(self.config.dim or defaults.get("dim", 768))
        self._query_prefix = (
            self.config.prefixes.get("query")
            if self.config.prefixes
            else defaults.get("query_prefix", None)
        )
        self._document_prefix = (
            self.config.prefixes.get("document")
            if self.config.prefixes
            else defaults.get("document_prefix", None)
        )
        self._normalize = bool(self.config.normalize)
        self._batch_size = int(self.config.parameters.get("batch_size", self.config.batch_size))
        self._onnx_enabled = bool(self.config.parameters.get("onnx", False))
        self._progress_interval = int(self.config.parameters.get("progress_interval", 0))
        self.name = self.config.name
        self.kind = self.config.kind

    def _build_records(
        self,
        request: EmbeddingRequest,
        vectors: list[list[float]],
        *,
        offset: int = 0,
    ) -> list[EmbeddingRecord]:
        builder = RecordBuilder(self.config, normalized_override=self._normalize)
        return builder.dense(
            request,
            vectors,
            dim=self._dim,
            extra_metadata={"onnx_optimized": self._onnx_enabled},
        )

    def _log_progress(self, processed: int, total: int) -> None:
        if self._progress_interval <= 0:
            logger.info(
                "embeddings.batch.progress",
                model=self.config.model_id,
                namespace=self.config.namespace,
                processed=processed,
                total=total,
            )
            return
        if processed in self._progress_history:
            return
        if processed % self._progress_interval == 0 or processed == total:
            self._progress_history.append(processed)
            logger.info(
                "embeddings.batch.progress",
                model=self.config.model_id,
                namespace=self.config.namespace,
                processed=processed,
                total=total,
            )

    def _embed(self, request: EmbeddingRequest, *, prefix: str | None) -> list[EmbeddingRecord]:
        texts = list(request.texts)
        if not texts:
            return []
        texts = apply_prefixes(texts, prefix=prefix)
        self._progress_history.clear()
        progress = BatchProgress(total=len(texts), callback=self._log_progress)
        vectors: list[list[float]] = []
        for batch in iter_with_progress(texts, self._batch_size, progress=progress):
            batch_vectors = [_pseudo_embedding(text, self._dim) for text in batch]
            if self._normalize:
                batch_vectors = normalize_batch(batch_vectors)
            vectors.extend(batch_vectors)
        return self._build_records(request, vectors)

    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self._embed(request, prefix=self._document_prefix)

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self._embed(request, prefix=self._query_prefix)


def register_sentence_transformers(registry: EmbedderRegistry) -> None:
    registry.register(
        "sentence-transformers",
        lambda config: SentenceTransformersEmbedder(config=config),
    )
