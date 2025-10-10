"""Pyserini SPLADE adapter with OpenSearch rank_features integration."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass

import structlog

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry
from ..utils.records import RecordBuilder

logger = structlog.get_logger(__name__)


def build_rank_features_mapping(namespace: str) -> Mapping[str, object]:
    """Generate an OpenSearch mapping for SPLADE `rank_features`."""
    field_name = namespace.replace(".", "_")
    return {
        "properties": {
            field_name: {
                "type": "rank_features",
                "positive_score_impact": True,
            }
        }
    }


class PyseriniNotInstalledError(RuntimeError):
    """Raised when Pyserini is required but not available."""


def _load_pyserini_encoder(mode: str) -> type:
    """Import the appropriate Pyserini encoder class."""
    try:
        module = importlib.import_module("pyserini.encode")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dependency
        raise PyseriniNotInstalledError(
            "pyserini>=0.22.0 is required for SPLADE embeddings"
        ) from exc
    class_name = "SpladeQueryEncoder" if mode == "query" else "SpladeDocumentEncoder"
    try:
        return getattr(module, class_name)
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise PyseriniNotInstalledError(f"pyserini.encode.{class_name} is unavailable") from exc


@dataclass(slots=True)
class PyseriniSparseEmbedder:
    """Thin wrapper delegating SPLADE expansion to Pyserini."""

    config: EmbedderConfig
    _mode: str = "document"
    _top_k: int = 400
    _max_terms: int | None = None
    _encoder: object | None = None
    _builder: RecordBuilder | None = None
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        params = self.config.parameters
        self._mode = str(params.get("mode", "document")).lower()
        self._top_k = int(params.get("top_k", 400))
        self._max_terms = params.get("max_terms")
        encoder_cls = _load_pyserini_encoder(self._mode)
        self._encoder = encoder_cls(self.config.model_id)
        self._builder = RecordBuilder(self.config, normalized_override=self.config.normalize)
        self.name = self.config.name
        self.kind = self.config.kind
        logger.info(
            "embedding.splade.pyserini.initialised",
            namespace=self.config.namespace,
            mode=self._mode,
            top_k=self._top_k,
            model=self.config.model_id,
        )

    def _expand(self, text: str) -> dict[str, float]:
        if not text:
            return {}
        assert self._encoder is not None
        weights = self._encoder.encode(text, top_k=self._top_k)  # type: ignore[attr-defined]
        if not isinstance(weights, dict):  # pragma: no cover - Pyserini contract
            raise TypeError("Pyserini SPLADE encoder must return a dict of term weights")
        if self._max_terms is not None:
            items = sorted(weights.items(), key=lambda item: item[1], reverse=True)[
                : self._max_terms
            ]
            return {term: float(score) for term, score in items}
        return {term: float(score) for term, score in weights.items()}

    def _records(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        assert self._builder is not None
        expansions = [self._expand(text) for text in request.texts]
        metadata = {
            "mode": self._mode,
            "top_k": self._top_k,
            "provider": self.config.provider,
        }
        safe_expansions = [weights or {"__empty__": 0.0} for weights in expansions]
        records = self._builder.sparse(
            request,
            safe_expansions,
            extra_metadata=metadata,
            dim_from_terms=True,
        )
        final_records: list[EmbeddingRecord] = []
        for weights, record in zip(expansions, records, strict=False):
            if weights:
                final_records.append(record)
                continue
            final_records.append(
                EmbeddingRecord(
                    id=record.id,
                    tenant_id=record.tenant_id,
                    namespace=record.namespace,
                    model_id=record.model_id,
                    model_version=record.model_version,
                    kind=record.kind,
                    dim=0,
                    terms={},
                    metadata=record.metadata,
                    normalized=record.normalized,
                    correlation_id=record.correlation_id,
                )
            )
        return final_records

    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self._records(request)

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        # Query-side expansion uses the same encoder but may have different top_k configured.
        return self._records(request)


def register_sparse(registry: EmbedderRegistry) -> None:
    registry.register("pyserini", lambda config: PyseriniSparseEmbedder(config=config))
