"""Factory responsible for instantiating reranker implementations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .cross_encoder import BGEReranker, MiniLMReranker, MonoT5Reranker, QwenReranker
from .errors import RerankingError, UnknownRerankerError
from .late_interaction import ColbertIndexReranker, ColBERTReranker, QdrantColBERTReranker
from .lexical import BM25FReranker, BM25Reranker
from .ltr import OpenSearchLTRReranker, VespaRankProfileReranker
from .models import RerankerConfig
from .ports import RerankerPort


@dataclass(slots=True)
class RerankerFactory:
    """Registry + factory to create reranker instances on demand."""

    _constructors: dict[str, Callable[[], RerankerPort]] = field(default_factory=dict)
    _instances: dict[tuple[str, tuple[Any, ...] | None], RerankerPort] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._constructors:
            self._constructors = {
                "cross_encoder:bge": lambda: BGEReranker(),
                "cross_encoder:minilm": lambda: MiniLMReranker(),
                "cross_encoder:monot5": lambda: MonoT5Reranker(),
                "cross_encoder:qwen": lambda: QwenReranker(),
                "late_interaction:colbert": lambda: ColBERTReranker(),
                "late_interaction:colbert_index": lambda: ColbertIndexReranker(
                    index=_LazyColbertIndex()
                ),
                "late_interaction:qdrant": lambda: QdrantColBERTReranker(
                    client=_LazyQdrant(),
                    collection="colbert",
                ),
                "lexical:bm25": lambda: BM25Reranker(),
                "lexical:bm25f": lambda: BM25FReranker(),
                "ltr:opensearch": lambda: OpenSearchLTRReranker(),
                "ltr:vespa": lambda: VespaRankProfileReranker(),
            }

    @property
    def available(self) -> list[str]:
        return sorted(self._constructors.keys())

    def register(self, name: str, factory: Callable[[], RerankerPort]) -> None:
        self._constructors[name] = factory
        self._instances = {
            key: instance for key, instance in self._instances.items() if key[0] != name
        }

    def clear_cache(self) -> None:
        """Evict cached reranker instances (useful for tests)."""
        self._instances.clear()

    def resolve(self, name: str | None, config: RerankerConfig | None = None) -> RerankerPort:
        key = name or "cross_encoder:bge"
        constructor = self._constructors.get(key)
        if constructor is None:
            raise UnknownRerankerError(key, self.available)
        config_signature: tuple[Any, ...] | None = None
        if config is not None:
            config_signature = (
                getattr(config, "batch_size", None),
                getattr(config, "precision", None),
                getattr(config, "device", None),
                getattr(config, "quantization", None),
                getattr(config, "requires_gpu", None),
            )
        cache_key = (key, config_signature)
        reranker = self._instances.get(cache_key)
        if reranker is None:
            reranker = constructor()
            self._instances[cache_key] = reranker
        if config is not None:
            # Basic configuration hooks where supported
            if hasattr(reranker, "batch_size") and config.batch_size:
                reranker.batch_size = config.batch_size  # type: ignore[attr-defined]
            if hasattr(reranker, "precision"):
                reranker.precision = config.precision
            if hasattr(reranker, "device"):
                reranker.device = config.device
            if hasattr(reranker, "quantization") and config.quantization:
                reranker.quantization = config.quantization
        return reranker


class _LazyColbertIndex:
    """Placeholder index that raises helpful errors until configured."""

    def encode_queries(self, queries: Sequence[str]) -> Sequence[Sequence[Sequence[float]]]:
        raise RerankingError(
            title="ColBERT index not configured",
            status=503,
            detail=(
                "External ColBERT integration requires providing an index instance via "
                "RerankerFactory.register('late_interaction:colbert_index', ...)"
            ),
        )

    def get_document_vectors(self, doc_id: str) -> Sequence[Sequence[float]]:
        raise RerankingError(
            title="ColBERT index not configured",
            status=503,
            detail="External ColBERT index instance has not been initialised",
        )


class _LazyQdrant:
    """Placeholder client providing descriptive errors until configured."""

    def retrieve(
        self, *args: Any, **kwargs: Any
    ) -> Sequence[Any]:  # pragma: no cover - simple guard
        raise RerankingError(
            title="Qdrant not configured",
            status=503,
            detail=(
                "Qdrant integration requires registering a configured client via "
                "RerankerFactory.register('late_interaction:qdrant', ...)"
            ),
        )
