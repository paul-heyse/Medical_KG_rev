"""Factory responsible for instantiating reranker implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

from .cross_encoder import BGEReranker, MiniLMReranker, MonoT5Reranker, QwenReranker
from .errors import UnknownRerankerError
from .late_interaction import ColBERTReranker
from .lexical import BM25FReranker, BM25Reranker
from .ltr import OpenSearchLTRReranker, VespaRankProfileReranker
from .models import RerankerConfig
from .ports import RerankerPort


@dataclass(slots=True)
class RerankerFactory:
    """Registry + factory to create reranker instances on demand."""

    _constructors: Dict[str, Callable[[], RerankerPort]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._constructors:
            self._constructors = {
                "cross_encoder:bge": lambda: BGEReranker(),
                "cross_encoder:minilm": lambda: MiniLMReranker(),
                "cross_encoder:monot5": lambda: MonoT5Reranker(),
                "cross_encoder:qwen": lambda: QwenReranker(),
                "late_interaction:colbert": lambda: ColBERTReranker(),
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

    def resolve(self, name: str | None, config: RerankerConfig | None = None) -> RerankerPort:
        key = name or "cross_encoder:bge"
        constructor = self._constructors.get(key)
        if constructor is None:
            raise UnknownRerankerError(key, self.available)
        reranker = constructor()
        if config is not None:
            # Basic configuration hooks where supported
            if hasattr(reranker, "batch_size") and config.batch_size:
                reranker.batch_size = config.batch_size  # type: ignore[attr-defined]
            if hasattr(reranker, "precision"):
                setattr(reranker, "precision", config.precision)
            if hasattr(reranker, "device"):
                setattr(reranker, "device", config.device)
            if hasattr(reranker, "quantization") and config.quantization:
                setattr(reranker, "quantization", config.quantization)
        return reranker
