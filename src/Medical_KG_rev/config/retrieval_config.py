"""Configuration helpers for the hybrid retrieval subsystem."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover - graceful fallback when PyYAML missing
    raise ImportError("PyYAML is required to load retrieval configuration") from exc

DEFAULT_RETRIEVAL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "retrieval.yaml"


def _to_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


@dataclass(slots=True)
class BM25Config:
    """Configuration for the structured BM25 retriever."""

    index_path: Path = Path("indexes/bm25")
    field_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "title": 3.5,
            "section_headers": 2.5,
            "paragraph": 1.0,
            "caption": 1.5,
            "table_text": 1.2,
            "footnote": 0.5,
            "refs_text": 0.1,
        }
    )
    analyzer: str = "medical_standard"
    synonyms_path: Path | None = None
    enable_synonyms: bool = True
    query_timeout_ms: int = 250
    cache_ttl_seconds: int = 300

    def __post_init__(self) -> None:
        if isinstance(self.index_path, str):  # pragma: no cover - defensive
            self.index_path = Path(self.index_path)
        self.synonyms_path = _to_path(self.synonyms_path)
        self._validate()

    def _validate(self) -> None:
        if not self.field_boosts:
            raise ValueError("field_boosts cannot be empty for BM25 configuration")
        for field_name, boost in self.field_boosts.items():
            if boost <= 0:
                raise ValueError(f"BM25 field '{field_name}' boost must be positive")
        if self.query_timeout_ms <= 0:
            raise ValueError("BM25 query timeout must be positive")
        if self.cache_ttl_seconds < 0:
            raise ValueError("BM25 cache TTL cannot be negative")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> BM25Config:
        if not data:
            return cls()
        payload = dict(data)
        if "index_path" in payload:
            path_value = _to_path(payload["index_path"])
            if path_value is None:
                payload.pop("index_path", None)
            else:
                payload["index_path"] = path_value
        if "synonyms_path" in payload:
            payload["synonyms_path"] = _to_path(payload["synonyms_path"])
        if "field_boosts" in payload and isinstance(payload["field_boosts"], Mapping):
            payload["field_boosts"] = {
                str(key): float(value) for key, value in payload["field_boosts"].items()
            }
        return cls(**payload)


@dataclass(slots=True)
class SPLADEConfig:
    """Configuration for SPLADE-v3 sparse retrieval."""

    index_path: Path = Path("indexes/splade_v3")
    model_name: str = "naver/splade-v3"
    tokenizer_name: str = "naver/splade-v3"
    max_tokens: int = 512
    sparsity_threshold: float = 0.01
    max_terms: int = 4096
    quantization_bits: int = 8
    batch_size: int = 16
    cache_ttl_seconds: int = 300
    query_timeout_ms: int = 400

    def __post_init__(self) -> None:
        if isinstance(self.index_path, str):  # pragma: no cover - defensive
            self.index_path = Path(self.index_path)
        self._validate()

    def _validate(self) -> None:
        if not self.model_name:
            raise ValueError("SPLADE model_name cannot be empty")
        if not self.tokenizer_name:
            raise ValueError("SPLADE tokenizer_name cannot be empty")
        if self.max_tokens <= 0 or self.max_tokens > 512:
            raise ValueError("SPLADE max_tokens must be between 1 and 512")
        if not 0.0 <= self.sparsity_threshold <= 1.0:
            raise ValueError("SPLADE sparsity_threshold must be in [0, 1]")
        if self.max_terms <= 0:
            raise ValueError("SPLADE max_terms must be positive")
        if self.quantization_bits not in {4, 8, 16}:
            raise ValueError("SPLADE quantization_bits must be one of {4, 8, 16}")
        if self.batch_size <= 0:
            raise ValueError("SPLADE batch_size must be positive")
        if self.cache_ttl_seconds < 0:
            raise ValueError("SPLADE cache TTL cannot be negative")
        if self.query_timeout_ms <= 0:
            raise ValueError("SPLADE query timeout must be positive")
        if self.tokenizer_name != self.model_name:
            raise ValueError("SPLADE tokenizer_name must match model_name for alignment")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> SPLADEConfig:
        if not data:
            return cls()
        payload = dict(data)
        if "index_path" in payload:
            path_value = _to_path(payload["index_path"])
            if path_value is None:
                payload.pop("index_path", None)
            else:
                payload["index_path"] = path_value
        return cls(**payload)


@dataclass(slots=True)
class Qwen3Config:
    """Configuration for Qwen3 dense embedding retrieval."""

    index_path: Path = Path("vectors/qwen3.faiss")
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"
    embedding_dimension: int = 4096
    batch_size: int = 32
    backend: str = "faiss"
    ann_search_k: int = 100
    normalize_embeddings: bool = True
    cache_ttl_seconds: int = 300
    query_timeout_ms: int = 400

    def __post_init__(self) -> None:
        if isinstance(self.index_path, str):  # pragma: no cover - defensive
            self.index_path = Path(self.index_path)
        self._validate()

    def _validate(self) -> None:
        if not self.model_name:
            raise ValueError("Qwen3 model_name cannot be empty")
        if not self.tokenizer_name:
            raise ValueError("Qwen3 tokenizer_name cannot be empty")
        if self.embedding_dimension <= 0:
            raise ValueError("Qwen3 embedding_dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("Qwen3 batch_size must be positive")
        if self.ann_search_k <= 0:
            raise ValueError("Qwen3 ann_search_k must be positive")
        if self.cache_ttl_seconds < 0:
            raise ValueError("Qwen3 cache TTL cannot be negative")
        if self.query_timeout_ms <= 0:
            raise ValueError("Qwen3 query timeout must be positive")
        if self.backend not in {"faiss", "qdrant"}:
            raise ValueError("Qwen3 backend must be either 'faiss' or 'qdrant'")
        if self.tokenizer_name != self.model_name:
            raise ValueError("Qwen3 tokenizer_name must match model_name for alignment")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> Qwen3Config:
        if not data:
            return cls()
        payload = dict(data)
        if "index_path" in payload:
            path_value = _to_path(payload["index_path"])
            if path_value is None:
                payload.pop("index_path", None)
            else:
                payload["index_path"] = path_value
        return cls(**payload)


@dataclass(slots=True)
class FusionConfig:
    """Configuration for fusion ranking of retrieval components."""

    strategy: str = "rrf"
    rrf_k: int = 60
    weights: dict[str, float] = field(default_factory=dict)
    cache_ttl_seconds: int = 300
    query_timeout_ms: int = 500

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.rrf_k <= 0:
            raise ValueError("Fusion rrf_k must be positive")
        if self.cache_ttl_seconds < 0:
            raise ValueError("Fusion cache TTL cannot be negative")
        if self.query_timeout_ms <= 0:
            raise ValueError("Fusion query timeout must be positive")
        for component, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"Fusion weight for '{component}' cannot be negative")


@dataclass(slots=True)
class RetrievalConfig:
    """Composite configuration for the hybrid retrieval system."""

    default_backend: str = "hybrid"
    bm25: BM25Config = field(default_factory=BM25Config)
    splade: SPLADEConfig = field(default_factory=SPLADEConfig)
    qwen3: Qwen3Config = field(default_factory=Qwen3Config)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    def __post_init__(self) -> None:
        if self.default_backend not in {"bm25", "splade", "qwen3", "hybrid"}:
            raise ValueError(
                "default_backend must be one of 'bm25', 'splade', 'qwen3', or 'hybrid'"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> RetrievalConfig:
        if not data:
            return cls()
        payload = dict(data)
        bm25_payload = payload.get("bm25")
        splade_payload = payload.get("splade")
        qwen_payload = payload.get("qwen3")
        fusion_payload = payload.get("fusion")
        return cls(
            default_backend=str(payload.get("default_backend", "hybrid")),
            bm25=BM25Config.from_mapping(
                bm25_payload if isinstance(bm25_payload, Mapping) else None
            ),
            splade=SPLADEConfig.from_mapping(
                splade_payload if isinstance(splade_payload, Mapping) else None
            ),
            qwen3=Qwen3Config.from_mapping(
                qwen_payload if isinstance(qwen_payload, Mapping) else None
            ),
            fusion=(
                FusionConfig(**dict(fusion_payload))
                if isinstance(fusion_payload, Mapping)
                else FusionConfig()
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> RetrievalConfig:
        return cls.from_mapping(data)

    @classmethod
    def from_yaml(
        cls, path: str | Path | None = None
    ) -> RetrievalConfig:  # pragma: no cover - simple IO
        target = Path(path) if path is not None else DEFAULT_RETRIEVAL_CONFIG_PATH
        if yaml is None or not target.exists():
            return cls()
        payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"Invalid retrieval config structure in {target}")
        return cls.from_mapping(payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "default_backend": self.default_backend,
            "bm25": {
                "index_path": str(self.bm25.index_path),
                "field_boosts": dict(self.bm25.field_boosts),
                "analyzer": self.bm25.analyzer,
                "synonyms_path": str(self.bm25.synonyms_path) if self.bm25.synonyms_path else None,
                "enable_synonyms": self.bm25.enable_synonyms,
                "query_timeout_ms": self.bm25.query_timeout_ms,
                "cache_ttl_seconds": self.bm25.cache_ttl_seconds,
            },
            "splade": {
                "index_path": str(self.splade.index_path),
                "model_name": self.splade.model_name,
                "tokenizer_name": self.splade.tokenizer_name,
                "max_tokens": self.splade.max_tokens,
                "sparsity_threshold": self.splade.sparsity_threshold,
                "max_terms": self.splade.max_terms,
                "quantization_bits": self.splade.quantization_bits,
                "batch_size": self.splade.batch_size,
                "cache_ttl_seconds": self.splade.cache_ttl_seconds,
                "query_timeout_ms": self.splade.query_timeout_ms,
            },
            "qwen3": {
                "index_path": str(self.qwen3.index_path),
                "model_name": self.qwen3.model_name,
                "tokenizer_name": self.qwen3.tokenizer_name,
                "embedding_dimension": self.qwen3.embedding_dimension,
                "batch_size": self.qwen3.batch_size,
                "backend": self.qwen3.backend,
                "ann_search_k": self.qwen3.ann_search_k,
                "normalize_embeddings": self.qwen3.normalize_embeddings,
                "cache_ttl_seconds": self.qwen3.cache_ttl_seconds,
                "query_timeout_ms": self.qwen3.query_timeout_ms,
            },
            "fusion": {
                "strategy": self.fusion.strategy,
                "rrf_k": self.fusion.rrf_k,
                "weights": dict(self.fusion.weights),
                "cache_ttl_seconds": self.fusion.cache_ttl_seconds,
                "query_timeout_ms": self.fusion.query_timeout_ms,
            },
        }


__all__ = [
    "DEFAULT_RETRIEVAL_CONFIG_PATH",
    "BM25Config",
    "FusionConfig",
    "Qwen3Config",
    "RetrievalConfig",
    "SPLADEConfig",
]
