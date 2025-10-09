"""Registry and downloader for reranking models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from Medical_KG_rev.observability import logger as global_logger

logger = global_logger.bind(module="reranking.model_registry")

DEFAULT_CONFIG_PATH = Path("config/retrieval/reranking_models.yaml")
DEFAULT_CACHE_DIR = Path("model_cache/rerankers")


@dataclass(slots=True, frozen=True)
class RerankerModel:
    """Structured representation of a reranker model entry."""

    key: str
    reranker_id: str
    model_id: str
    version: str
    provider: str = "huggingface"
    revision: str | None = None
    requires_gpu: bool = False
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cache_subdir: str | None = None

    def cache_path(self, base_dir: Path) -> Path:
        folder = self.cache_subdir or self.key
        return base_dir / folder


@dataclass(slots=True)
class ModelHandle:
    """Return type describing an ensured model and its cache path."""

    model: RerankerModel
    path: Path


class ModelDownloadError(RuntimeError):
    """Raised when a model could not be prepared for use."""


class ModelDownloader:
    """Simple downloader that materialises a cache manifest for a model."""

    def __init__(self, *, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or DEFAULT_CACHE_DIR

    def fetch(self, model: RerankerModel) -> Path:
        cache_dir = model.cache_path(self.base_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = cache_dir / "manifest.json"
        if not manifest.exists():
            manifest.write_text(
                json.dumps(
                    {
                        "model_id": model.model_id,
                        "version": model.version,
                        "provider": model.provider,
                        "revision": model.revision,
                        "requires_gpu": model.requires_gpu,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return cache_dir


@dataclass(slots=True)
class RerankerModelRegistry:
    """Loads reranker model metadata and ensures local availability."""

    config_path: Path | None = None
    cache_dir: Path | None = None
    downloader: ModelDownloader | None = None
    _models: dict[str, RerankerModel] = field(init=False, default_factory=dict)
    _default_key: str = field(init=False, default="bge-reranker-base")
    _by_model_id: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        config = Path(self.config_path) if self.config_path else DEFAULT_CONFIG_PATH
        cache = Path(self.cache_dir) if self.cache_dir else DEFAULT_CACHE_DIR
        self.config_path = config
        self.cache_dir = cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = self.downloader or ModelDownloader(base_dir=self.cache_dir)
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        payload: Mapping[str, Any] = {}
        if self.config_path and self.config_path.exists():
            content = self.config_path.read_text("utf-8")
            payload = yaml.safe_load(content) or {}
        else:
            logger.warning(
                "reranking.model_registry.config_missing",
                path=str(self.config_path),
            )
        cache_dir = payload.get("cache_dir")
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.downloader:
                self.downloader.base_dir = self.cache_dir
        self._default_key = str(payload.get("default", self._default_key))
        models_section = payload.get("models") or {}
        if not models_section:
            models_section = {
                "bge-reranker-base": {
                    "reranker_id": "cross_encoder:bge",
                    "model_id": "BAAI/bge-reranker-base",
                    "version": "v1.0",
                    "provider": "huggingface",
                }
            }
        self._models = {}
        self._by_model_id = {}
        for key, data in models_section.items():
            reranker_id = str(data.get("reranker_id", "cross_encoder:bge"))
            model_id = str(data.get("model_id", key))
            version = str(data.get("version", "v1"))
            model = RerankerModel(
                key=str(key),
                reranker_id=reranker_id,
                model_id=model_id,
                version=version,
                provider=str(data.get("provider", "huggingface")),
                revision=data.get("revision"),
                requires_gpu=bool(data.get("requires_gpu", False)),
                description=data.get("description"),
                metadata=data.get("metadata") or {},
                cache_subdir=data.get("cache_subdir"),
            )
            self._models[model.key] = model
            self._by_model_id[model.model_id] = model.key
        if self._default_key not in self._models:
            logger.warning(
                "reranking.model_registry.invalid_default",
                default=self._default_key,
            )
            self._default_key = next(iter(self._models))

    # ------------------------------------------------------------------
    @property
    def default_key(self) -> str:
        return self._default_key

    # ------------------------------------------------------------------
    def list_models(self) -> list[RerankerModel]:
        return sorted(self._models.values(), key=lambda entry: entry.key)

    # ------------------------------------------------------------------
    def resolve_key(self, identifier: str | None) -> str:
        if not identifier:
            return self._default_key
        if identifier in self._models:
            return identifier
        if identifier in self._by_model_id:
            return self._by_model_id[identifier]
        for entry in self._models.values():
            if identifier == entry.reranker_id:
                return entry.key
        raise KeyError(f"Unknown reranker model '{identifier}'")

    # ------------------------------------------------------------------
    def get(self, identifier: str | None = None) -> RerankerModel:
        key = self.resolve_key(identifier)
        return self._models[key]

    # ------------------------------------------------------------------
    def ensure(self, identifier: str | None = None) -> ModelHandle:
        model = self.get(identifier)
        try:
            path = self.downloader.fetch(model) if self.downloader else model.cache_path(self.cache_dir)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelDownloadError(str(exc)) from exc
        return ModelHandle(model=model, path=path)

    # ------------------------------------------------------------------
    def reload(self) -> None:
        self._load()


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CONFIG_PATH",
    "ModelDownloadError",
    "ModelDownloader",
    "ModelHandle",
    "RerankerModel",
    "RerankerModelRegistry",
]
