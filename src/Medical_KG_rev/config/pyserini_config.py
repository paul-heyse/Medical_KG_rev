"""Configuration loader for Pyserini SPLADE service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover
    BaseModel = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


DEFAULT_PYSERINI_CONFIG = (
    Path(__file__).resolve().parents[3] / "config" / "embedding" / "pyserini.yaml"
)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    return yaml.safe_load(path.read_text()) or {}


@dataclass(slots=True)
class PyseriniServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8002
    gpu_memory_utilization: float = 0.6


@dataclass(slots=True)
class PyseriniModelConfig:
    name: str = "naver/splade-cocondenser-ensembledistil"
    cache_dir: str = "/models/splade"


@dataclass(slots=True)
class PyseriniExpansionSideConfig:
    enabled: bool = True
    top_k_terms: int = 400
    normalize_weights: bool = True


@dataclass(slots=True)
class PyseriniExpansionConfig:
    doc_side: PyseriniExpansionSideConfig = field(default_factory=PyseriniExpansionSideConfig)
    query_side: PyseriniExpansionSideConfig = field(
        default_factory=lambda: PyseriniExpansionSideConfig(enabled=False, top_k_terms=200)
    )


@dataclass(slots=True)
class PyseriniOpenSearchConfig:
    rank_features_field: str = "splade_terms"
    max_weight: float = 10.0


@dataclass(slots=True)
class PyseriniConfig:
    service: PyseriniServiceConfig = field(default_factory=PyseriniServiceConfig)
    model: PyseriniModelConfig = field(default_factory=PyseriniModelConfig)
    expansion: PyseriniExpansionConfig = field(default_factory=PyseriniExpansionConfig)
    opensearch: PyseriniOpenSearchConfig = field(default_factory=PyseriniOpenSearchConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PyseriniConfig:
        service = PyseriniServiceConfig(**data.get("service", {}))
        model = PyseriniModelConfig(**data.get("model", {}))
        expansion = PyseriniExpansionConfig(
            doc_side=PyseriniExpansionSideConfig(**data.get("expansion", {}).get("doc_side", {})),
            query_side=PyseriniExpansionSideConfig(
                **{
                    "enabled": False,
                    "top_k_terms": 200,
                    **data.get("expansion", {}).get("query_side", {}),
                }
            ),
        )
        opensearch = PyseriniOpenSearchConfig(**data.get("opensearch", {}))
        return cls(service=service, model=model, expansion=expansion, opensearch=opensearch)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> PyseriniConfig:
        config_path = Path(path) if path is not None else DEFAULT_PYSERINI_CONFIG
        data = _load_mapping(config_path)
        if BaseModel is not None:
            model = _PyseriniConfigModel.model_validate(data)
            return cls.from_mapping(model.model_dump())
        return cls.from_mapping(data)


if BaseModel is not None:  # pragma: no cover - executed when pydantic installed

    class _PyseriniServiceModel(BaseModel):
        host: str = Field(default="0.0.0.0")
        port: int = Field(default=8002, ge=1)
        gpu_memory_utilization: float = Field(default=0.6, ge=0.0, le=1.0)

    class _PyseriniModelModel(BaseModel):
        name: str = Field(default="naver/splade-cocondenser-ensembledistil")
        cache_dir: str = Field(default="/models/splade")

    class _PyseriniExpansionSideModel(BaseModel):
        enabled: bool = Field(default=True)
        top_k_terms: int = Field(default=400, ge=1)
        normalize_weights: bool = Field(default=True)

    class _PyseriniExpansionModel(BaseModel):
        doc_side: _PyseriniExpansionSideModel = Field(default_factory=_PyseriniExpansionSideModel)
        query_side: _PyseriniExpansionSideModel = Field(
            default_factory=lambda: _PyseriniExpansionSideModel(enabled=False, top_k_terms=200)
        )

    class _PyseriniOpenSearchModel(BaseModel):
        rank_features_field: str = Field(default="splade_terms")
        max_weight: float = Field(default=10.0, ge=0.0)

    class _PyseriniConfigModel(BaseModel):
        service: _PyseriniServiceModel = Field(default_factory=_PyseriniServiceModel)
        model: _PyseriniModelModel = Field(default_factory=_PyseriniModelModel)
        expansion: _PyseriniExpansionModel = Field(default_factory=_PyseriniExpansionModel)
        opensearch: _PyseriniOpenSearchModel = Field(default_factory=_PyseriniOpenSearchModel)


def load_pyserini_config(path: str | Path | None = None) -> PyseriniConfig:
    return PyseriniConfig.from_yaml(path)


__all__ = [
    "DEFAULT_PYSERINI_CONFIG",
    "PyseriniConfig",
    "PyseriniExpansionConfig",
    "PyseriniExpansionSideConfig",
    "PyseriniModelConfig",
    "PyseriniOpenSearchConfig",
    "PyseriniServiceConfig",
    "load_pyserini_config",
]
