"""Utilities for loading embedding namespace configurations from disk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback when PyYAML is unavailable
    yaml = None  # type: ignore[assignment]

from Medical_KG_rev.config.embeddings import EmbeddingsConfiguration, NamespaceDefinition

from .registry import EmbeddingNamespaceRegistry
from .schema import EmbeddingKind, NamespaceConfig, NamespaceConfigFile

DEFAULT_NAMESPACE_DIR = Path(__file__).resolve().parents[4] / "config" / "embedding" / "namespaces"


def _load_mapping(text: str) -> Mapping[str, object]:
    if yaml is not None:
        data = yaml.safe_load(text)  # type: ignore[no-any-unimported]
        return data or {}
    return json.loads(text)


def _definition_to_config(namespace: str, definition: NamespaceDefinition) -> NamespaceConfig:
    params = dict(definition.parameters)
    endpoint = params.pop("endpoint", None)
    return NamespaceConfig(
        name=definition.name,
        provider=definition.provider,
        kind=EmbeddingKind(definition.kind),
        model_id=definition.model_id,
        model_version=definition.model_version,
        dim=definition.dim,
        pooling=definition.pooling,
        normalize=definition.normalize,
        batch_size=definition.batch_size,
        requires_gpu=definition.requires_gpu,
        endpoint=endpoint,
        parameters=params,
    )


def load_namespace_configs(
    directory: Path | None = None,
    *,
    fallback_config: EmbeddingsConfiguration | None = None,
) -> dict[str, NamespaceConfig]:
    """Load namespace configurations from YAML files or configuration fallback."""

    namespace_dir = directory or Path(os.environ.get("MK_EMBEDDING_NAMESPACE_DIR", DEFAULT_NAMESPACE_DIR))
    configs: dict[str, NamespaceConfig] = {}
    aggregated_path = namespace_dir.parent / "namespaces.yaml"
    if aggregated_path.exists():
        raw_mapping = _load_mapping(aggregated_path.read_text())
        if hasattr(NamespaceConfigFile, "model_validate"):
            file_model = NamespaceConfigFile.model_validate(raw_mapping)  # type: ignore[attr-defined]
            configs.update({name: config for name, config in file_model.namespaces.items()})
        else:
            namespaces = raw_mapping.get("namespaces", raw_mapping)
            for key, value in (namespaces or {}).items():
                config = NamespaceConfig(**value)
                configs[str(key)] = config
    if namespace_dir.exists():
        for path in sorted(namespace_dir.glob("*.y*ml")):
            raw = _load_mapping(path.read_text())
            config = NamespaceConfig(**raw)
            configs[path.stem] = config
    if configs:
        return configs
    if fallback_config is None:
        fallback_config = EmbeddingsConfiguration()
    for namespace, definition in fallback_config.namespaces.items():
        configs[namespace] = _definition_to_config(namespace, definition)
    return configs


def load_registry(
    directory: Path | None = None,
    *,
    fallback_config: EmbeddingsConfiguration | None = None,
) -> EmbeddingNamespaceRegistry:
    """Load namespaces into a runtime registry instance."""

    registry = EmbeddingNamespaceRegistry()
    registry.bulk_register(load_namespace_configs(directory, fallback_config=fallback_config))
    return registry


__all__ = ["DEFAULT_NAMESPACE_DIR", "load_namespace_configs", "load_registry"]
