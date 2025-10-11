"""Utilities for loading embedding namespace configurations from disk."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
import os

import yaml

from .registry import EmbeddingNamespaceRegistry
from .schema import NamespaceConfig, NamespaceConfigFile


DEFAULT_NAMESPACE_DIR = Path(__file__).resolve().parents[4] / "config" / "embedding" / "namespaces"


def _load_mapping(text: str) -> Mapping[str, object]:
    if yaml is not None:
        data = yaml.safe_load(text)  # type: ignore[no-any-unimported]
        return data or {}
    return json.loads(text)


def load_namespace_configs(
    directory: Path | None = None,
) -> dict[str, NamespaceConfig]:
    """Load namespace configurations from YAML files."""
    namespace_dir = directory or Path(
        os.environ.get("MK_EMBEDDING_NAMESPACE_DIR", DEFAULT_NAMESPACE_DIR)
    )
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
    raise RuntimeError(
        f"No embedding namespace configuration files found in {namespace_dir} "
        "and fallback configuration is disabled."
    )


def load_registry(
    directory: Path | None = None,
) -> EmbeddingNamespaceRegistry:
    """Load namespaces into a runtime registry instance."""
    registry = EmbeddingNamespaceRegistry()
    registry.bulk_register(load_namespace_configs(directory))
    return registry


__all__ = ["DEFAULT_NAMESPACE_DIR", "load_namespace_configs", "load_registry"]
