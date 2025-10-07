"""Namespace governance utilities for embedding configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .ports import EmbedderConfig, EmbeddingKind


class DimensionMismatchError(RuntimeError):
    """Raised when an embedding's dimensionality does not match namespace expectations."""


@dataclass(slots=True)
class NamespaceConfig:
    """Registered namespace details used for validation."""

    name: str
    kind: EmbeddingKind
    expected_dim: int | None
    model_id: str
    model_version: str
    embedder_name: str


class NamespaceManager:
    """Tracks namespaces and enforces dimensionality invariants."""

    def __init__(self) -> None:
        self._namespaces: dict[str, NamespaceConfig] = {}
        self._observed_dims: dict[str, int] = {}

    def reset(self) -> None:
        """Clear registered namespaces and observed dimensions."""

        self._namespaces.clear()
        self._observed_dims.clear()

    def register(self, config: EmbedderConfig) -> NamespaceConfig:
        parts = config.namespace_parts
        dim = None if parts["dim"] == "auto" else int(parts["dim"])
        namespace = NamespaceConfig(
            name=config.namespace,
            kind=config.kind,
            expected_dim=dim or config.dim,
            model_id=config.model_id,
            model_version=config.model_version,
            embedder_name=config.name,
        )
        self._namespaces[namespace.name] = namespace
        return namespace

    def namespaces(self) -> Mapping[str, NamespaceConfig]:
        return dict(self._namespaces)

    def introspect_dimension(self, namespace: str, dimension: int) -> None:
        if namespace not in self._namespaces:
            raise KeyError(f"Namespace '{namespace}' is not registered")
        expected = self._namespaces[namespace].expected_dim
        if expected is not None and expected != dimension:
            if self._namespaces[namespace].kind == "sparse" and dimension <= expected:
                self._observed_dims.setdefault(namespace, dimension)
                return
            raise DimensionMismatchError(
                f"Namespace '{namespace}' expected dimension {expected} but observed {dimension}"
            )
        self._observed_dims.setdefault(namespace, dimension)

    def get_dimension(self, namespace: str) -> int | None:
        if namespace in self._observed_dims:
            return self._observed_dims[namespace]
        config = self._namespaces.get(namespace)
        return config.expected_dim if config else None

    def validate_record(self, namespace: str, record_dim: int | None) -> None:
        if record_dim is None:
            return
        expected = self._namespaces.get(namespace)
        if expected and expected.expected_dim and expected.expected_dim != record_dim:
            if expected.kind == "sparse" and record_dim <= expected.expected_dim:
                return
            raise DimensionMismatchError(
                f"Embedding record dimension {record_dim} does not match namespace {expected.expected_dim}"
            )
