from __future__ import annotations

from pathlib import Path
import json

import pytest

from Medical_KG_rev.services.embedding.namespace.loader import load_namespace_configs
from Medical_KG_rev.services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from Medical_KG_rev.services.embedding.namespace.schema import EmbeddingKind, NamespaceConfig


def _sample_config(kind: EmbeddingKind = EmbeddingKind.SINGLE_VECTOR) -> NamespaceConfig:
    return NamespaceConfig(
        name="test-model",
        kind=kind,
        model_id="test/model",
        model_version="v1",
        dim=4,
        provider="unit-test",
        parameters={"foo": "bar"},
    )


def test_namespace_registry_lists_available() -> None:
    registry = EmbeddingNamespaceRegistry()
    registry.register("single_vector.test.4.v1", _sample_config())
    registry.register("sparse.test.8.v1", _sample_config(EmbeddingKind.SPARSE))

    assert registry.get("single_vector.test.4.v1").provider == "unit-test"
    assert registry.list_by_kind(EmbeddingKind.SINGLE_VECTOR) == ["single_vector.test.4.v1"]
    with pytest.raises(ValueError) as exc:
        registry.get("missing.namespace")
    assert "Available: single_vector.test.4.v1, sparse.test.8.v1" in str(exc.value)


def test_load_namespace_configs_from_directory(tmp_path: Path) -> None:
    namespace_dir = tmp_path / "namespaces"
    namespace_dir.mkdir()
    payload = {
        "name": "demo",
        "kind": "single_vector",
        "model_id": "demo/model",
        "model_version": "v1",
        "dim": 16,
        "provider": "demo-provider",
        "parameters": {"max_tokens": 2048},
        "endpoint": "http://localhost:9000/v1",
    }
    (namespace_dir / "single_vector.demo.16.v1.yaml").write_text(json.dumps(payload))

    configs = load_namespace_configs(namespace_dir)
    assert "single_vector.demo.16.v1" in configs
    config = configs["single_vector.demo.16.v1"]
    assert config.endpoint == "http://localhost:9000/v1"
    assert config.parameters["max_tokens"] == 2048
