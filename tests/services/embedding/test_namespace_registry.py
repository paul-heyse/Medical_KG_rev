from __future__ import annotations

import json
from pathlib import Path

import pytest

from Medical_KG_rev.services.embedding.namespace.loader import load_namespace_configs
from Medical_KG_rev.services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from Medical_KG_rev.services.embedding.namespace.schema import EmbeddingKind, NamespaceConfig


def _sample_config(
    kind: EmbeddingKind = EmbeddingKind.SINGLE_VECTOR,
    provider: str = "unit-test",
) -> NamespaceConfig:
    return NamespaceConfig(
        name="test-model",
        kind=kind,
        model_id="test/model",
        model_version="v1",
        dim=4,
        provider=provider,
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


def test_namespace_bulk_register_overwrites() -> None:
    registry = EmbeddingNamespaceRegistry()
    registry.bulk_register(
        {
            "single_vector.test.4.v1": _sample_config(provider="initial"),
        }
    )
    registry.bulk_register(
        {
            "single_vector.test.4.v1": _sample_config(provider="updated"),
        }
    )
    assert registry.get("single_vector.test.4.v1").provider == "updated"


def test_namespace_reset_clears_entries() -> None:
    registry = EmbeddingNamespaceRegistry()
    registry.register("single_vector.test.4.v1", _sample_config())
    registry.reset()
    with pytest.raises(ValueError):
        registry.get("single_vector.test.4.v1")


def test_namespace_contains_operator() -> None:
    registry = EmbeddingNamespaceRegistry()
    registry.register("single_vector.test.4.v1", _sample_config())
    assert "single_vector.test.4.v1" in registry
    assert "missing.namespace" not in registry


def test_namespace_get_error_lists_available() -> None:
    registry = EmbeddingNamespaceRegistry()
    registry.register("single_vector.test.4.v1", _sample_config())
    registry.register("sparse.test.8.v1", _sample_config(EmbeddingKind.SPARSE))
    with pytest.raises(ValueError) as exc:
        registry.get("unknown.namespace")
    message = str(exc.value)
    assert "single_vector.test.4.v1" in message
    assert "sparse.test.8.v1" in message


def test_namespace_list_sorted() -> None:
    registry = EmbeddingNamespaceRegistry()
    registry.register("single_vector.b.4.v1", _sample_config())
    registry.register("single_vector.a.4.v1", _sample_config())
    assert registry.list_namespaces() == [
        "single_vector.a.4.v1",
        "single_vector.b.4.v1",
    ]


def test_namespace_list_by_kind_filters() -> None:
    registry = EmbeddingNamespaceRegistry()
    registry.register("single_vector.a.4.v1", _sample_config())
    registry.register("sparse.b.8.v1", _sample_config(EmbeddingKind.SPARSE))
    assert registry.list_by_kind(EmbeddingKind.SPARSE) == ["sparse.b.8.v1"]


def test_load_namespace_configs_handles_missing_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "missing"
    configs = load_namespace_configs(empty_dir)
    assert configs == {}


def test_namespace_config_round_trip(tmp_path: Path) -> None:
    namespace_dir = tmp_path / "namespaces"
    namespace_dir.mkdir()
    payload = {
        "name": "roundtrip",
        "kind": "single_vector",
        "model_id": "demo/model",
        "model_version": "v1",
        "dim": 32,
        "provider": "demo",
        "parameters": {"max_tokens": 1024},
        "endpoint": "http://localhost:8100/v1",
    }
    (namespace_dir / "single_vector.roundtrip.32.v1.yaml").write_text(json.dumps(payload))
    configs = load_namespace_configs(namespace_dir)
    registry = EmbeddingNamespaceRegistry()
    registry.bulk_register(configs)
    config = registry.get("single_vector.roundtrip.32.v1")
    assert config.parameters["max_tokens"] == 1024
    assert (
        config.to_embedder_config("single_vector.roundtrip.32.v1").parameters["endpoint"]
        == "http://localhost:8100/v1"
    )


def test_load_namespace_configs_from_aggregated_file(tmp_path: Path) -> None:
    directory = tmp_path / "namespaces"
    directory.mkdir()
    aggregated = tmp_path / "namespaces.yaml"
    aggregated.write_text(
        json.dumps(
            {
                "namespaces": {
                    "single_vector.demo.8.v1": {
                        "name": "demo",
                        "kind": "single_vector",
                        "model_id": "demo/model",
                        "provider": "demo-provider",
                        "dim": 8,
                    }
                }
            }
        )
    )
    configs = load_namespace_configs(directory)
    assert "single_vector.demo.8.v1" in configs
    assert configs["single_vector.demo.8.v1"].model_id == "demo/model"
