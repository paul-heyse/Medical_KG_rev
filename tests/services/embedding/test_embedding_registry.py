from __future__ import annotations

from pathlib import Path

import yaml

from Medical_KG_rev.services.embedding.registry import EmbeddingModelRegistry
from Medical_KG_rev.services.embedding.service import EmbeddingVector


def _write_config(path: Path, namespace: str, *, dim: int, model_name: str) -> None:
    payload = {
        "active_namespaces": [namespace],
        "namespaces": {
            namespace: {
                "name": model_name,
                "provider": "mock-provider",
                "kind": "single_vector",
                "model_id": model_name,
                "model_version": "v1",
                "dim": dim,
                "normalize": True,
                "batch_size": 4,
            }
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_registry_reload_refreshes_namespaces(tmp_path) -> None:
    config_path = Path(tmp_path) / "embeddings.yaml"
    original_ns = "single_vector.mock_model.4.v1"
    updated_ns = "single_vector.new_model.8.v1"

    _write_config(config_path, original_ns, dim=4, model_name="mock-model")
    registry = EmbeddingModelRegistry(config_path=str(config_path))

    assert {cfg.namespace for cfg in registry.active_configs()} == {original_ns}
    assert set(registry.namespace_manager.namespaces().keys()) == {original_ns}

    _write_config(config_path, updated_ns, dim=8, model_name="new-model")
    registry.reload()

    assert {cfg.namespace for cfg in registry.active_configs()} == {updated_ns}
    assert set(registry.namespace_manager.namespaces().keys()) == {updated_ns}


def test_embedding_vector_values_returns_copy() -> None:
    vector = EmbeddingVector(
        id="chunk-1",
        model="mock",
        namespace="single_vector.mock_model.4.v1",
        kind="single_vector",
        vectors=[[1.0, 2.0, 3.0, 4.0]],
        terms=None,
        dimension=4,
        metadata={},
    )

    values = vector.values
    assert values == [1.0, 2.0, 3.0, 4.0]
    assert vector.vectors is not None
    values[0] = 99.0
    assert vector.vectors[0][0] == 1.0
