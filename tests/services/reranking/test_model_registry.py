from __future__ import annotations

from pathlib import Path

import yaml

from Medical_KG_rev.services.reranking.model_registry import RerankerModelRegistry


def _write_config(tmp_path: Path) -> Path:
    config = {
        "default": "alpha",
        "cache_dir": str(tmp_path / "cache"),
        "models": {
            "alpha": {
                "reranker_id": "cross_encoder:bge",
                "model_id": "alpha/model",
                "version": "v1.0",
            },
            "beta": {
                "reranker_id": "cross_encoder:minilm",
                "model_id": "beta/model",
                "version": "v2.0",
            },
        },
    }
    config_path = tmp_path / "models.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_registry_ensure_creates_manifest(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    registry = RerankerModelRegistry(config_path=config_path)

    handle = registry.ensure()

    assert handle.path.exists()
    assert (handle.path / "manifest.json").exists()
    assert handle.model.key == "alpha"


def test_registry_resolves_by_model_id(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    registry = RerankerModelRegistry(config_path=config_path)

    key = registry.resolve_key("beta/model")
    assert key == "beta"

    handle = registry.ensure(key)
    assert handle.model.reranker_id == "cross_encoder:minilm"


def test_registry_unknown_model_falls_back(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    registry = RerankerModelRegistry(config_path=config_path)

    key = registry.resolve_key("does-not-exist")
    assert key == "alpha"

    handle = registry.ensure("does-not-exist")
    assert handle.model.key == "alpha"
