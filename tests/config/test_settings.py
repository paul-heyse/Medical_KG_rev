import json
from pathlib import Path

import pytest
import yaml

from Medical_KG_rev.config import (
    DomainRegistry,
    FeatureFlagSettings,
    SecretResolver,
    load_settings,
)


def test_load_settings_defaults(monkeypatch):
    monkeypatch.delenv("MK_ENV", raising=False)
    settings = load_settings()
    assert settings.environment.value == "dev"
    assert settings.telemetry.exporter == "console"


def test_load_settings_explicit_environment():
    settings = load_settings("staging")
    assert settings.environment.value == "staging"
    assert settings.telemetry.sample_ratio == 0.25


def test_secret_resolver_env(monkeypatch):
    monkeypatch.setenv("SECRET_PATH", json.dumps({"token": "value"}))
    settings = load_settings()
    resolver = SecretResolver(settings)
    secret = resolver.get_secret("secret/path")
    assert secret["token"] == "value"

    with pytest.raises(KeyError):
        resolver.get_secret("missing/path")


def test_domain_registry(tmp_path: Path):
    config = {"domains": [{"id": "medical", "description": "Medical", "adapters": {"a": "b"}}]}
    path = tmp_path / "domains.yaml"
    path.write_text(yaml.safe_dump(config))
    registry = DomainRegistry.from_path(path)
    assert registry.get("medical").description == "Medical"
    with pytest.raises(KeyError):
        registry.get("unknown")


def test_feature_flag_settings_lookup():
    settings = FeatureFlagSettings(
        pdf_processing_backend="docling_vlm",
        retrieval_backend="splade",
        docling_rollout_percentage=25,
        retrieval_rollout_percentage=75,
        flags={"feature": True},
    )
    assert settings.is_enabled("FEATURE")
    assert settings.is_enabled("pdf_processing_backend:docling_vlm")
    assert settings.is_enabled("retrieval_backend:splade")
    assert not settings.is_enabled("retrieval_backend:bm25")
    assert settings.selected_pdf_backend() == "docling_vlm"
    assert settings.selected_retrieval_backend() == "splade"
    assert settings.docling_rollout_percentage == 25
    assert settings.retrieval_rollout_percentage == 75


def test_retrieval_settings_defaults():
    settings = load_settings()
    retrieval = settings.retrieval
    assert retrieval.default_backend == "hybrid"
    config = retrieval.as_config()
    assert config.default_backend == "hybrid"
    assert config.qwen3.embedding_dimension == 4096
    assert config.splade.max_tokens <= 512


def test_retrieval_settings_env_overrides(monkeypatch):
    monkeypatch.setenv("MK_RETRIEVAL__DEFAULT_BACKEND", "bm25")
    monkeypatch.setenv("MK_RETRIEVAL__QWEN3__EMBEDDING_DIMENSION", "2048")
    try:
        settings = load_settings()
    finally:
        monkeypatch.delenv("MK_RETRIEVAL__DEFAULT_BACKEND", raising=False)
        monkeypatch.delenv("MK_RETRIEVAL__QWEN3__EMBEDDING_DIMENSION", raising=False)
    assert settings.retrieval.default_backend == "bm25"
    assert settings.retrieval.qwen3.embedding_dimension == 2048
