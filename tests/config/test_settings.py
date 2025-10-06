import json
from pathlib import Path

import pytest
import yaml

from Medical_KG_rev.config import DomainRegistry, FeatureFlagSettings, SecretResolver, load_settings


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
    settings = FeatureFlagSettings(flags={"feature": True})
    assert settings.is_enabled("FEATURE")
