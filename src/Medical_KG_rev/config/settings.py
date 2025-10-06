"""Configuration system for the foundation layer."""
from __future__ import annotations

import json
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environments supported by the platform."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class TelemetrySettings(BaseModel):
    """Configuration block for OpenTelemetry export."""

    exporter: str = Field(default="console", description="Target exporter type")
    endpoint: Optional[str] = Field(default=None, description="Exporter endpoint")
    sample_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class VaultSettings(BaseModel):
    """Settings for the optional HashiCorp Vault integration."""

    enabled: bool = False
    address: Optional[str] = None
    token: Optional[str] = None
    namespace: Optional[str] = None


class FeatureFlagSettings(BaseModel):
    """Dynamic feature flag configuration."""

    flags: Dict[str, bool] = Field(default_factory=dict)

    def is_enabled(self, name: str) -> bool:
        return self.flags.get(name.lower(), False)


class AppSettings(BaseSettings):
    """Top-level application settings."""

    environment: Environment = Environment.DEV
    debug: bool = False
    service_name: str = "medical-kg"
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    vault: VaultSettings = Field(default_factory=VaultSettings)
    feature_flags: FeatureFlagSettings = Field(default_factory=FeatureFlagSettings)
    domains_config_path: Optional[Path] = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="MK_", env_nested_delimiter="__")


ENVIRONMENT_DEFAULTS: Mapping[Environment, Dict[str, Any]] = {
    Environment.DEV: {"debug": True, "telemetry": {"exporter": "console"}},
    Environment.STAGING: {"telemetry": {"exporter": "otlp", "sample_ratio": 0.25}},
    Environment.PROD: {"telemetry": {"exporter": "otlp", "sample_ratio": 0.05}},
}


class SecretResolver:
    """Simple secret resolution utility with Vault integration."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client = None
        if settings.vault.enabled and settings.vault.address:
            try:
                import hvac  # type: ignore

                self._client = hvac.Client(
                    url=settings.vault.address,
                    token=settings.vault.token,
                    namespace=settings.vault.namespace,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                raise RuntimeError("Failed to initialise Vault client") from exc

    def get_secret(self, path: str) -> Dict[str, Any]:
        """Resolve secret either from Vault or environment."""

        if self._client is not None:
            response = self._client.secrets.kv.v2.read_secret_version(path=path)
            return response["data"]["data"]
        env_key = path.upper().replace("/", "_")
        raw = os.getenv(env_key)
        if raw is None:
            raise KeyError(f"Secret '{path}' not found in environment")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"value": raw}


def load_settings(environment: Optional[str] = None) -> AppSettings:
    """Load application settings with environment specific defaults applied."""

    env_value = (environment or os.getenv("MK_ENV", "dev")).lower()
    env = Environment(env_value)
    defaults = ENVIRONMENT_DEFAULTS.get(env, {})
    try:
        base_settings = AppSettings()
    except ValidationError as err:
        raise RuntimeError(f"Invalid configuration: {err}") from err
    merged = base_settings.model_dump()
    merged.update(defaults)
    merged["environment"] = env
    return AppSettings.model_validate(merged)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Cached accessor used by production code."""

    return load_settings()
