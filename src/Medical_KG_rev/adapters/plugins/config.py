"""Configuration helpers for adapter plugins."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import hvac
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

import structlog

logger = structlog.get_logger(__name__)


class AdapterSettings(BaseSettings):
    """Top-level adapter settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MK_ADAPTER_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    timeout_seconds: int = Field(30, ge=1)
    rate_limit_per_second: float = Field(5.0, ge=0)
    retry_max_attempts: int = Field(3, ge=1)
    vault_url: str | None = Field(default=None)
    vault_token: SecretStr | None = Field(default=None)
    vault_mount_point: str = Field("secret")

    def documentation_schema(self) -> Mapping[str, Any]:
        return self.model_json_schema()

    @classmethod
    def load(cls) -> AdapterSettings:
        settings = cls()
        logger.debug(
            "adapter.settings.loaded", settings=settings.model_dump(exclude={"vault_token"})
        )
        return settings


class VaultSecretProvider:
    """Simple Vault client wrapper with caching and fallbacks."""

    def __init__(
        self,
        url: str,
        token: SecretStr,
        mount_point: str = "secret",
        cache_ttl: timedelta = timedelta(minutes=15),
    ) -> None:
        self._client = hvac.Client(url=url, token=token.get_secret_value())
        self._mount_point = mount_point
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[datetime, Mapping[str, Any]]] = {}

    def get(self, path: str) -> Mapping[str, Any]:
        cached = self._cache.get(path)
        if cached and cached[0] > datetime.now(UTC):
            return cached[1]
        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=self._mount_point
            )
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("adapter.vault.unavailable", path=path, exc_info=exc)
            return {}
        data = response.get("data", {}).get("data", {})
        self._cache[path] = (datetime.now(UTC) + self._cache_ttl, data)
        return data


class SettingsHotReloader:
    """Background watcher that refreshes adapter settings periodically."""

    def __init__(
        self, settings_factory: Callable[[], AdapterSettings], interval_seconds: float = 30.0
    ) -> None:
        self._settings_factory = settings_factory
        self._interval = interval_seconds
        self._current = settings_factory()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="adapter-settings-hot-reload", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def current(self) -> AdapterSettings:
        with self._lock:
            return self._current

    def _loop(self) -> None:  # pragma: no cover - background thread
        while not self._stop.wait(self._interval):
            try:
                settings = self._settings_factory()
            except ValidationError as exc:
                logger.error("adapter.settings.invalid", errors=exc.errors())
                continue
            with self._lock:
                self._current = settings
                logger.info("adapter.settings.reloaded")


def migrate_yaml_to_env(yaml_path: Path, env_prefix: str = "MK_ADAPTER_") -> dict[str, str]:
    """Convert legacy YAML configuration to environment variables."""
    import yaml

    with yaml_path.open("r", encoding="utf-8") as handle:
        payload: Mapping[str, Any] = yaml.safe_load(handle) or {}

    env_vars: dict[str, str] = {}
    for key, value in payload.items():
        env_key = f"{env_prefix}{key.upper()}"
        env_vars[env_key] = str(value)
    return env_vars


def apply_env_overrides(env_vars: Mapping[str, str]) -> None:
    for key, value in env_vars.items():
        os.environ[key] = value


@dataclass
class ConfigValidationResult:
    valid: bool
    errors: list[str]

    @classmethod
    def from_exception(cls, exc: ValidationError) -> ConfigValidationResult:
        return cls(False, [str(error["msg"]) for error in exc.errors()])


def validate_on_startup(
    settings_cls: type[AdapterSettings] = AdapterSettings,
) -> ConfigValidationResult:
    try:
        settings_cls.load()
    except ValidationError as exc:
        result = ConfigValidationResult.from_exception(exc)
        logger.error("adapter.settings.validation_failed", errors=result.errors)
        return result
    return ConfigValidationResult(True, [])
