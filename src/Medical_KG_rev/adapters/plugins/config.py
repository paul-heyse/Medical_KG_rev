"""Configuration helpers for adapter plugins (placeholder)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class AdapterSettings:
    """Minimal settings container used during refactoring."""

    config: Mapping[str, Any] | None = None

    @classmethod
    def load(cls) -> "AdapterSettings":
        return cls()


def load_yaml_as_env_vars(path: str) -> Mapping[str, str]:
    return {}


def apply_env_overrides(env_vars: Mapping[str, str]) -> None:
    for _key, _value in env_vars.items():
        pass


@dataclass
class ConfigValidationResult:
    valid: bool
    errors: list[str]


def validate_on_startup(settings_cls: type[AdapterSettings] = AdapterSettings) -> ConfigValidationResult:
    return ConfigValidationResult(valid=True, errors=[])
