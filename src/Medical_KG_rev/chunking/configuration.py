"""Configuration models for the modular chunking system."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from .exceptions import ChunkerConfigurationError, ProfileNotFoundError
from .models import ChunkerConfig, Granularity


class ChunkerSettings(BaseModel):
    """Configuration payload for a single chunker instance."""

    strategy: str
    granularity: Granularity | None = Field(default=None)
    params: dict[str, Any] = Field(default_factory=dict)

    def to_config(self) -> ChunkerConfig:
        return ChunkerConfig(
            name=self.strategy,
            granularity=self.granularity,
            params=self.params,
        )


class ChunkingProfile(BaseModel):
    """Configuration for a multi-granularity profile."""

    primary: ChunkerSettings
    auxiliaries: list[ChunkerSettings] = Field(default_factory=list)
    enable_multi_granularity: bool = Field(default=True)

    def all_chunkers(self) -> list[ChunkerSettings]:
        if not self.enable_multi_granularity:
            return [self.primary]
        return [self.primary, *self.auxiliaries]


class ChunkingConfig(BaseModel):
    """Top-level chunking configuration supporting multiple profiles."""

    default_profile: str = Field(default="default")
    profiles: dict[str, ChunkingProfile] = Field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: Path | None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> ChunkingConfig:
        environment = env or os.environ
        override_path = environment.get("CHUNKING_CONFIG_PATH")
        effective_path = Path(override_path) if override_path else path
        if effective_path is None:
            effective_path = DEFAULT_CONFIG_PATH
        if not effective_path.exists():
            raise ChunkerConfigurationError(
                f"Chunking configuration not found at '{effective_path}'"
            )
        data = yaml.safe_load(effective_path.read_text()) or {}
        default_profile = environment.get("CHUNKING_DEFAULT_PROFILE")
        if default_profile:
            data.setdefault("default_profile", default_profile)
            data["default_profile"] = default_profile
        overrides = environment.get("CHUNKING_CONFIG_OVERRIDES")
        if overrides:
            try:
                override_data = yaml.safe_load(overrides) or {}
            except yaml.YAMLError as exc:  # pragma: no cover - invalid override is rare
                raise ChunkerConfigurationError("Invalid CHUNKING_CONFIG_OVERRIDES payload") from exc
            data = _deep_merge(data, override_data)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - pydantic formatting tested elsewhere
            raise ChunkerConfigurationError(str(exc)) from exc

    def profile_for_source(self, source: str | None) -> ChunkingProfile:
        available = tuple(self.profiles.keys())
        if source:
            try:
                return self.profiles[source]
            except KeyError as exc:
                raise ProfileNotFoundError(source, available) from exc
        try:
            return self.profiles[self.default_profile]
        except KeyError as exc:
            raise ChunkerConfigurationError(
                f"Default chunking profile '{self.default_profile}' is not defined"
            ) from exc


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "chunking.yaml"


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
