"""Configuration models for the modular chunking system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from .models import ChunkerConfig, Granularity
from .exceptions import ChunkerConfigurationError


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
    def load(cls, path: Path) -> ChunkingConfig:
        if not path.exists():
            raise ChunkerConfigurationError(f"Chunking configuration not found at '{path}'")
        data = yaml.safe_load(path.read_text()) or {}
        try:
            return cls.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - pydantic formatting tested elsewhere
            raise ChunkerConfigurationError(str(exc)) from exc

    def profile_for_source(self, source: str | None) -> ChunkingProfile:
        if source and source in self.profiles:
            return self.profiles[source]
        try:
            return self.profiles[self.default_profile]
        except KeyError as exc:
            raise ChunkerConfigurationError(
                f"Default chunking profile '{self.default_profile}' is not defined"
            ) from exc


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "chunking.yaml"
