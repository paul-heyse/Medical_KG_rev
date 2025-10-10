"""Configuration helpers for the Docling Gemma3 VLM integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field, PositiveInt, model_validator
from pydantic_settings import SettingsConfigDict


class DoclingVLMConfig(BaseModel):
    """Configuration block describing the Docling Gemma3 VLM runtime."""

    model_config = SettingsConfigDict(env_prefix="DOCLING_VLM_", env_nested_delimiter="__")

    model_path: Path = Field(
        default=Path("/models/gemma3-12b"),
        description="Filesystem path that hosts the Gemma3 checkpoint.",
    )
    revision: str | None = Field(
        default=None,
        description="Optional model revision to fetch from HuggingFace.",
    )
    batch_size: PositiveInt = Field(
        default=4,
        description="Number of PDF pages processed in a single Docling batch.",
    )
    timeout_seconds: PositiveInt = Field(
        default=300,
        description="Maximum amount of time allowed for a single PDF job.",
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts for transient Docling failures.",
    )
    gpu_memory_fraction: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Fraction of detected GPU memory reserved for the Gemma3 model.",
    )
    max_model_len: PositiveInt = Field(
        default=4096,
        description="Maximum context window supported by the Gemma3 model.",
    )
    min_gpu_memory_mb: PositiveInt = Field(
        default=24_000,
        description="Absolute minimum GPU memory required to schedule inference.",
    )

    @model_validator(mode="after")
    def _validate_paths(self) -> Self:
        if not self.model_path:
            msg = "Docling model path cannot be empty"
            raise ValueError(msg)
        return self

    @property
    def resolved_model_path(self) -> Path:
        """Return the expanded absolute path to the Gemma3 model directory."""

        return self.model_path.expanduser().resolve()

    def required_gpu_memory_mb(self) -> int:
        """Compute the amount of GPU memory the runtime expects to reserve."""

        return max(
            int(self.min_gpu_memory_mb * self.gpu_memory_fraction),
            self.min_gpu_memory_mb,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DoclingVLMConfig":
        """Instantiate configuration from a dictionary payload."""

        return cls.model_validate(payload)

    @classmethod
    def from_yaml(cls, path: Path) -> "DoclingVLMConfig":
        """Instantiate configuration from a YAML file."""

        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):  # pragma: no cover - defensive guard
            msg = "Docling configuration YAML must define a mapping"
            raise ValueError(msg)
        return cls.from_dict(data)

