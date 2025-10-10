"""Configuration helpers for the Docling Gemma3 12B vision-language model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover - optional dependency for YAML loading
    import yaml
except Exception:  # pragma: no cover - fall back when PyYAML missing
    yaml = None  # type: ignore[assignment]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "docling_vlm.yaml"
DEFAULT_MODEL_PATH = Path("/models/docling-vlm")
DEFAULT_MODEL_NAME = "google/gemma-3-12b-it"


@dataclass(slots=True)
class DoclingVLMConfig:
    """Runtime configuration for the Docling Gemma3 VLM service."""

    model_path: Path = DEFAULT_MODEL_PATH
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 8
    timeout_seconds: int = 300
    retry_attempts: int = 3
    gpu_memory_fraction: float = 0.95
    max_model_len: int = 4096
    device: str = "cuda"
    warmup_prompts: int = 1
    required_total_memory_mb: int = 24 * 1024

    def __post_init__(self) -> None:
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        self.validate()

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts cannot be negative")
        if not 0.0 < self.gpu_memory_fraction <= 1.0:
            raise ValueError("gpu_memory_fraction must be in (0, 1]")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.required_total_memory_mb <= 0:
            raise ValueError("required_total_memory_mb must be positive")

    def ensure_model_path(self) -> Path:
        self.model_path.mkdir(parents=True, exist_ok=True)
        return self.model_path

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DoclingVLMConfig:
        payload = dict(data)
        if "model_path" in payload and isinstance(payload["model_path"], str):
            payload["model_path"] = Path(payload["model_path"])
        return cls(**payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> DoclingVLMConfig:
        if not data:
            return cls()
        return cls.from_mapping(data)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> DoclingVLMConfig:
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if yaml is None or not config_path.exists():
            return cls()
        payload = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"Invalid Docling config structure in {config_path}")
        return cls.from_mapping(payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "max_model_len": self.max_model_len,
            "device": self.device,
            "warmup_prompts": self.warmup_prompts,
            "required_total_memory_mb": self.required_total_memory_mb,
        }


__all__ = ["DoclingVLMConfig", "DEFAULT_CONFIG_PATH", "DEFAULT_MODEL_PATH", "DEFAULT_MODEL_NAME"]
