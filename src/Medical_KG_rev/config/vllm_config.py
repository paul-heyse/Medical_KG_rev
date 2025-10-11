"""Configuration loader for vLLM embedding service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback when PyYAML unavailable
    yaml = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - allow running without pydantic
    BaseModel = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


DEFAULT_VLLM_CONFIG = Path(__file__).resolve().parents[3] / "config" / "embedding" / "vllm.yaml"


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    return yaml.safe_load(path.read_text()) or {}


@dataclass(slots=True)
class VLLMServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8001
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 512
    dtype: str = "float16"


@dataclass(slots=True)
class VLLMModelConfig:
    name: str = "Qwen/Qwen2.5-Coder-1.5B"
    trust_remote_code: bool = True
    download_dir: str = "/models/qwen3-embedding"
    revision: str | None = "main"
    model_type: str = "vllm"  # "vllm" | "docling_vlm"


@dataclass(slots=True)
class VLLMBatchingConfig:
    max_batch_size: int = 64
    max_wait_time_ms: int = 50
    preferred_batch_size: int = 32


@dataclass(slots=True)
class VLLMHealthCheckConfig:
    enabled: bool = True
    gpu_check_interval_seconds: int = 30
    fail_fast_on_gpu_unavailable: bool = True


@dataclass(slots=True)
class VLLMLoggingConfig:
    level: str = "INFO"
    format: str = "json"


@dataclass(slots=True)
class VLLMGPUConfig:
    model_type: str = "vllm"  # "vllm" | "docling_vlm"
    gpu_memory_utilization: float = 0.8
    required_total_memory_mb: int | None = None
    docling_vlm_memory_mb: int = 24 * 1024  # 24GB for Docling VLM


@dataclass(slots=True)
class UnifiedGPUConfig:
    """Unified GPU configuration that works for both vLLM and Docling VLM."""

    model_type: str = "vllm"  # "vllm" | "docling_vlm"
    gpu_memory_utilization: float = 0.8
    required_total_memory_mb: int | None = None
    docling_vlm_memory_mb: int = 24 * 1024  # 24GB for Docling VLM

    def get_required_memory_mb(self) -> int:
        """Get the required memory based on model type."""
        if self.model_type == "docling_vlm":
            return self.docling_vlm_memory_mb
        return self.required_total_memory_mb or 8 * 1024  # Default 8GB for vLLM


@dataclass(slots=True)
class VLLMConfig:
    service: VLLMServiceConfig = field(default_factory=VLLMServiceConfig)
    model: VLLMModelConfig = field(default_factory=VLLMModelConfig)
    batching: VLLMBatchingConfig = field(default_factory=VLLMBatchingConfig)
    health_check: VLLMHealthCheckConfig = field(default_factory=VLLMHealthCheckConfig)
    logging: VLLMLoggingConfig = field(default_factory=VLLMLoggingConfig)
    gpu: VLLMGPUConfig = field(default_factory=VLLMGPUConfig)
    unified_gpu: UnifiedGPUConfig = field(default_factory=UnifiedGPUConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> VLLMConfig:
        service = VLLMServiceConfig(**data.get("service", {}))
        model = VLLMModelConfig(**data.get("model", {}))
        batching = VLLMBatchingConfig(**data.get("batching", {}))
        health = VLLMHealthCheckConfig(**data.get("health_check", {}))
        logging = VLLMLoggingConfig(**data.get("logging", {}))
        gpu = VLLMGPUConfig(**data.get("gpu", {}))
        unified_gpu = UnifiedGPUConfig(**data.get("unified_gpu", {}))
        return cls(
            service=service,
            model=model,
            batching=batching,
            health_check=health,
            logging=logging,
            gpu=gpu,
            unified_gpu=unified_gpu,
        )

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> VLLMConfig:
        config_path = Path(path) if path is not None else DEFAULT_VLLM_CONFIG
        data = _load_mapping(config_path)
        if BaseModel is not None:
            model = _VLLMConfigModel.model_validate(data)
            return cls.from_mapping(model.model_dump())
        return cls.from_mapping(data)


if BaseModel is not None:  # pragma: no cover - exercised when pydantic installed

    class _VLLMServiceModel(BaseModel):
        host: str = Field(default="0.0.0.0")
        port: int = Field(default=8001, ge=1)
        gpu_memory_utilization: float = Field(default=0.8, ge=0.0, le=1.0)
        max_model_len: int = Field(default=512, ge=1)
        dtype: str = Field(default="float16")

    class _VLLMModelModel(BaseModel):
        name: str = Field(default="Qwen/Qwen2.5-Coder-1.5B")
        trust_remote_code: bool = Field(default=True)
        download_dir: str = Field(default="/models/qwen3-embedding")
        revision: str | None = Field(default="main")
        model_type: str = Field(default="vllm")

    class _VLLMBatchingModel(BaseModel):
        max_batch_size: int = Field(default=64, ge=1)
        max_wait_time_ms: int = Field(default=50, ge=0)
        preferred_batch_size: int = Field(default=32, ge=1)

    class _VLLMHealthCheckModel(BaseModel):
        enabled: bool = Field(default=True)
        gpu_check_interval_seconds: int = Field(default=30, ge=1)
        fail_fast_on_gpu_unavailable: bool = Field(default=True)

    class _VLLMLoggingModel(BaseModel):
        level: str = Field(default="INFO")
        format: str = Field(default="json")

    class _VLLMGPUModel(BaseModel):
        model_type: str = Field(default="vllm")
        gpu_memory_utilization: float = Field(default=0.8, ge=0.0, le=1.0)
        required_total_memory_mb: int | None = Field(default=None, ge=1)
        docling_vlm_memory_mb: int = Field(default=24 * 1024, ge=1)

    class _UnifiedGPUModel(BaseModel):
        model_type: str = Field(default="vllm")
        gpu_memory_utilization: float = Field(default=0.8, ge=0.0, le=1.0)
        required_total_memory_mb: int | None = Field(default=None, ge=1)
        docling_vlm_memory_mb: int = Field(default=24 * 1024, ge=1)

    class _VLLMConfigModel(BaseModel):
        service: _VLLMServiceModel = Field(default_factory=_VLLMServiceModel)
        model: _VLLMModelModel = Field(default_factory=_VLLMModelModel)
        batching: _VLLMBatchingModel = Field(default_factory=_VLLMBatchingModel)
        health_check: _VLLMHealthCheckModel = Field(default_factory=_VLLMHealthCheckModel)
        logging: _VLLMLoggingModel = Field(default_factory=_VLLMLoggingModel)
        gpu: _VLLMGPUModel = Field(default_factory=_VLLMGPUModel)
        unified_gpu: _UnifiedGPUModel = Field(default_factory=_UnifiedGPUModel)


def load_vllm_config(path: str | Path | None = None) -> VLLMConfig:
    """Load the vLLM configuration from disk."""
    return VLLMConfig.from_yaml(path)


__all__ = [
    "DEFAULT_VLLM_CONFIG",
    "UnifiedGPUConfig",
    "VLLMBatchingConfig",
    "VLLMConfig",
    "VLLMGPUConfig",
    "VLLMHealthCheckConfig",
    "VLLMLoggingConfig",
    "VLLMModelConfig",
    "VLLMServiceConfig",
    "load_vllm_config",
]
