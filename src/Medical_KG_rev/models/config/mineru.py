"""Pydantic models for MinerU split-container configuration."""

from pydantic import BaseModel, Field, HttpUrl


class VLLMServerConfig(BaseModel):
    """vLLM server configuration."""

    enabled: bool = True
    base_url: HttpUrl = Field(default="http://vllm-server:8000")
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    health_check_interval_seconds: int = Field(default=30, ge=10)
    connection_timeout_seconds: int = Field(default=300, ge=30)


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    enabled: bool = True
    failure_threshold: int = Field(default=5, ge=1)
    recovery_timeout_seconds: int = Field(default=60, ge=10)
    success_threshold: int = Field(default=2, ge=1)


class HTTPClientConfig(BaseModel):
    """HTTP client configuration."""

    connection_pool_size: int = Field(default=10, ge=1)
    keepalive_connections: int = Field(default=5, ge=1)
    timeout_seconds: int = Field(default=300, ge=30)
    retry_attempts: int = Field(default=3, ge=0)
    retry_backoff_multiplier: float = Field(default=1.0, ge=0.1)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)


class WorkersConfig(BaseModel):
    """Worker pool configuration."""

    count: int = Field(default=8, ge=1)
    backend: str = Field(default="vlm-http-client")
    cpu_per_worker: int = Field(default=2, ge=1)
    memory_per_worker_gb: int = Field(default=4, ge=2)
    batch_size: int = Field(default=4, ge=1)
    timeout_seconds: int = Field(default=300, ge=30)


class MinerUConfig(BaseModel):
    """Complete MinerU configuration."""

    deployment_mode: str = Field(default="split-container")
    vllm_server: VLLMServerConfig = Field(default_factory=VLLMServerConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    http_client: HTTPClientConfig = Field(default_factory=HTTPClientConfig)
