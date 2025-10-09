"""Configuration system for the foundation layer."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environments supported by the platform."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class TelemetrySettings(BaseModel):
    """Configuration block for OpenTelemetry export."""

    exporter: str = Field(default="console", description="Target exporter type")
    endpoint: str | None = Field(default=None, description="Exporter endpoint")
    sample_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class LoggingSettings(BaseModel):
    """Structured logging configuration."""

    level: str = Field(default="INFO", description="Log level for application output")
    correlation_id_header: str = Field(
        default="X-Correlation-ID", description="Header used for trace correlation"
    )
    scrub_fields: Sequence[str] = Field(
        default_factory=lambda: ["password", "token", "secret", "authorization"],
        description="Fields that should be redacted in logs",
    )


class MetricsSettings(BaseModel):
    """Prometheus metrics configuration."""

    enabled: bool = True
    path: str = Field(default="/metrics", description="HTTP path for Prometheus metrics")


class SentrySettings(BaseModel):
    """Sentry error tracking configuration."""

    dsn: str | None = Field(default=None, description="Sentry DSN for reporting errors")
    traces_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    send_default_pii: bool = False
    environment: str | None = Field(default=None, description="Override environment tag")


class ObservabilitySettings(BaseModel):
    """Aggregate observability configuration."""

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    sentry: SentrySettings = Field(default_factory=SentrySettings)


class ObjectStorageSettings(BaseModel):
    """S3/MinIO object storage configuration."""

    bucket: str = Field(default="medical-kg-pdf", description="Target bucket for stored PDFs")
    key_prefix: str = Field(default="pdf", description="Base key prefix for stored artefacts")
    endpoint_url: AnyHttpUrl | None = Field(
        default="http://minio:9000",
        description="Custom endpoint URL (set to MinIO or S3-compatible service)",
    )
    region: str | None = Field(default=None, description="AWS region (leave blank for MinIO/dev)")
    access_key_id: SecretStr | None = Field(default=None, description="S3 access key ID")
    secret_access_key: SecretStr | None = Field(default=None, description="S3 secret access key")
    session_token: SecretStr | None = Field(default=None, description="Temporary session token")
    use_tls: bool = Field(default=False, description="Whether to require TLS for the endpoint")
    checksum_algorithm: Literal["sha256", "md5"] = Field(
        default="sha256", description="Checksum algorithm used for stored PDFs"
    )
    signed_url_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="TTL for presigned URLs handed to downstream systems",
    )


class RedisCacheSettings(BaseModel):
    """Redis cache configuration for PDF metadata."""

    url: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL (supports redis+tls scheme)",
    )
    password: SecretStr | None = Field(default=None, description="Redis password (if required)")
    use_tls: bool = Field(default=False, description="Enable TLS for Redis connections")
    key_prefix: str = Field(default="medical_kg:pdf", description="Key prefix for cached entries")
    max_connections: int = Field(default=128, ge=1, description="Redis connection pool limit")
    tls_cert_path: str | None = Field(
        default=None,
        description="Path to CA bundle when TLS is enabled (optional for dev)",
    )
    default_ttl_seconds: int = Field(
        default=24 * 60 * 60,
        ge=60,
        description="Default TTL for cached PDF metadata (in seconds)",
    )


class OpenAlexSettings(BaseModel):
    """Configuration for the OpenAlex adapter."""

    contact_email: str = Field(
        default="oss@medical-kg.local",
        description="Contact email used for OpenAlex polite pool compliance",
    )
    user_agent: str | None = Field(
        default=None,
        description="Custom User-Agent identifying this deployment",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=200,
        description="Maximum number of OpenAlex results to fetch per request",
    )
    requests_per_second: float = Field(
        default=5.0,
        gt=0,
        description="Polite pool target RPS used by rate limiter",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="HTTP timeout applied to OpenAlex requests",
    )


class MineruCircuitBreakerSettings(BaseModel):
    """Circuit breaker thresholds for the MinerU vLLM client."""

    enabled: bool = True
    failure_threshold: int = Field(default=5, ge=1)
    recovery_timeout_seconds: float = Field(default=60.0, ge=5.0)
    success_threshold: int = Field(default=2, ge=1)


class MineruHttpClientSettings(BaseModel):
    """HTTP client configuration for MinerU → vLLM communication."""

    connection_pool_size: int = Field(default=10, ge=1)
    keepalive_connections: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=300.0, ge=30.0)
    retry_attempts: int = Field(default=3, ge=0)
    retry_backoff_multiplier: float = Field(default=1.0, ge=0.1)
    circuit_breaker: MineruCircuitBreakerSettings = Field(
        default_factory=MineruCircuitBreakerSettings
    )


class MineruVllmServerSettings(BaseModel):
    """Configuration for the dedicated vLLM server."""

    enabled: bool = True
    base_url: AnyHttpUrl = Field(default="http://vllm-server:8000")
    model: str = Field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    health_check_interval_seconds: int = Field(default=30, ge=5)
    connection_timeout_seconds: float = Field(default=300.0, ge=30.0)


class MineruWorkerSettings(BaseModel):
    """CPU-oriented MinerU worker configuration."""

    count: int = Field(default=8, ge=1)
    backend: Literal["vlm-http-client"] = "vlm-http-client"
    cpu_per_worker: int = Field(default=2, ge=1)
    memory_per_worker_gb: int = Field(default=4, ge=1)
    batch_size: int = Field(default=4, ge=1)
    timeout_seconds: int = Field(default=300, ge=30)


class MineruSettings(BaseModel):
    """Top-level MinerU split-container configuration."""

    deployment_mode: Literal["split-container"] = "split-container"
    cli_command: str = Field(default="mineru", description="Path to MinerU CLI")
    expected_version: str = Field(default=">=2.5.4")
    vllm_server: MineruVllmServerSettings = Field(default_factory=MineruVllmServerSettings)
    workers: MineruWorkerSettings = Field(default_factory=MineruWorkerSettings)
    http_client: MineruHttpClientSettings = Field(default_factory=MineruHttpClientSettings)

    def cli_timeout_seconds(self) -> int:
        """Expose worker timeout for CLI invocations."""

        return self.workers.timeout_seconds


class VaultSettings(BaseModel):
    """Settings for the optional HashiCorp Vault integration."""

    enabled: bool = False
    address: str | None = None
    token: str | None = None
    namespace: str | None = None


class FeatureFlagSettings(BaseModel):
    """Dynamic feature flag configuration."""

    flags: dict[str, bool] = Field(default_factory=dict)

    def is_enabled(self, name: str) -> bool:
        return self.flags.get(name.lower(), False)


class OAuthProvider(str, Enum):
    """Supported OAuth providers for configuration hints."""

    KEYCLOAK = "keycloak"
    AUTH0 = "auth0"
    CUSTOM = "custom"


class OAuthClientSettings(BaseModel):
    """OAuth 2.0 client credentials configuration."""

    provider: OAuthProvider = OAuthProvider.KEYCLOAK
    issuer: str = Field(..., description="Expected issuer claim")
    audience: str = Field(..., description="Expected audience claim")
    token_url: str = Field(..., description="OAuth token endpoint")
    jwks_url: str = Field(..., description="JWKS endpoint for signature validation")
    client_id: str = Field(..., description="Service client identifier")
    client_secret: SecretStr = Field(..., description="Service client secret")
    scopes: Sequence[str] = Field(
        default_factory=lambda: ["ingest:write", "kg:read"]
    )  # default scopes

    @model_validator(mode="before")
    @classmethod
    def _coerce_scopes(cls, values: dict[str, Any]) -> dict[str, Any]:
        scopes = values.get("scopes")
        if isinstance(scopes, str):
            candidate = scopes.strip()
            if candidate.startswith("[") or candidate.startswith("{"):
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    parsed = None
                else:
                    if isinstance(parsed, list):
                        values["scopes"] = [str(item) for item in parsed]
                        return values
            values["scopes"] = [
                item.strip() for item in scopes.replace(",", " ").split() if item.strip()
            ]
        elif isinstance(scopes, dict):
            values["scopes"] = [str(item) for item in scopes.values()]
        elif isinstance(scopes, Sequence):
            flattened: list[str] = []
            for item in scopes:
                if isinstance(item, str):
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, list):
                            flattened.extend(str(entry) for entry in parsed)
                            continue
                    except json.JSONDecodeError:
                        pass
                    flattened.append(item)
                else:
                    flattened.append(str(item))
            values["scopes"] = flattened
        return values


class RateLimitSettings(BaseModel):
    """Token bucket configuration for API rate limiting."""

    requests_per_minute: int = Field(default=60, ge=1, description="Default per-subject RPM")
    burst: int = Field(default=10, ge=1, description="Token bucket burst capacity")
    endpoint_overrides: dict[str, int] = Field(
        default_factory=dict, description="Endpoint specific RPM overrides"
    )


class EndpointCachePolicy(BaseModel):
    """Cache policy per endpoint for ETag middleware."""

    ttl: int = Field(default=60, ge=0)
    scope: Literal["public", "private", "no-store"] = "private"
    vary: Sequence[str] = Field(default_factory=lambda: ["Accept"])
    etag: bool = True
    last_modified: bool = True

    @model_validator(mode="before")
    @classmethod
    def _coerce_vary(cls, values: dict[str, Any]) -> dict[str, Any]:
        vary = values.get("vary")
        if isinstance(vary, str):
            values["vary"] = [
                item.strip() for item in vary.replace(",", " ").split() if item.strip()
            ]
        return values


class CachingSettings(BaseModel):
    """Configuration for HTTP caching policies."""

    default: EndpointCachePolicy = Field(default_factory=EndpointCachePolicy)
    endpoints: dict[str, EndpointCachePolicy] = Field(default_factory=dict)

    def policy_for(self, path: str) -> EndpointCachePolicy:
        return self.endpoints.get(path, self.default)


class APIKeyRecord(BaseModel):
    """Metadata associated with an API key stored in configuration or Vault."""

    hashed_secret: str = Field(..., description="Hashed API key")
    tenant_id: str = Field(..., description="Tenant the key is scoped to")
    scopes: Sequence[str] = Field(default_factory=list)
    rotated_at: str | None = Field(default=None, description="ISO timestamp of last rotation")

    @model_validator(mode="before")
    @classmethod
    def _coerce_scopes(cls, values: dict[str, Any]) -> dict[str, Any]:
        scopes = values.get("scopes")
        if isinstance(scopes, str):
            candidate = scopes.strip()
            if candidate.startswith("[") or candidate.startswith("{"):
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    parsed = None
                else:
                    if isinstance(parsed, list):
                        values["scopes"] = [str(item) for item in parsed]
                        return values
            values["scopes"] = [
                item.strip() for item in scopes.replace(",", " ").split() if item.strip()
            ]
        elif isinstance(scopes, dict):
            values["scopes"] = [str(item) for item in scopes.values()]
        return values


class APIKeySettings(BaseModel):
    """API key management configuration."""

    enabled: bool = True
    hashing_algorithm: str = Field(default="sha256")
    secret_store_path: str | None = Field(
        default="security/api-keys",
        description="Path used with secret resolver when enabled",
    )
    keys: dict[str, APIKeyRecord] = Field(default_factory=dict)


class SecurityHeaderSettings(BaseModel):
    """HTTP security header configuration."""

    hsts_max_age: int = Field(default=63072000, description="HSTS max-age in seconds")
    content_security_policy: str = Field(
        default="default-src 'self'",
        description="CSP applied to responses",
    )
    frame_options: str = Field(default="DENY")


class CORSSecuritySettings(BaseModel):
    """CORS configuration consumed by the FastAPI application."""

    allow_origins: Sequence[str] = Field(default_factory=lambda: ["https://localhost"])
    allow_methods: Sequence[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    allow_headers: Sequence[str] = Field(
        default_factory=lambda: ["Authorization", "Content-Type", "X-API-Key"]
    )

    @model_validator(mode="before")
    @classmethod
    def _normalise_sequences(cls, values: dict[str, Any]) -> dict[str, Any]:
        for field in ("allow_origins", "allow_methods", "allow_headers"):
            current = values.get(field)
            if isinstance(current, str):
                values[field] = [
                    item.strip() for item in current.replace(",", " ").split() if item.strip()
                ]
            elif isinstance(current, dict):
                values[field] = [str(item) for item in current.values()]
        return values


class SecuritySettings(BaseModel):
    """Aggregate security configuration."""

    oauth: OAuthClientSettings
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    api_keys: APIKeySettings = Field(default_factory=APIKeySettings)
    headers: SecurityHeaderSettings = Field(default_factory=SecurityHeaderSettings)
    cors: CORSSecuritySettings = Field(default_factory=CORSSecuritySettings)
    enforce_https: bool = True

    @model_validator(mode="after")
    def validate_cors(self) -> SecuritySettings:
        if not self.cors.allow_origins:
            raise ValueError("At least one CORS origin must be configured")
        return self


class RerankerModelSettings(BaseModel):
    """Configuration block describing the active reranker implementation."""

    reranker_id: str = Field(default="cross_encoder:bge")
    model: str = Field(default="BAAI/bge-reranker-v2-m3")
    batch_size: int = Field(default=32, ge=1, le=256)
    device: str = Field(default="cpu")
    precision: Literal["fp32", "fp16", "int8"] = "fp16"
    onnx_optimize: bool = False
    quantization: Literal["int8", "fp16"] | None = None
    requires_gpu: bool = False

    @model_validator(mode="after")
    def validate_model_availability(self) -> "RerankerModelSettings":
        from Medical_KG_rev.services.reranking.errors import UnknownRerankerError
        from Medical_KG_rev.services.reranking.factory import RerankerFactory

        factory = RerankerFactory()
        if self.reranker_id not in factory.available:
            raise ValueError(
                f"Reranker '{self.reranker_id}' is not registered. Available: {factory.available}"
            )
        try:
            reranker = factory.resolve(self.reranker_id)
        except UnknownRerankerError as exc:  # pragma: no cover - defensive
            raise ValueError(str(exc)) from exc
        if self.requires_gpu:
            try:
                import torch

                if not torch.cuda.is_available():  # type: ignore[attr-defined]
                    raise ValueError("GPU is required for the configured reranker but unavailable")
            except Exception as exc:  # pragma: no cover - torch optional
                raise ValueError("GPU is required for the configured reranker but unavailable") from exc
            if not getattr(reranker, "requires_gpu", False):
                raise ValueError(
                    f"Reranker '{self.reranker_id}' does not support GPU execution"
                )
        return self


class FusionAlgorithmSettings(BaseModel):
    """Fusion algorithm configuration."""

    strategy: Literal["rrf", "weighted", "learned"] = "rrf"
    rrf_k: int = Field(default=60, ge=1, le=1000)
    weights: dict[str, float] = Field(default_factory=dict)
    normalization: Literal["min_max", "z_score", "softmax"] = "min_max"
    deduplicate: bool = True
    aggregation: Literal["max", "mean", "sum"] = "max"

    @model_validator(mode="after")
    def validate_weights(self) -> FusionAlgorithmSettings:
        if self.strategy in {"weighted", "learned"}:
            if not self.weights:
                raise ValueError("Fusion weights are required for weighted strategies")
            total = sum(float(value) for value in self.weights.values())
            if total <= 0:
                raise ValueError("Fusion weights must sum to a positive value")
        return self


class PipelineStageSettings(BaseModel):
    """Two-stage pipeline sizing configuration."""

    retrieve_candidates: int = Field(default=1000, ge=10, le=5000)
    rerank_candidates: int = Field(default=100, ge=1, le=1000)
    return_top_k: int = Field(default=10, ge=1, le=200)


class RerankingSettings(BaseModel):
    """Top level reranking configuration exposed in settings."""

    enabled: bool = True
    cache_ttl: int = Field(default=3600, ge=0)
    circuit_breaker_failures: int = Field(default=5, ge=1, le=50)
    circuit_breaker_reset: float = Field(default=30.0, ge=1.0, le=600.0)
    model: RerankerModelSettings = Field(default_factory=RerankerModelSettings)
    fusion: FusionAlgorithmSettings = Field(default_factory=FusionAlgorithmSettings)
    pipeline: PipelineStageSettings = Field(default_factory=PipelineStageSettings)


class EmbeddingPolicyRuntimeSettings(BaseModel):
    """Namespace policy runtime configuration exposed via settings."""

    cache_ttl_seconds: float = Field(default=60.0, ge=0.0)
    max_cache_entries: int = Field(default=512, ge=1)
    dry_run: bool = False


class EmbeddingPersisterRuntimeSettings(BaseModel):
    """Persistence backend selection and tuning parameters."""

    backend: Literal["vector_store", "database", "dry_run", "hybrid"] = "vector_store"
    cache_limit: int = Field(default=256, ge=0)
    hybrid_backends: dict[str, Literal["vector_store", "database", "dry_run"]] = Field(
        default_factory=dict
    )


class EmbeddingRuntimeSettings(BaseModel):
    """Aggregated embedding runtime configuration."""

    policy: EmbeddingPolicyRuntimeSettings = Field(default_factory=EmbeddingPolicyRuntimeSettings)
    persister: EmbeddingPersisterRuntimeSettings = Field(
        default_factory=EmbeddingPersisterRuntimeSettings
    )


class ObjectStorageSettings(BaseModel):
    """S3/MinIO object storage configuration."""

    bucket: str = Field(default="medical-kg-pdf", description="S3 bucket name for PDF storage")
    region: str = Field(default="us-east-1", description="AWS region for S3 operations")
    endpoint_url: str | None = Field(default=None, description="Custom S3 endpoint (e.g., MinIO)")
    access_key_id: str | None = Field(default=None, description="AWS access key ID")
    secret_access_key: SecretStr | None = Field(default=None, description="AWS secret access key")
    session_token: SecretStr | None = Field(default=None, description="AWS session token")
    use_tls: bool = Field(default=True, description="Use TLS for S3 connections")
    tls_cert_path: str | None = Field(default=None, description="Path to TLS certificate")
    max_file_size: int = Field(default=100 * 1024 * 1024, description="Maximum file size in bytes (100MB)")
    key_prefix: str = Field(default="pdf", description="Key prefix for stored objects")


class RedisCacheSettings(BaseModel):
    """Redis cache configuration."""

    url: str = Field(default="redis://redis:6379/0", description="Redis connection URL")
    password: SecretStr | None = Field(default=None, description="Redis password")
    use_tls: bool = Field(default=False, description="Use TLS for Redis connections")
    tls_cert_path: str | None = Field(default=None, description="Path to TLS certificate")
    db_index: int = Field(default=0, ge=0, le=15, description="Redis database index")
    key_prefix: str = Field(default="medical-kg", description="Key prefix for cache entries")
    default_ttl: int = Field(default=3600, ge=0, description="Default TTL in seconds")
    max_connections: int = Field(default=10, ge=1, description="Maximum connection pool size")


def migrate_reranking_config(payload: Mapping[str, Any]) -> RerankingSettings:
    """Convert legacy reranking configuration dictionaries into the new schema."""

    migrated: dict[str, Any] = dict(payload)
    legacy_model = migrated.pop("model_name", None)
    if legacy_model and "model" not in migrated:
        migrated["model"] = {"model": legacy_model}
    fusion_strategy = migrated.pop("fusion_strategy", None)
    if fusion_strategy and "fusion" not in migrated:
        migrated["fusion"] = {"strategy": fusion_strategy}
    cache_ttl = migrated.pop("cacheTtl", None)
    if cache_ttl is not None and "cache_ttl" not in migrated:
        migrated["cache_ttl"] = cache_ttl
    return RerankingSettings(**migrated)


class AppSettings(BaseSettings):
    """Top-level application settings."""

    environment: Environment = Environment.DEV
    debug: bool = False
    service_name: str = "medical-kg"
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    mineru: MineruSettings = Field(default_factory=MineruSettings)
    vault: VaultSettings = Field(default_factory=VaultSettings)
    feature_flags: FeatureFlagSettings = Field(default_factory=FeatureFlagSettings)
    object_storage: ObjectStorageSettings = Field(default_factory=ObjectStorageSettings)
    redis_cache: RedisCacheSettings = Field(default_factory=RedisCacheSettings)
    openalex: OpenAlexSettings = Field(default_factory=OpenAlexSettings)
    domains_config_path: Path | None = Field(default=None)
    security: SecuritySettings = Field(
        default_factory=lambda: SecuritySettings(
            oauth=OAuthClientSettings(
                issuer="https://idp.local/realms/medical",  # sensible defaults for dev
                audience="medical-kg",
                token_url="https://idp.local/realms/medical/protocol/openid-connect/token",
                jwks_url="https://idp.local/realms/medical/protocol/openid-connect/certs",
                client_id="medical-gateway",
                client_secret=SecretStr("dev-secret"),
            )
        )
    )
    reranking: RerankingSettings = Field(default_factory=RerankingSettings)
    caching: CachingSettings = Field(
        default_factory=lambda: CachingSettings(
            default=EndpointCachePolicy(ttl=30, scope="private"),
            endpoints={
                "/v1/retrieve": EndpointCachePolicy(
                    ttl=300, scope="private", vary=["Accept", "Content-Type"]
                ),
                "/v1/search": EndpointCachePolicy(
                    ttl=300, scope="private", vary=["Accept", "Content-Type"]
                ),
                "/v1/jobs": EndpointCachePolicy(ttl=30, scope="private"),
                "/v1/jobs/{job_id}": EndpointCachePolicy(ttl=15, scope="private"),
            },
        )
    )
    embedding: EmbeddingRuntimeSettings = Field(default_factory=EmbeddingRuntimeSettings)
    object_storage: ObjectStorageSettings = Field(default_factory=ObjectStorageSettings)
    redis_cache: RedisCacheSettings = Field(default_factory=RedisCacheSettings)

    model_config = SettingsConfigDict(env_prefix="MK_", env_nested_delimiter="__")


ENVIRONMENT_DEFAULTS: Mapping[Environment, dict[str, Any]] = {
    Environment.DEV: {
        "debug": True,
        "telemetry": {"exporter": "console"},
        "security": {"enforce_https": False},
        "object_storage": {
            "endpoint_url": "http://minio:9000",
            "bucket": "medical-kg-pdf",
            "use_tls": False,
        },
        "redis_cache": {
            "url": "redis://redis:6379/0",
            "use_tls": False,
        },
    },
    Environment.STAGING: {
        "telemetry": {"exporter": "otlp", "sample_ratio": 0.25},
        "object_storage": {"use_tls": True},
        "redis_cache": {"use_tls": True},
    },
    Environment.PROD: {
        "telemetry": {"exporter": "otlp", "sample_ratio": 0.05},
        "object_storage": {"use_tls": True},
        "redis_cache": {"use_tls": True},
    },
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

    def get_secret(self, path: str) -> dict[str, Any]:
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


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            target[key] = _deep_update(dict(current), value)
        else:
            target[key] = value
    return target


def load_settings(environment: str | None = None) -> AppSettings:
    """Load application settings with environment specific defaults applied."""

    env_value = (environment or os.getenv("MK_ENV", "dev")).lower()
    env = Environment(env_value)
    defaults = ENVIRONMENT_DEFAULTS.get(env, {})
    try:
        base_settings = AppSettings()
    except ValidationError as err:
        raise RuntimeError(f"Invalid configuration: {err}") from err
    merged = base_settings.model_dump()
    merged = _deep_update(merged, defaults)
    merged["environment"] = env
    return AppSettings.model_validate(merged)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Cached accessor used by production code."""

    return load_settings()
