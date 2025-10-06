"""Configuration system for the foundation layer."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator
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


class AppSettings(BaseSettings):
    """Top-level application settings."""

    environment: Environment = Environment.DEV
    debug: bool = False
    service_name: str = "medical-kg"
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    vault: VaultSettings = Field(default_factory=VaultSettings)
    feature_flags: FeatureFlagSettings = Field(default_factory=FeatureFlagSettings)
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

    model_config = SettingsConfigDict(env_prefix="MK_", env_nested_delimiter="__")


ENVIRONMENT_DEFAULTS: Mapping[Environment, dict[str, Any]] = {
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
    merged.update(defaults)
    merged["environment"] = env
    return AppSettings.model_validate(merged)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Cached accessor used by production code."""

    return load_settings()
