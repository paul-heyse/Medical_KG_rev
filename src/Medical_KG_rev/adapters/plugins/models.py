"""Pydantic models used by the adapter plugin framework."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, ClassVar, Mapping

from pydantic import BaseModel, Field, PositiveInt, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AdapterDomainLiteral = tuple[str, ...]


class AdapterDomain(str, Enum):
    """Supported adapter domains."""

    BIOMEDICAL = "biomedical"
    FINANCIAL = "financial"
    LEGAL = "legal"

    @classmethod
    def canonical(cls) -> AdapterDomainLiteral:
        return tuple(member.value for member in cls)


class Pagination(BaseModel):
    """Pagination parameters carried in adapter requests."""

    page_size: PositiveInt = Field(50, description="Number of items to fetch in a single request.")
    page_token: str | None = Field(
        default=None,
        description="Opaque token for fetching the next page of results.",
    )


class AdapterConfig(BaseSettings):
    """Base configuration for adapters using pydantic-settings."""

    model_config = SettingsConfigDict(
        env_prefix="MK_ADAPTER_",
        case_sensitive=False,
        extra="ignore",
    )

    timeout_seconds: PositiveInt = Field(
        30, description="HTTP timeout in seconds for upstream calls."
    )
    rate_limit_per_second: float = Field(
        5.0,
        ge=0,
        description="Maximum number of upstream calls allowed per second.",
    )
    retry_max_attempts: PositiveInt = Field(
        3,
        description="Maximum number of retry attempts executed by the resilience layer.",
    )

    def json_schema(self) -> Mapping[str, Any]:
        """Return JSON schema for documentation purposes."""

        return self.model_json_schema()

    @classmethod
    def from_env(cls) -> "AdapterConfig":
        """Load configuration from environment variables respecting precedence."""

        return cls()


class AdapterRequest(BaseModel):
    """Input payload for adapter fetch operations."""

    tenant_id: str = Field(description="Tenant identifier for multi-tenant deployments.")
    correlation_id: str = Field(description="Correlation ID for tracing requests across services.")
    domain: AdapterDomain = Field(description="Domain of the adapter handling the request.")
    parameters: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary adapter-specific parameters.",
    )
    pagination: Pagination = Field(
        default_factory=Pagination,
        description="Pagination information for incremental fetching.",
    )
    config: AdapterConfig | None = Field(
        default=None,
        description="Optional configuration overriding defaults for the request.",
    )

    @field_validator("tenant_id", "correlation_id")
    @classmethod
    def _ensure_non_empty(cls, value: str, info: ValidationInfo) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        return value


class AdapterCostEstimate(BaseModel):
    """Represents an estimation of upstream cost for a request."""

    estimated_requests: PositiveInt = Field(1, description="Number of upstream API calls expected.")
    estimated_latency_seconds: float = Field(
        0.0,
        ge=0,
        description="Estimated latency required to execute the request.",
    )
    estimated_expiry: datetime | None = Field(
        default=None,
        description="Optional timestamp indicating when this estimate becomes stale.",
    )

    @classmethod
    def from_rate_limit(
        cls,
        requests_per_second: float,
        window: timedelta,
    ) -> "AdapterCostEstimate":
        """Derive a naive cost estimate based on rate limits and window sizes."""

        if requests_per_second <= 0:
            return cls(
                estimated_requests=1, estimated_latency_seconds=float(window.total_seconds())
            )
        estimated_requests = max(1, int(window.total_seconds() * requests_per_second))
        latency = estimated_requests / max(requests_per_second, 1e-6)
        return cls(
            estimated_requests=estimated_requests,
            estimated_latency_seconds=latency,
            estimated_expiry=datetime.now(UTC) + window,
        )


class AdapterResponse(BaseModel):
    """Standardised response envelope for adapter outputs."""

    items: list[Any] = Field(default_factory=list)
    next_page_token: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def has_more(self) -> bool:
        return self.next_page_token is not None


class ValidationOutcome(BaseModel):
    """Result of adapter payload validation."""

    valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def success(cls) -> "ValidationOutcome":
        return cls(valid=True)

    @classmethod
    def failure(cls, *errors: str) -> "ValidationOutcome":
        return cls(valid=False, errors=list(errors))


class AdapterMetadata(BaseModel):
    """Metadata describing adapter capabilities."""

    name: str
    version: str = Field(description="Semantic version of the adapter implementation.")
    domain: AdapterDomain
    summary: str = Field(default="", description="Short human-readable summary of the adapter.")
    capabilities: list[str] = Field(default_factory=list)
    maintainer: str | None = Field(default=None, description="Contact for adapter maintenance.")
    entry_point: str | None = Field(
        default=None, description="Entry point path used for registration."
    )
    schema_version: str = Field(
        default="1.0.0", description="Version of the adapter contract schema."
    )
    config_schema: Mapping[str, Any] = Field(default_factory=dict)
    extra: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("Adapter name must not be empty")
        return value

    @classmethod
    def from_config(
        cls,
        name: str,
        domain: AdapterDomain,
        summary: str,
        config_model: type[AdapterConfig] | None = None,
        **kwargs: Any,
    ) -> "AdapterMetadata":
        schema: Mapping[str, Any] = {}
        if config_model is not None:
            schema = config_model().json_schema()
        return cls(
            name=name,
            domain=domain,
            summary=summary,
            config_schema=schema,
            **kwargs,
        )


class BiomedicalPayload(BaseModel):
    """Domain-specific extension payload for biomedical adapters."""

    mesh_terms: list[str] = Field(default_factory=list)
    trial_phase: str | None = None


class FinancialPayload(BaseModel):
    """Domain-specific extension payload for financial adapters."""

    ticker: str
    market: str | None = None


class LegalPayload(BaseModel):
    """Domain-specific extension payload for legal adapters."""

    jurisdiction: str
    case_number: str | None = None


class AdapterResponseEnvelope(AdapterResponse):
    """Extended response envelope binding domain payloads."""

    biomedical: BiomedicalPayload | None = None
    financial: FinancialPayload | None = None
    legal: LegalPayload | None = None

    DOMAIN_ATTRIBUTE: ClassVar[dict[AdapterDomain, str]] = {
        AdapterDomain.BIOMEDICAL: "biomedical",
        AdapterDomain.FINANCIAL: "financial",
        AdapterDomain.LEGAL: "legal",
    }

    def attach_payload(self, domain: AdapterDomain, payload: BaseModel) -> None:
        attribute = self.DOMAIN_ATTRIBUTE[domain]
        setattr(self, attribute, payload)
