"""Organization and tenant models used for multi-tenancy."""
from __future__ import annotations

from typing import Dict, Optional

from pydantic import Field, field_validator

from .ir import IRBaseModel


class Organization(IRBaseModel):
    """Represents an organization that owns data in the system."""

    id: str
    name: str
    domain: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator("domain")
    @classmethod
    def _validate_domain(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if "." not in value:
            raise ValueError("Organization domain must include a dot")
        return value.lower()


class TenantContext(IRBaseModel):
    """Tenant specific context that should accompany all requests."""

    tenant_id: str
    organization: Organization
    correlation_id: str
    feature_flags: Dict[str, bool] = Field(default_factory=dict)

    @field_validator("feature_flags")
    @classmethod
    def _normalize_flags(cls, value: Dict[str, bool]) -> Dict[str, bool]:
        return {key.lower(): bool(enabled) for key, enabled in value.items()}
