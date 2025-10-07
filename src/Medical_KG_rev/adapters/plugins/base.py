"""Base classes for adapter plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta

from Medical_KG_rev.adapters.plugins.manager import hookimpl
from Medical_KG_rev.adapters.plugins.models import (
    AdapterConfig,
    AdapterCostEstimate,
    AdapterMetadata,
    AdapterRequest,
    AdapterResponse,
    ValidationOutcome,
)


class BaseAdapterPlugin(ABC):
    """Abstract base class for adapter plugins."""

    metadata: AdapterMetadata
    config_model: type[AdapterConfig] = AdapterConfig

    def __init__(self, config: AdapterConfig | None = None) -> None:
        self.config = config or self.config_model()

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------
    @hookimpl
    def get_metadata(self) -> AdapterMetadata:
        metadata = self.metadata.model_copy(deep=True)
        if not metadata.config_schema:
            metadata.config_schema = self.config.json_schema()
        return metadata

    @hookimpl
    @abstractmethod
    def fetch(self, request: AdapterRequest) -> AdapterResponse:  # pragma: no cover - abstract
        """Fetch raw payloads from upstream systems."""

    @hookimpl
    @abstractmethod
    def parse(self, response: AdapterResponse, request: AdapterRequest) -> AdapterResponse:  # pragma: no cover
        """Parse raw payloads into canonical documents."""

    @hookimpl
    def validate(self, response: AdapterResponse, request: AdapterRequest) -> ValidationOutcome:
        return ValidationOutcome(valid=True, warnings=response.warnings)

    @hookimpl
    def health_check(self) -> bool:
        return True

    @hookimpl
    def estimate_cost(self, request: AdapterRequest) -> AdapterCostEstimate:
        rate_limit = max(self.config.rate_limit_per_second, 1e-6)
        window = timedelta(seconds=request.pagination.page_size / rate_limit)
        return AdapterCostEstimate.from_rate_limit(rate_limit, window)


class ReadOnlyAdapterPlugin(BaseAdapterPlugin):
    """Convenience base class for adapters that do not implement parse/validate overrides."""

    def parse(self, response: AdapterResponse, request: AdapterRequest) -> AdapterResponse:
        return response

