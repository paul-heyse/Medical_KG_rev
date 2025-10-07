"""Financial domain adapter plugins."""

from __future__ import annotations

from datetime import datetime

from Medical_KG_rev.adapters.plugins.base import BaseAdapterPlugin
from Medical_KG_rev.adapters.plugins.models import (
    AdapterConfig,
    AdapterRequest,
    AdapterResponse,
    ValidationOutcome,
)

from ..metadata import FinancialAdapterMetadata


class FinancialNewsConfig(AdapterConfig):
    """Configuration for the synthetic financial news adapter."""

    headline_prefix: str = "Market Update"


class FinancialNewsAdapterPlugin(BaseAdapterPlugin):
    """Generates synthetic financial news suitable for testing."""

    config_model = FinancialNewsConfig
    metadata = FinancialAdapterMetadata(
        name="financial-news",
        version="0.1.0",
        summary="Synthetic financial market news adapter",
        capabilities=["market-news"],
        maintainer="Financial Data Team",
        dataset="financial_news",
        reporting_frequency="hourly",
        compliance=["SOX"],
    )

    def fetch(self, request: AdapterRequest) -> AdapterResponse:
        headline = f"{self.config.headline_prefix}: {request.parameters.get('symbol', 'MK')}"
        item = {
            "headline": headline,
            "symbol": request.parameters.get("symbol", "MK"),
            "published_at": datetime.utcnow().isoformat(),
        }
        return AdapterResponse(items=[item])

    def parse(self, response: AdapterResponse, request: AdapterRequest) -> AdapterResponse:
        response.metadata["parsed"] = True
        return response

    def validate(self, response: AdapterResponse, request: AdapterRequest) -> ValidationOutcome:
        if not response.items:
            return ValidationOutcome.failure("No financial news generated")
        return ValidationOutcome(valid=True)


__all__ = [
    "FinancialNewsAdapterPlugin",
    "FinancialNewsConfig",
]
