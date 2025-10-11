"""Financial adapter plugin placeholder."""

from __future__ import annotations

from Medical_KG_rev.adapters.plugins.base import BaseAdapterPlugin
from Medical_KG_rev.adapters.plugins.models import AdapterMetadata


class FinancialNewsAdapterPlugin(BaseAdapterPlugin):
    metadata = AdapterMetadata(name="financial-news", version="0.0.0", domain="financial")

    def fetch(self, request):  # type: ignore[override]
        raise NotImplementedError

    def parse(self, response, request):  # type: ignore[override]
        return response


__all__ = ["FinancialNewsAdapterPlugin"]
