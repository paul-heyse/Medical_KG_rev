"""Biomedical plugin placeholders."""

from __future__ import annotations

from typing import Tuple

from Medical_KG_rev.adapters.plugins.base import BaseAdapterPlugin
from Medical_KG_rev.adapters.plugins.models import AdapterMetadata


class BiomedicalAdapterPlugin(BaseAdapterPlugin):
    metadata = AdapterMetadata(name="biomedical", version="0.0.0", domain="biomedical")

    def fetch(self, request):  # type: ignore[override]
        raise NotImplementedError

    def parse(self, response, request):  # type: ignore[override]
        return response


def builtin_biomedical_plugins() -> Tuple[BaseAdapterPlugin, ...]:
    return ()


def register_biomedical_plugins(manager) -> list[AdapterMetadata]:
    return []
