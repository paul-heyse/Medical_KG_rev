"""OpenFDA adapter placeholders."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from Medical_KG_rev.adapters.base import AdapterContext, BaseAdapter
from Medical_KG_rev.models import Document


class _OpenFDAAdapter(BaseAdapter):
    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        return []

    def parse(
        self, payloads: Iterable[dict[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        return []

    def write(self, documents: Sequence[Document], context: AdapterContext) -> None:
        return None


class OpenFDADeviceAdapter(_OpenFDAAdapter):
    pass


class OpenFDADrugLabelAdapter(_OpenFDAAdapter):
    pass


class OpenFDADrugEventAdapter(_OpenFDAAdapter):
    pass


__all__ = [
    "OpenFDADeviceAdapter",
    "OpenFDADrugLabelAdapter",
    "OpenFDADrugEventAdapter",
]
