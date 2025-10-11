"""PMC adapter placeholder."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from Medical_KG_rev.adapters.base import AdapterContext, BaseAdapter
from Medical_KG_rev.models import Document


class PMCAdapter(BaseAdapter):
    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        return []

    def parse(
        self, payloads: Iterable[dict[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        return []

    def write(self, documents: Sequence[Document], context: AdapterContext) -> None:
        return None


__all__ = ["PMCAdapter"]
