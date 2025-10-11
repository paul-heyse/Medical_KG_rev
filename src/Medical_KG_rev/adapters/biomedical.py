"""Placeholder biomedical adapters."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from Medical_KG_rev.adapters.base import AdapterContext, BaseAdapter
from Medical_KG_rev.models import Document


class _StubAdapter(BaseAdapter):
    """Lightweight adapter placeholder used during refactoring."""

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        return []

    def parse(
        self, payloads: Iterable[dict[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        return []

    def write(self, documents: Sequence[Document], context: AdapterContext) -> None:
        return None


class ClinicalTrialsAdapter(_StubAdapter):
    pass


class COREAdapter(_StubAdapter):
    pass


class CrossrefAdapter(_StubAdapter):
    pass


class OpenFDADeviceAdapter(_StubAdapter):
    pass


class OpenFDADrugLabelAdapter(_StubAdapter):
    pass


class OpenFDADrugEventAdapter(_StubAdapter):
    pass


class PMCAdapter(_StubAdapter):
    pass


class UnpaywallAdapter(_StubAdapter):
    pass


class RxNormAdapter(_StubAdapter):
    pass


class ICD11Adapter(_StubAdapter):
    pass


class MeSHAdapter(_StubAdapter):
    pass


class ChEMBLAdapter(_StubAdapter):
    pass


class SemanticScholarAdapter(_StubAdapter):
    pass


__all__ = [
    "ClinicalTrialsAdapter",
    "COREAdapter",
    "CrossrefAdapter",
    "OpenFDADeviceAdapter",
    "OpenFDADrugLabelAdapter",
    "OpenFDADrugEventAdapter",
    "PMCAdapter",
    "UnpaywallAdapter",
    "RxNormAdapter",
    "ICD11Adapter",
    "MeSHAdapter",
    "ChEMBLAdapter",
    "SemanticScholarAdapter",
]
