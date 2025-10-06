"""Adapter SDK base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from Medical_KG_rev.models import Document


@dataclass
class AdapterContext:
    """Shared context passed to adapter lifecycle methods."""

    tenant_id: str
    domain: str
    correlation_id: str
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass
class AdapterResult:
    """Represents the output of an adapter run."""

    documents: Sequence[Document]
    warnings: Sequence[str]


class BaseAdapter(ABC):
    """Base class that all adapters must inherit from."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def fetch(self, context: AdapterContext) -> Iterable[dict]:  # pragma: no cover - abstract
        """Fetch raw payloads from external systems."""

    @abstractmethod
    def parse(
        self, payloads: Iterable[dict], context: AdapterContext
    ) -> Sequence[Document]:  # pragma: no cover
        """Transform payloads into domain documents."""

    def validate(self, documents: Sequence[Document], context: AdapterContext) -> Sequence[str]:
        """Validate documents, returning warnings if applicable."""

        warnings: list[str] = []
        for document in documents:
            if not document.sections:
                warnings.append(f"Document {document.id} is empty")
        return warnings

    @abstractmethod
    def write(
        self, documents: Sequence[Document], context: AdapterContext
    ) -> None:  # pragma: no cover
        """Persist documents to downstream storage."""

    def run(self, context: AdapterContext) -> AdapterResult:
        """Execute the full adapter lifecycle."""

        payloads = self.fetch(context)
        documents = self.parse(payloads, context)
        warnings = self.validate(documents, context)
        self.write(documents, context)
        return AdapterResult(documents=documents, warnings=warnings)
