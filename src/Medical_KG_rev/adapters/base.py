"""Adapter SDK base classes.

This module provides the foundational classes for the adapter SDK,
defining the core interfaces and data structures that all adapters
must implement. It establishes the adapter lifecycle pattern and
provides base implementations for common functionality.

Key Components:
    - BaseAdapter: Abstract base class for all adapters
    - AdapterContext: Shared context for adapter operations
    - AdapterResult: Standardized result structure
    - AdapterDomain: Enumeration of supported domains

Responsibilities:
    - Define adapter lifecycle interface (fetch, parse, validate, run)
    - Provide shared context and result structures
    - Establish domain categorization system
    - Support adapter registration and discovery

Collaborators:
    - Upstream: Adapter implementations, adapter registry
    - Downstream: Document models, adapter plugins

Side Effects:
    - None: Pure interface definitions and data structures

Thread Safety:
    - Thread-safe: All classes are stateless or immutable

Performance Characteristics:
    - O(1) interface operations
    - Minimal overhead for adapter lifecycle
    - Efficient context passing and result aggregation

Example:
    >>> from Medical_KG_rev.adapters.base import BaseAdapter, AdapterContext
    >>> class MyAdapter(BaseAdapter):
    ...     def fetch(self, context): return []
    ...     def parse(self, payloads, context): return []
    >>> adapter = MyAdapter("my-adapter")
    >>> context = AdapterContext(tenant_id="t1", domain="medical", correlation_id="c1")
    >>> result = adapter.run(context)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from Medical_KG_rev.models import Document

# ==============================================================================
# DATA MODELS
# ==============================================================================


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


# ==============================================================================
# ADAPTER IMPLEMENTATION
# ==============================================================================

class BaseAdapter(ABC):
    """Base class that all adapters must inherit from."""

    name: str

    def __init__(self, name: str) -> None:
        """Initialize adapter with given name.

        Args:
            name: Unique identifier for this adapter instance.
        """
        self.name = name

    @abstractmethod
    def fetch(self, context: AdapterContext) -> Iterable[dict]:  # pragma: no cover - abstract
        """Fetch raw payloads from external systems.

        Args:
            context: Adapter context with tenant and domain information.

        Returns:
            Iterable of raw payload dictionaries from external systems.
        """

    @abstractmethod
    def parse(
        self, payloads: Iterable[dict], context: AdapterContext
    ) -> Sequence[Document]:  # pragma: no cover
        """Transform payloads into domain documents.

        Args:
            payloads: Raw payloads from external systems.
            context: Adapter context with tenant and domain information.

        Returns:
            Sequence of parsed domain documents.
        """

    def validate(self, documents: Sequence[Document], context: AdapterContext) -> Sequence[str]:
        """Validate documents, returning warnings if applicable.

        Args:
            documents: Documents to validate.
            context: Adapter context with tenant and domain information.

        Returns:
            Sequence of validation warning messages.
        """
        warnings: list[str] = []
        for document in documents:
            if not document.sections:
                warnings.append(f"Document {document.id} is empty")
        return warnings

    @abstractmethod
    def write(
        self, documents: Sequence[Document], context: AdapterContext
    ) -> None:  # pragma: no cover
        """Persist documents to downstream storage.

        Args:
            documents: Documents to persist.
            context: Adapter context with tenant and domain information.
        """

    def run(self, context: AdapterContext) -> AdapterResult:
        """Execute the full adapter lifecycle.

        Args:
            context: Adapter context with tenant and domain information.

        Returns:
            Adapter result containing documents and warnings.
        """
        payloads = self.fetch(context)
        documents = self.parse(payloads, context)
        warnings = self.validate(documents, context)
        self.write(documents, context)
        return AdapterResult(documents=documents, warnings=warnings)


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "AdapterContext",
    "AdapterResult",
    "BaseAdapter",
]
