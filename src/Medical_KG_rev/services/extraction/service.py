"""Extraction service data models and abstract contract.

Key Responsibilities:
    - Define domain dataclasses representing extracted spans and results
    - Provide the abstract service contract for entity extraction pipelines

Collaborators:
    - Upstream: Gateway coordinators and service orchestrators invoke
      extraction through this interface
    - Downstream: Concrete NLP or ML extraction engines implement the contract

Side Effects:
    - None; module contains immutable dataclasses and abstract interfaces

Thread Safety:
    - Thread-safe; dataclasses are immutable and the abstract service holds no
      shared state

Performance Characteristics:
    - Dataclass operations are O(1); concrete performance depends on
      implementations that subclass :class:`ExtractionService`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ==============================================================================
# DATA MODELS
# ============================================================================


@dataclass(slots=True)
class ExtractionSpan:
    """Individual extracted entity or span from text.

    Attributes:
        label: Category or type of the extracted entity.
        text: The actual text content of the span.
        start: Starting character position in the original text.
        end: Ending character position in the original text.
        confidence: Confidence score for the extraction (0.0 to 1.0).
    """
    label: str
    text: str
    start: int
    end: int
    confidence: float


@dataclass(slots=True)
class ExtractionResult:
    """Result of an entity extraction operation.

    Attributes:
        document_id: Identifier for the document that was processed.
        kind: Type of extraction that was performed.
        spans: List of extracted entities and spans.
    """
    document_id: str
    kind: str
    spans: Sequence[ExtractionSpan]


# ==============================================================================
# INTERFACES
# ============================================================================


class ExtractionService:
    """Abstract base class for entity extraction service implementations.

    This class defines the interface that concrete extraction services must implement.
    It serves as the contract for extracting entities and spans from text.

    Attributes:
        None (abstract base class)

    Invariants:
        - Implementations must handle empty text gracefully
        - Must validate document_id and return appropriate results
        - Must support multiple extraction kinds

    Thread Safety:
        - Implementations must be thread-safe for concurrent requests

    Lifecycle:
        - No explicit lifecycle management required
        - Implementations may cache models or resources as needed

    Example:
        >>> class MyExtractionService(ExtractionService):
        ...     def extract(self, document_id, text, kind="generic"):
        ...         # Implementation here
        ...         return ExtractionResult(document_id, kind, [])
    """

    def extract(self, document_id: str, text: str, *, kind: str = "generic") -> ExtractionResult:
        """Extract entities and spans from text content.

        Args:
            document_id: Unique identifier for the document being processed.
            text: Text content to extract entities from.
            kind: Type of extraction to perform (e.g., "generic", "medical").

        Returns:
            Extraction result containing identified spans and metadata.

        Raises:
            NotImplementedError: Always raised by this abstract implementation.
            ValueError: If parameters are invalid or text is empty.
            RuntimeError: If extraction service encounters an error.
        """
        raise NotImplementedError(
            "ExtractionService.extract() not implemented. "
            "This service requires a real extraction implementation. "
            "Please implement or configure a proper extraction service."
        )


__all__ = ["ExtractionService", "ExtractionResult", "ExtractionSpan"]
