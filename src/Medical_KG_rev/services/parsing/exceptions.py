"""Exception hierarchy for Docling Gemma3 VLM processing."""

from __future__ import annotations


class DoclingVLMError(RuntimeError):
    """Base exception raised for Docling VLM failures."""


class DoclingModelLoadError(DoclingVLMError):
    """Raised when the Gemma3 checkpoint or tokenizer cannot be loaded."""


class DoclingProcessingError(DoclingVLMError):
    """Raised when Docling fails to process a PDF document."""


class DoclingCircuitBreakerOpenError(DoclingVLMError):
    """Raised when the circuit breaker prevents further Docling execution."""


__all__ = [
    "DoclingCircuitBreakerOpenError",
    "DoclingModelLoadError",
    "DoclingProcessingError",
    "DoclingVLMError",
]

