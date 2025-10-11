"""Custom exceptions for Docling VLM processing."""



from __future__ import annotations

class DoclingVLMError(RuntimeError):
    """Base error for Docling VLM operations."""


class DoclingModelLoadError(DoclingVLMError):
    """Raised when the Gemma3 model cannot be loaded."""


class DoclingProcessingError(DoclingVLMError):
    """Raised when Docling fails to process a PDF document."""


class DoclingModelUnavailableError(DoclingVLMError):
    """Raised when the Docling model is unavailable."""


class DoclingOutOfMemoryError(DoclingVLMError):
    """Raised when Docling runs out of memory."""


class DoclingProcessingTimeoutError(DoclingVLMError):
    """Raised when Docling processing times out."""
