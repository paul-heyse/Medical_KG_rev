"""Contextualize chunk text for Qwen3 embedding generation.

This module assembles contextual strings for Qwen3 dense embedding requests.
It combines raw chunk content with surrounding metadata such as section paths,
captions, and table information while preserving medical terminology.

Key Responsibilities:
    - Normalize and contextualize chunk text ahead of embedding generation.
    - Preserve sensitive medical terminology through placeholder handling.
    - Provide batch processing utilities for efficient pipeline integration.

Collaborators:
    - Upstream: Retrieval chunk processors that supply `ChunkContext` objects.
    - Downstream: `Qwen3Service` which submits contextualized text to gRPC.

Side Effects:
    - Emits structured logs for observability.
    - Raises `FallbackNotAllowedError` when contextualization fails.

Thread Safety:
    - Thread-safe: Instances do not mutate shared state after construction.

Performance Characteristics:
    - Linear relative to number of chunk contexts; string operations only.
    - No external I/O or shared caches.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import logging
import re
from typing import Any

from Medical_KG_rev.utils.fallbacks import fallback_unavailable

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ContextualizedTextConfig(BaseModel):
    """Configuration for contextualized text processing."""

    include_section_path: bool = Field(default=True, description="Include section path in context")
    include_caption: bool = Field(default=True, description="Include caption in context")
    max_context_length: int = Field(default=100, description="Maximum context length")
    context_separator: str = Field(default=" | ", description="Context separator")
    preserve_medical_terms: bool = Field(default=True, description="Preserve medical terminology")


class ChunkContext(BaseModel):
    """Chunk context information."""

    chunk_id: str = Field(..., description="Chunk identifier")
    doc_id: str = Field(..., description="Document identifier")
    page_no: int = Field(..., description="Page number")
    element_label: str = Field(..., description="Element label")
    section_path: str = Field(default="", description="Section path")
    contextualized_text: str = Field(default="", description="Contextualized text")
    content_only_text: str = Field(default="", description="Content-only text")
    caption: str = Field(default="", description="Caption text")
    table_payload: dict[str, Any] = Field(default_factory=dict, description="Table payload")


class Qwen3ContextualizedProcessor:
    """Compute contextualized text payloads for Qwen3 embeddings.

    Attributes:
        config: Runtime configuration controlling context assembly behavior.

    Thread Safety:
        - Thread-safe: Does not retain mutable shared state after construction.

    Example:
        >>> processor = Qwen3ContextualizedProcessor()
        >>> context = ChunkContext(
        ...     chunk_id="chunk-1",
        ...     doc_id="doc-42",
        ...     page_no=1,
        ...     element_label="paragraph",
        ...     section_path="Introduction",
        ...     contextualized_text="",
        ...     content_only_text="sample text",
        ... )
        >>> processor.process_chunk_context(context)
        'Introduction | sample text'
    """

    def __init__(self, config: ContextualizedTextConfig | None = None):
        """Initialize the contextualized text processor.

        Args:
            config: Configuration describing how context should be constructed.
        """
        self.config = config or ContextualizedTextConfig()

        # Medical terminology patterns
        self.medical_patterns = [
            r"\b\d+\.?\d*\s*(mg|ml|kg|g|mcg|μg|μl|μmol|mmol|iu|units?)\b",
            r"\b\d+\.?\d*\s*(mg/kg|ml/kg|g/kg|mcg/kg)\b",
            r"\b\d+\.?\d*\s*(mg/dl|mg/l|μg/dl|μg/l|mmol/l|iu/ml)\b",
            r"\b\d+\.?\d*\s*(bpm|hr|min|sec|h|d|wk|mo|yr)\b",
            r"\b\d+\.?\d*\s*(°c|°f|kpa|mmhg|cmh2o)\b",
            r"\b\d+\.?\d*\s*(mm|cm|m|in|ft)\b",
            r"\b\d+\.?\d*\s*(ml|dl|l|gal|qt|pt)\b",
            r"\b\d+\.?\d*\s*(g|kg|lb|oz)\b",
            r"\b\d+\.?\d*\s*(%|percent)\b",
            r"\b\d+\.?\d*\s*(x|×)\s*\d+\.?\d*\b",
        ]

        # Compile patterns
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.medical_patterns
        ]

        logger.info(
            "Initialized Qwen3 contextualized processor",
            extra={
                "include_section_path": self.config.include_section_path,
                "include_caption": self.config.include_caption,
                "max_context_length": self.config.max_context_length,
                "context_separator": self.config.context_separator,
                "preserve_medical_terms": self.config.preserve_medical_terms,
            },
        )

    def _preserve_medical_terms(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace medical terms with placeholders to prevent truncation.

        Args:
            text: Raw chunk text.

        Returns:
            A tuple containing the masked text and a mapping from placeholders
            to the original medical terms.
        """
        placeholders = {}
        processed_text = text

        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(processed_text)
            for j, match in enumerate(matches):
                placeholder = f"__MEDICAL_{i}_{j}__"
                placeholders[placeholder] = match
                processed_text = processed_text.replace(match, placeholder, 1)

        return processed_text, placeholders

    def _restore_medical_terms(self, text: str, placeholders: dict[str, str]) -> str:
        """Restore medical terminology that was placeholder-substituted.

        Args:
            text: Text containing placeholders.
            placeholders: Mapping from placeholder tokens to original terms.

        Returns:
            The input text with all placeholders replaced by medical terms.
        """
        restored_text = text
        for placeholder, original in placeholders.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text

    def _extract_section_context(self, section_path: str) -> str:
        """Normalize a section path into a context string."""
        if not section_path:
            return ""

        # Split section path and create context
        parts = section_path.split(" > ")
        if len(parts) <= 1:
            return section_path

        # Create hierarchical context
        context_parts = []
        for i, part in enumerate(parts):
            if i == 0:
                context_parts.append(f"Document: {part}")
            else:
                context_parts.append(f"Section: {part}")

        return " > ".join(context_parts)

    def _extract_caption_context(self, caption: str) -> str:
        """Format caption text into contextual metadata."""
        if not caption:
            return ""

        return f"Caption: {caption}"

    def _extract_table_context(self, table_payload: dict[str, Any]) -> str:
        """Extract table context from table payload.

        Args:
        ----
            table_payload: Table payload

        Returns:
        -------
            Table context

        """
        if not table_payload:
            return ""

        context_parts = []

        # Extract headers
        if "headers" in table_payload:
            headers = table_payload["headers"]
            if isinstance(headers, list):
                context_parts.append(f"Table headers: {', '.join(headers)}")

        # Extract caption
        if "caption" in table_payload:
            caption = table_payload["caption"]
            if caption:
                context_parts.append(f"Table caption: {caption}")

        return " | ".join(context_parts)

    def _create_contextualized_text(self, chunk_context: ChunkContext) -> str:
        """Generate the context string submitted to Qwen3 embeddings.

        Args:
            chunk_context: Chunk-level metadata and text payload.

        Returns:
            A context string containing metadata and chunk text.
        """
        context_parts = []

        # Add section context if configured
        if self.config.include_section_path and chunk_context.section_path:
            section_context = self._extract_section_context(chunk_context.section_path)
            if section_context:
                context_parts.append(section_context)

        # Add caption context if configured
        if self.config.include_caption and chunk_context.caption:
            caption_context = self._extract_caption_context(chunk_context.caption)
            if caption_context:
                context_parts.append(caption_context)

        # Add table context if present
        if chunk_context.table_payload:
            table_context = self._extract_table_context(chunk_context.table_payload)
            if table_context:
                context_parts.append(table_context)

        # Add main content
        main_content = chunk_context.contextualized_text or chunk_context.content_only_text
        if main_content:
            context_parts.append(main_content)

        # Join with separator
        contextualized_text = self.config.context_separator.join(context_parts)

        # Truncate if too long
        if (
            len(contextualized_text) > self.config.max_context_length * 4
        ):  # Rough character to token ratio
            contextualized_text = contextualized_text[: self.config.max_context_length * 4]

        return contextualized_text

    def process_chunk_context(self, chunk_context: ChunkContext) -> str:
        """Transform a single chunk context into contextualized text.

        Args:
            chunk_context: Chunk metadata and text to contextualize.

        Returns:
            The contextualized string supplied to Qwen3.
        """
        try:
            # Create contextualized text
            contextualized_text = self._create_contextualized_text(chunk_context)

            # Preserve medical terms if configured
            if self.config.preserve_medical_terms:
                processed_text, placeholders = self._preserve_medical_terms(contextualized_text)
                # Restore medical terms
                contextualized_text = self._restore_medical_terms(processed_text, placeholders)

            logger.debug(
                "Processed chunk context for Qwen3 embedding",
                extra={
                    "chunk_id": chunk_context.chunk_id,
                    "element_label": chunk_context.element_label,
                    "contextualized_text_length": len(contextualized_text),
                    "section_path": chunk_context.section_path,
                    "caption": chunk_context.caption,
                },
            )

            return contextualized_text

        except Exception as e:
            logger.error(
                "Failed to process chunk context for Qwen3 embedding",
                extra={
                    "chunk_id": chunk_context.chunk_id,
                    "error": str(e),
                },
            )
            raise

    def validate_contextualized_text(self, contextualized_text: str) -> list[str]:
        """Validate that contextualized text meets formatting requirements.

        Args:
            contextualized_text: Context string to validate.

        Returns:
            A list of validation error messages. Empty if valid.
        """
        errors = []

        # Check text length
        if len(contextualized_text) == 0:
            errors.append("Contextualized text is empty")
        elif len(contextualized_text) > self.config.max_context_length * 4:
            errors.append(f"Contextualized text is too long: {len(contextualized_text)} characters")

        # Check for valid content
        if contextualized_text.strip() == "":
            errors.append("Contextualized text contains only whitespace")

        return errors

    def get_processing_stats(self, chunk_context: ChunkContext) -> dict[str, Any]:
        """Get processing statistics for chunk context.

        Args:
        ----
            chunk_context: Chunk context information

        Returns:
        -------
            Dictionary with processing statistics

        """
        contextualized_text = self._create_contextualized_text(chunk_context)

        return {
            "chunk_id": chunk_context.chunk_id,
            "element_label": chunk_context.element_label,
            "section_path": chunk_context.section_path,
            "caption": chunk_context.caption,
            "contextualized_text_length": len(contextualized_text),
            "content_only_text_length": len(chunk_context.content_only_text),
            "has_table": bool(chunk_context.table_payload),
            "config": {
                "include_section_path": self.config.include_section_path,
                "include_caption": self.config.include_caption,
                "max_context_length": self.config.max_context_length,
                "context_separator": self.config.context_separator,
                "preserve_medical_terms": self.config.preserve_medical_terms,
            },
        }

    def batch_process_chunk_contexts(self, chunk_contexts: list[ChunkContext]) -> list[str]:
        """Contextualize many chunk contexts sequentially.

        Args:
            chunk_contexts: Chunk contexts to contextualize.

        Returns:
            A list of contextualized text strings corresponding to the input.
        """
        contextualized_texts = []

        for chunk_context in chunk_contexts:
            try:
                contextualized_text = self.process_chunk_context(chunk_context)
                contextualized_texts.append(contextualized_text)
            except Exception as e:
                fallback_unavailable(
                    "Qwen3 contextualized text processing",
                    f"chunk_id={chunk_context.chunk_id}: {e}",
                )

        logger.info(
            "Processed chunk contexts in batch",
            extra={
                "chunk_count": len(chunk_contexts),
                "successful_count": sum(1 for text in contextualized_texts if text),
            },
        )

        return contextualized_texts
