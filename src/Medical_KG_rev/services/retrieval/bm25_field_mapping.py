"""BM25 field mapping for medical document structure.

This module implements field mapping from chunk structure to BM25 fields
with appropriate boosts and medical terminology handling.
"""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChunkStructure(BaseModel):
    """Chunk structure for field mapping."""

    chunk_id: str = Field(..., description="Chunk identifier")
    doc_id: str = Field(..., description="Document identifier")
    page_no: int = Field(..., description="Page number")
    element_label: str = Field(..., description="Element label (TITLE, PARAGRAPH, TABLE, etc.)")
    section_path: str = Field(default="", description="Section path")
    contextualized_text: str = Field(default="", description="Contextualized text")
    content_only_text: str = Field(default="", description="Content-only text")
    table_payload: dict[str, Any] = Field(default_factory=dict, description="Table payload")
    caption: str = Field(default="", description="Caption text")
    footnote: str = Field(default="", description="Footnote text")
    refs_text: str = Field(default="", description="References text")


class BM25FieldMapping(BaseModel):
    """BM25 field mapping configuration."""

    title_boost: float = Field(default=3.0, description="Title field boost")
    section_headers_boost: float = Field(default=2.5, description="Section headers boost")
    paragraph_boost: float = Field(default=1.0, description="Paragraph field boost")
    caption_boost: float = Field(default=2.0, description="Caption field boost")
    table_text_boost: float = Field(default=1.5, description="Table text boost")
    footnote_boost: float = Field(default=0.5, description="Footnote boost")
    refs_text_boost: float = Field(default=0.1, description="References text boost")

    # Medical terminology preservation
    preserve_medical_terms: bool = Field(default=True, description="Preserve medical terminology")
    medical_term_patterns: list[str] = Field(
        default_factory=lambda: [
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
        ],
        description="Medical term patterns to preserve",
    )


class BM25FieldMapper:
    """BM25 field mapper for medical document structure.

    This class maps chunk structure to BM25 fields with appropriate
    boosts and medical terminology handling.
    """

    def __init__(self, config: BM25FieldMapping | None = None):
        """Initialize BM25 field mapper.

        Args:
            config: Field mapping configuration

        """
        self.config = config or BM25FieldMapping()

        # Compile medical term patterns
        self.medical_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.config.medical_term_patterns
        ]

        logger.info(
            "Initialized BM25 field mapper",
            extra={
                "title_boost": self.config.title_boost,
                "section_headers_boost": self.config.section_headers_boost,
                "paragraph_boost": self.config.paragraph_boost,
                "caption_boost": self.config.caption_boost,
                "table_text_boost": self.config.table_text_boost,
                "footnote_boost": self.config.footnote_boost,
                "refs_text_boost": self.config.refs_text_boost,
                "preserve_medical_terms": self.config.preserve_medical_terms,
            },
        )

    def _extract_title(self, chunk: ChunkStructure) -> str:
        """Extract title from chunk structure.

        Args:
            chunk: Chunk structure

        Returns:
            Title text

        """
        if chunk.element_label == "TITLE":
            return chunk.content_only_text

        # Extract title from section path
        if chunk.section_path:
            # Get the first part of the section path as title
            parts = chunk.section_path.split(" > ")
            if parts:
                return parts[0]

        return ""

    def _extract_section_headers(self, chunk: ChunkStructure) -> str:
        """Extract section headers from chunk structure.

        Args:
            chunk: Chunk structure

        Returns:
            Section headers text

        """
        if chunk.element_label == "SECTION_HEADER":
            return chunk.content_only_text

        # Extract from section path
        if chunk.section_path:
            # Get all parts of the section path as headers
            parts = chunk.section_path.split(" > ")
            return " ".join(parts)

        return ""

    def _extract_paragraph(self, chunk: ChunkStructure) -> str:
        """Extract paragraph text from chunk structure.

        Args:
            chunk: Chunk structure

        Returns:
            Paragraph text

        """
        if chunk.element_label == "PARAGRAPH":
            return chunk.content_only_text

        # For other elements, use content-only text
        return chunk.content_only_text

    def _extract_caption(self, chunk: ChunkStructure) -> str:
        """Extract caption text from chunk structure.

        Args:
            chunk: Chunk structure

        Returns:
            Caption text

        """
        if chunk.caption:
            return chunk.caption

        # Extract caption from contextualized text if present
        if chunk.contextualized_text and "caption:" in chunk.contextualized_text.lower():
            # Look for caption pattern in contextualized text
            caption_match = re.search(
                r"caption:\s*([^\n]+)", chunk.contextualized_text, re.IGNORECASE
            )
            if caption_match:
                return caption_match.group(1).strip()

        return ""

    def _extract_table_text(self, chunk: ChunkStructure) -> str:
        """Extract table text from chunk structure.

        Args:
            chunk: Chunk structure

        Returns:
            Table text

        """
        if chunk.element_label == "TABLE":
            # Extract text from table payload
            if chunk.table_payload:
                table_text = []

                # Extract headers
                if "headers" in chunk.table_payload:
                    headers = chunk.table_payload["headers"]
                    if isinstance(headers, list):
                        table_text.append(" ".join(headers))

                # Extract cells
                if "cells" in chunk.table_payload:
                    cells = chunk.table_payload["cells"]
                    if isinstance(cells, list):
                        for row in cells:
                            if isinstance(row, list):
                                table_text.append(" ".join(str(cell) for cell in row))

                return " ".join(table_text)

            # Fallback to content-only text
            return chunk.content_only_text

        return ""

    def _extract_footnote(self, chunk: ChunkStructure) -> str:
        """Extract footnote text from chunk structure.

        Args:
            chunk: Chunk structure

        Returns:
            Footnote text

        """
        if chunk.footnote:
            return chunk.footnote

        # Extract footnote from contextualized text if present
        if chunk.contextualized_text and "footnote:" in chunk.contextualized_text.lower():
            # Look for footnote pattern in contextualized text
            footnote_match = re.search(
                r"footnote:\s*([^\n]+)", chunk.contextualized_text, re.IGNORECASE
            )
            if footnote_match:
                return footnote_match.group(1).strip()

        return ""

    def _extract_refs_text(self, chunk: ChunkStructure) -> str:
        """Extract references text from chunk structure.

        Args:
            chunk: Chunk structure

        Returns:
            References text

        """
        if chunk.refs_text:
            return chunk.refs_text

        # Extract references from contextualized text if present
        if chunk.contextualized_text and "references:" in chunk.contextualized_text.lower():
            # Look for references pattern in contextualized text
            refs_match = re.search(
                r"references:\s*([^\n]+)", chunk.contextualized_text, re.IGNORECASE
            )
            if refs_match:
                return refs_match.group(1).strip()

        return ""

    def _preserve_medical_terms(self, text: str) -> str:
        """Preserve medical terms in text.

        Args:
            text: Input text

        Returns:
            Text with preserved medical terms

        """
        if not self.config.preserve_medical_terms:
            return text

        # Replace medical terms with placeholders to preserve them
        placeholders = {}
        processed_text = text

        for i, pattern in enumerate(self.medical_patterns):
            matches = pattern.findall(processed_text)
            for j, match in enumerate(matches):
                placeholder = f"__MEDICAL_{i}_{j}__"
                placeholders[placeholder] = match
                processed_text = processed_text.replace(match, placeholder, 1)

        return processed_text, placeholders

    def _restore_medical_terms(self, text: str, placeholders: dict[str, str]) -> str:
        """Restore medical terms in text.

        Args:
            text: Processed text with placeholders
            placeholders: Dictionary mapping placeholders to original terms

        Returns:
            Text with restored medical terms

        """
        restored_text = text
        for placeholder, original in placeholders.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text

    def map_chunk_to_fields(self, chunk: ChunkStructure) -> dict[str, str]:
        """Map chunk structure to BM25 fields.

        Args:
            chunk: Chunk structure

        Returns:
            Dictionary mapping field names to field text

        """
        try:
            # Extract field text
            title = self._extract_title(chunk)
            section_headers = self._extract_section_headers(chunk)
            paragraph = self._extract_paragraph(chunk)
            caption = self._extract_caption(chunk)
            table_text = self._extract_table_text(chunk)
            footnote = self._extract_footnote(chunk)
            refs_text = self._extract_refs_text(chunk)

            # Preserve medical terms if configured
            if self.config.preserve_medical_terms:
                title, title_placeholders = self._preserve_medical_terms(title)
                section_headers, section_placeholders = self._preserve_medical_terms(
                    section_headers
                )
                caption, caption_placeholders = self._preserve_medical_terms(caption)

                # Restore medical terms
                title = self._restore_medical_terms(title, title_placeholders)
                section_headers = self._restore_medical_terms(section_headers, section_placeholders)
                caption = self._restore_medical_terms(caption, caption_placeholders)

            # Create field mapping
            fields = {
                "title": title,
                "section_headers": section_headers,
                "paragraph": paragraph,
                "caption": caption,
                "table_text": table_text,
                "footnote": footnote,
                "refs_text": refs_text,
            }

            logger.debug(
                "Mapped chunk to BM25 fields",
                extra={
                    "chunk_id": chunk.chunk_id,
                    "element_label": chunk.element_label,
                    "field_lengths": {field: len(text) for field, text in fields.items()},
                },
            )

            return fields

        except Exception as e:
            logger.error(
                "Failed to map chunk to BM25 fields",
                extra={
                    "chunk_id": chunk.chunk_id,
                    "error": str(e),
                },
            )
            raise

    def get_field_boosts(self) -> dict[str, float]:
        """Get field boost factors.

        Returns:
            Dictionary mapping field names to boost factors

        """
        return {
            "title": self.config.title_boost,
            "section_headers": self.config.section_headers_boost,
            "paragraph": self.config.paragraph_boost,
            "caption": self.config.caption_boost,
            "table_text": self.config.table_text_boost,
            "footnote": self.config.footnote_boost,
            "refs_text": self.config.refs_text_boost,
        }

    def validate_field_mapping(self, fields: dict[str, str]) -> list[str]:
        """Validate field mapping.

        Args:
            fields: Field mapping to validate

        Returns:
            List of validation error messages

        """
        errors = []

        # Check required fields
        required_fields = [
            "title",
            "section_headers",
            "paragraph",
            "caption",
            "table_text",
            "footnote",
            "refs_text",
        ]
        for field in required_fields:
            if field not in fields:
                errors.append(f"Missing required field: {field}")

        # Check field content
        for field_name, field_text in fields.items():
            if not isinstance(field_text, str):
                errors.append(f"Field {field_name} must be a string")
            elif len(field_text) > 10000:  # Reasonable limit
                errors.append(f"Field {field_name} is too long: {len(field_text)} characters")

        return errors

    def get_mapping_stats(self, fields: dict[str, str]) -> dict[str, Any]:
        """Get mapping statistics.

        Args:
            fields: Field mapping to analyze

        Returns:
            Dictionary with mapping statistics

        """
        stats = {
            "total_fields": len(fields),
            "field_lengths": {field: len(text) for field, text in fields.items()},
            "total_text_length": sum(len(text) for text in fields.values()),
            "non_empty_fields": sum(1 for text in fields.values() if text.strip()),
            "field_boosts": self.get_field_boosts(),
        }

        return stats
