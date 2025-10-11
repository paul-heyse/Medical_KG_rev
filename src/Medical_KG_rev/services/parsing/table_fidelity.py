"""Table fidelity preservation service for biomedical document processing.

This module provides table structure preservation capabilities including:
- Table structure preservation in chunk serialization
- Header mapping intact when flattening tables
- "Header: Value" phrasing for table content
- Caption text inclusion in table chunks
- Table schema preservation for rendering
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TableStructureType(Enum):
    """Types of table structures."""

    SIMPLE = "simple"  # Basic table with headers and rows
    COMPLEX = "complex"  # Multi-level headers, merged cells
    NESTED = "nested"  # Tables within tables
    MATRIX = "matrix"  # Matrix-style data


@dataclass
class TableCell:
    """Represents a table cell."""

    value: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    header: bool = False
    merged: bool = False


@dataclass
class TableHeader:
    """Represents a table header."""

    text: str
    level: int  # Header level (1, 2, 3, etc.)
    span: int  # Number of columns spanned
    subheaders: list["TableHeader"] = None


@dataclass
class TableSchema:
    """Table schema for preservation."""

    headers: list[TableHeader]
    rows: int
    cols: int
    structure_type: TableStructureType
    caption: str | None = None
    footnotes: list[str] = None


@dataclass
class TableChunk:
    """Table chunk with preserved structure."""

    chunk_id: str
    table_schema: TableSchema
    flattened_content: str
    contextualized_content: str
    machine_content: str
    preservation_metadata: dict[str, Any]


class TableFidelityPreserver:
    """Preserves table fidelity during document processing.

    Handles:
    - Table structure preservation in chunk serialization
    - Header mapping intact when flattening tables
    - "Header: Value" phrasing for table content
    - Caption text inclusion in table chunks
    - Table schema preservation for rendering
    """

    def __init__(self, preserve_structure: bool = True, include_captions: bool = True):
        """Initialize the table fidelity preserver.

        Args:
            preserve_structure: Whether to preserve table structure
            include_captions: Whether to include captions in table chunks

        """
        self.preserve_structure = preserve_structure
        self.include_captions = include_captions
        self._init_patterns()

    def _init_patterns(self) -> None:
        """Initialize regex patterns for table processing."""
        # Table detection patterns
        self.table_patterns = [
            # Standard table markers
            r"<table[^>]*>.*?</table>",
            r"\|.*\|.*\|",  # Pipe-separated tables
            r"^\s*\|.*\|.*$",  # Markdown-style tables
            # Medical table patterns
            r"Table\s+\d+[.:]\s*.*",  # "Table 1: ..."
            r"Table\s+\d+[.:]\s*.*\n.*",  # Multi-line table headers
        ]

        # Header detection patterns
        self.header_patterns = [
            r"^[A-Z][^a-z]*$",  # All caps headers
            r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$",  # Title case headers
            r"^\d+\.\s*[A-Z]",  # Numbered headers
            r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:$",  # Headers ending with colon
        ]

        # Cell content patterns
        self.cell_patterns = [
            r"^\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|IU|U|mEq|mol|mmol|μmol|nmol|pmol)",  # Medical units
            r"^\d+(?:\.\d+)?\s*%",  # Percentages
            r"^\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?",  # Ranges with ±
            r"^\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?",  # Ranges with -
            r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$",  # Proper nouns
        ]

    def preserve_table_structure(self, table_data: dict[str, Any]) -> TableChunk:
        """Preserve table structure during processing.

        Args:
            table_data: Raw table data from document processing

        Returns:
            TableChunk with preserved structure

        """
        try:
            # Extract table schema
            schema = self._extract_table_schema(table_data)

            # Create flattened content with header mapping
            flattened_content = self._create_flattened_content(table_data, schema)

            # Create contextualized content
            contextualized_content = self._create_contextualized_content(table_data, schema)

            # Create machine content
            machine_content = self._create_machine_content(table_data, schema)

            # Generate chunk ID
            chunk_id = self._generate_chunk_id(table_data, schema)

            # Create preservation metadata
            preservation_metadata = self._create_preservation_metadata(table_data, schema)

            return TableChunk(
                chunk_id=chunk_id,
                table_schema=schema,
                flattened_content=flattened_content,
                contextualized_content=contextualized_content,
                machine_content=machine_content,
                preservation_metadata=preservation_metadata,
            )

        except Exception as e:
            logger.error(f"Error preserving table structure: {e}")
            # Return minimal table chunk
            return self._create_fallback_table_chunk(table_data)

    def _extract_table_schema(self, table_data: dict[str, Any]) -> TableSchema:
        """Extract table schema from raw data."""
        headers = []
        rows = 0
        cols = 0
        structure_type = TableStructureType.SIMPLE
        caption = None
        footnotes = []

        # Extract headers
        if "headers" in table_data:
            headers = self._process_headers(table_data["headers"])

        # Extract dimensions
        if "rows" in table_data:
            rows = table_data["rows"]
        if "cols" in table_data:
            cols = table_data["cols"]

        # Determine structure type
        if "structure_type" in table_data:
            structure_type = TableStructureType(table_data["structure_type"])
        else:
            structure_type = self._detect_structure_type(table_data)

        # Extract caption
        if "caption" in table_data and self.include_captions:
            caption = table_data["caption"]

        # Extract footnotes
        if "footnotes" in table_data:
            footnotes = table_data["footnotes"]

        return TableSchema(
            headers=headers,
            rows=rows,
            cols=cols,
            structure_type=structure_type,
            caption=caption,
            footnotes=footnotes,
        )

    def _process_headers(self, headers_data: list[Any]) -> list[TableHeader]:
        """Process table headers."""
        headers = []

        for i, header_data in enumerate(headers_data):
            if isinstance(header_data, dict):
                header = TableHeader(
                    text=header_data.get("text", ""),
                    level=header_data.get("level", 1),
                    span=header_data.get("span", 1),
                    subheaders=header_data.get("subheaders", []),
                )
            else:
                header = TableHeader(text=str(header_data), level=1, span=1, subheaders=[])
            headers.append(header)

        return headers

    def _detect_structure_type(self, table_data: dict[str, Any]) -> TableStructureType:
        """Detect table structure type."""
        # Check for complex headers
        if "headers" in table_data:
            headers = table_data["headers"]
            if any(isinstance(h, dict) and h.get("level", 1) > 1 for h in headers):
                return TableStructureType.COMPLEX

        # Check for merged cells
        if "cells" in table_data:
            cells = table_data["cells"]
            if any(cell.get("merged", False) for cell in cells):
                return TableStructureType.COMPLEX

        # Check for nested tables
        if table_data.get("nested_tables"):
            return TableStructureType.NESTED

        return TableStructureType.SIMPLE

    def _create_flattened_content(self, table_data: dict[str, Any], schema: TableSchema) -> str:
        """Create flattened content with header mapping."""
        content_parts = []

        # Add caption if present
        if schema.caption and self.include_captions:
            content_parts.append(f"Table: {schema.caption}")

        # Process table rows
        if "rows" in table_data:
            rows = table_data["rows"]
            headers = schema.headers

            for row_idx, row in enumerate(rows):
                row_content = []

                for col_idx, cell_value in enumerate(row):
                    if col_idx < len(headers):
                        header_text = headers[col_idx].text
                        if cell_value and str(cell_value).strip():
                            row_content.append(f"{header_text}: {cell_value}")

                if row_content:
                    content_parts.append(" | ".join(row_content))

        return "\n".join(content_parts)

    def _create_contextualized_content(
        self, table_data: dict[str, Any], schema: TableSchema
    ) -> str:
        """Create contextualized content with section path."""
        content_parts = []

        # Add section path context
        section_path = table_data.get("section_path", "Table")
        content_parts.append(f"Section: {section_path}")

        # Add caption if present
        if schema.caption and self.include_captions:
            content_parts.append(f"Table: {schema.caption}")

        # Add table structure information
        content_parts.append(f"Table Structure: {schema.structure_type.value}")
        content_parts.append(f"Dimensions: {schema.rows} rows × {schema.cols} columns")

        # Add flattened content
        flattened = self._create_flattened_content(table_data, schema)
        if flattened:
            content_parts.append(flattened)

        return "\n".join(content_parts)

    def _create_machine_content(self, table_data: dict[str, Any], schema: TableSchema) -> str:
        """Create machine-readable content."""
        content_parts = []

        # Add structured data
        if "rows" in table_data:
            rows = table_data["rows"]
            headers = [h.text for h in schema.headers]

            # Create CSV-like format
            content_parts.append("|".join(headers))

            for row in rows:
                row_values = [str(cell) if cell is not None else "" for cell in row]
                content_parts.append("|".join(row_values))

        return "\n".join(content_parts)

    def _generate_chunk_id(self, table_data: dict[str, Any], schema: TableSchema) -> str:
        """Generate deterministic chunk ID."""
        # Use table position and content hash
        position = table_data.get("position", {})
        page_no = position.get("page", 1)
        bbox = position.get("bbox", [0, 0, 0, 0])

        # Create content hash
        content_hash = hash(str(schema.headers) + str(table_data.get("rows", [])))

        return f"table_{page_no}_{bbox[0]}_{bbox[1]}_{abs(content_hash)}"

    def _create_preservation_metadata(
        self, table_data: dict[str, Any], schema: TableSchema
    ) -> dict[str, Any]:
        """Create preservation metadata."""
        return {
            "preservation_method": "table_fidelity_preserver",
            "structure_type": schema.structure_type.value,
            "headers_count": len(schema.headers),
            "rows_count": schema.rows,
            "cols_count": schema.cols,
            "has_caption": schema.caption is not None,
            "has_footnotes": schema.footnotes is not None and len(schema.footnotes) > 0,
            "preservation_timestamp": table_data.get("timestamp", ""),
            "original_format": table_data.get("format", "unknown"),
        }

    def _create_fallback_table_chunk(self, table_data: dict[str, Any]) -> TableChunk:
        """Create fallback table chunk when preservation fails."""
        # Create minimal schema
        schema = TableSchema(
            headers=[],
            rows=0,
            cols=0,
            structure_type=TableStructureType.SIMPLE,
            caption=None,
            footnotes=[],
        )

        # Create minimal content
        content = table_data.get("text", "Table content unavailable")

        return TableChunk(
            chunk_id=f"fallback_table_{hash(content)}",
            table_schema=schema,
            flattened_content=content,
            contextualized_content=content,
            machine_content=content,
            preservation_metadata={"fallback": True, "error": "preservation_failed"},
        )

    def validate_table_preservation(self, chunk: TableChunk) -> bool:
        """Validate table preservation quality.

        Args:
            chunk: TableChunk to validate

        Returns:
            True if validation passes, False otherwise

        """
        # Check for fallback chunk
        if chunk.preservation_metadata.get("fallback", False):
            return False

        # Check for empty content
        if not chunk.flattened_content.strip():
            return False

        # Check for reasonable content length
        if len(chunk.flattened_content) < 10:
            return False

        # Check for header-value pairs
        if ":" not in chunk.flattened_content:
            return False

        return True

    def get_preservation_stats(self) -> dict[str, Any]:
        """Get preservation statistics."""
        return {
            "preserve_structure": self.preserve_structure,
            "include_captions": self.include_captions,
            "table_patterns": len(self.table_patterns),
            "header_patterns": len(self.header_patterns),
            "cell_patterns": len(self.cell_patterns),
        }


def preserve_table_fidelity(
    table_data: dict[str, Any], preserve_structure: bool = True, include_captions: bool = True
) -> TableChunk:
    """Convenience function for table fidelity preservation.

    Args:
        table_data: Raw table data
        preserve_structure: Whether to preserve structure
        include_captions: Whether to include captions

    Returns:
        TableChunk with preserved structure

    """
    preserver = TableFidelityPreserver(preserve_structure, include_captions)
    return preserver.preserve_table_structure(table_data)
