"""Parser for MinerU CLI structured output.

This module provides utilities for parsing structured JSON output from
the MinerU CLI, including blocks, tables, figures, and equations. It
handles validation, type conversion, and error handling for MinerU
output files.

Key Components:
    - MineruOutputParser: Main parser class for MinerU output
    - ParsedBlock: Represents a parsed text block with metadata
    - ParsedDocument: Aggregates all parsed content for a document
    - MineruOutputParserError: Custom exception for parsing errors

Responsibilities:
    - Parse MinerU JSON output files
    - Validate and convert data types
    - Handle missing or malformed data gracefully
    - Support both JSON and Markdown input formats
    - Extract structured artifacts (tables, figures, equations)

Collaborators:
    - MinerU CLI for output generation
    - Document processing pipeline for parsed content
    - Model classes for structured data representation

Side Effects:
    - Reads files from filesystem
    - Logs parsing operations and statistics
    - May raise exceptions for invalid input

Thread Safety:
    - Thread-safe: Parser instances are stateless
    - File operations are read-only

Performance Characteristics:
    - O(n) parsing time for n blocks/artifacts
    - Memory usage scales with document size
    - Efficient type conversion and validation
    - Supports large documents with many artifacts

Example:
    >>> parser = MineruOutputParser()
    >>> document = parser.parse_path(Path("output.json"))
    >>> assert len(document.blocks) > 0
    >>> assert len(document.tables) >= 0

"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from Medical_KG_rev.models.equation import Equation
from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.models.table import Table, TableCell

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logger = structlog.get_logger(__name__)


# ==============================================================================
# EXCEPTION CLASSES
# ==============================================================================

class MineruOutputParserError(RuntimeError):
    """Raised when MinerU CLI output cannot be parsed.

    This exception is raised when the MinerU output parser encounters
    invalid or malformed data that cannot be processed. Common causes
    include missing required fields, invalid JSON, or type conversion
    errors.

    Example:
        >>> parser = MineruOutputParser()
        >>> try:
        ...     document = parser.parse_path(Path("invalid.json"))
        ... except MineruOutputParserError as e:
        ...     print(f"Parsing failed: {e}")

    """

    pass


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass(slots=True)
class ParsedBlock:
    """Represents a parsed text block from MinerU output.

    This dataclass represents a single text block extracted by MinerU,
    including its position, content, and metadata. Blocks can be
    associated with tables, figures, or equations through their IDs.

    Attributes:
        id: Unique identifier for the block
        page: Page number where the block appears
        type: Type of block (paragraph, heading, etc.)
        text: Text content of the block
        bbox: Bounding box coordinates (x, y, width, height)
        confidence: Confidence score for the extraction
        reading_order: Reading order within the document
        metadata: Additional metadata for the block
        table_id: Associated table identifier
        figure_id: Associated figure identifier
        equation_id: Associated equation identifier

    Invariants:
        - id is not empty
        - page >= 1
        - bbox has 4 elements if not None
        - confidence is between 0.0 and 1.0 if not None

    Example:
        >>> block = ParsedBlock(
        ...     id="blk-1",
        ...     page=1,
        ...     type="paragraph",
        ...     text="Sample text",
        ...     bbox=(10.0, 20.0, 100.0, 30.0),
        ...     confidence=0.95,
        ...     reading_order=1,
        ...     metadata={"source": "mineru"}
        ... )

    """

    id: str
    page: int
    type: str
    text: str | None
    bbox: tuple[float, float, float, float] | None
    confidence: float | None
    reading_order: int | None
    metadata: dict[str, Any]
    table_id: str | None = None
    figure_id: str | None = None
    equation_id: str | None = None


@dataclass(slots=True)
class ParsedDocument:
    """Aggregates all parsed content for a document.

    This dataclass represents a complete parsed document from MinerU,
    including all blocks, tables, figures, and equations extracted
    from the source document.

    Attributes:
        document_id: Unique identifier for the document
        blocks: List of parsed text blocks
        tables: List of extracted tables
        figures: List of extracted figures
        equations: List of extracted equations
        metadata: Document-level metadata

    Invariants:
        - document_id is not empty
        - All blocks have unique IDs
        - All tables have unique IDs
        - All figures have unique IDs
        - All equations have unique IDs

    Example:
        >>> document = ParsedDocument(
        ...     document_id="doc-1",
        ...     blocks=[block1, block2],
        ...     tables=[table1],
        ...     figures=[figure1],
        ...     equations=[equation1],
        ...     metadata={"format": "pdf"}
        ... )

    """

    document_id: str
    blocks: list[ParsedBlock]
    tables: list[Table]
    figures: list[Figure]
    equations: list[Equation]
    metadata: dict[str, Any]


# ==============================================================================
# PARSER IMPLEMENTATION
# ==============================================================================

class MineruOutputParser:
    """Parses structured JSON output emitted by MinerU CLI.

    This class provides methods for parsing MinerU CLI output in various
    formats, including JSON files and Markdown text. It handles validation,
    type conversion, and error handling for MinerU output data.

    Attributes:
        None: This class is stateless

    Thread Safety:
        - Thread-safe: Parser instances are stateless
        - All methods are pure functions

    Example:
        >>> parser = MineruOutputParser()
        >>> document = parser.parse_path(Path("output.json"))
        >>> assert document.document_id is not None

    """

    def parse_json(self, json_content: str) -> ParsedDocument:
        """Parse MinerU output from JSON content string.

        Args:
            json_content: MinerU output as JSON string

        Returns:
            Parsed document with all extracted content

        Raises:
            MineruOutputParserError: If JSON is invalid

        Example:
            >>> parser = MineruOutputParser()
            >>> document = parser.parse_json('{"blocks": [...]}')
            >>> assert len(document.blocks) > 0

        """
        try:
            data = json.loads(json_content)
            return self.parse_dict(data)
        except json.JSONDecodeError as exc:
            raise MineruOutputParserError(f"Invalid JSON content: {exc}") from exc

    def parse_path(self, path: Path) -> ParsedDocument:
        """Parse MinerU output from a file path.

        Args:
            path: Path to the MinerU output JSON file

        Returns:
            Parsed document with all extracted content

        Raises:
            MineruOutputParserError: If file is missing or invalid

        Example:
            >>> parser = MineruOutputParser()
            >>> document = parser.parse_path(Path("output.json"))
            >>> assert len(document.blocks) > 0

        """
        if not path.exists():
            raise MineruOutputParserError(f"MinerU output file not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MineruOutputParserError(f"Invalid MinerU JSON output: {exc}") from exc
        return self.parse_dict(payload)

    def parse_dict(self, payload: dict[str, Any] | list[Any]) -> ParsedDocument:
        """Parse MinerU output from a dictionary or list.

        Args:
            payload: Dictionary or list containing MinerU output data
                    - Old format (dict): {"document_id": "...", "blocks": [...]}
                    - New format (list): [[{block}, {block}], [{block}]]  # pages of blocks

        Returns:
            Parsed document with all extracted content

        Raises:
            MineruOutputParserError: If required fields are missing

        Example:
            >>> parser = MineruOutputParser()
            >>> payload = {"document_id": "doc-1", "blocks": [], "tables": []}
            >>> document = parser.parse_dict(payload)
            >>> assert document.document_id == "doc-1"

        """
        # Handle MinerU v2.5.4+ formats (list)
        if isinstance(payload, list):
            # Check if it's the new content_list format (flat list) or old pages format (nested list)
            if payload and isinstance(payload[0], dict):
                # Flat list with dicts - content_list.json format
                return self._parse_content_list_format(payload)
            else:
                # Nested list - old _model.json format (list of pages)
                return self._parse_pages_format(payload)

        # Handle old format (dict)
        document_id = str(payload.get("document_id", ""))
        if not document_id:
            raise MineruOutputParserError("MinerU output missing 'document_id'")
        blocks = [self._parse_block(entry) for entry in payload.get("blocks", [])]
        tables = [self._parse_table(entry) for entry in payload.get("tables", [])]
        figures = [self._parse_figure(entry) for entry in payload.get("figures", [])]
        equations = [self._parse_equation(entry) for entry in payload.get("equations", [])]
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"value": metadata}
        parsed = ParsedDocument(
            document_id=document_id,
            blocks=blocks,
            tables=tables,
            figures=figures,
            equations=equations,
            metadata=metadata,
        )
        logger.debug(
            "mineru.output.parsed",
            document_id=document_id,
            blocks=len(blocks),
            tables=len(tables),
            figures=len(figures),
            equations=len(equations),
        )
        return parsed

    def _parse_content_list_format(self, content_list: list[dict[str, Any]]) -> ParsedDocument:
        """Parse MinerU _content_list.json format with hierarchy.

        Args:
            content_list: Flat list of content blocks with hierarchy metadata

        Returns:
            Parsed document with all extracted content and hierarchy preserved

        Note:
            content_list.json format includes:
            - text_level: heading hierarchy (1=H1, 2=H2, etc.)
            - page_idx: page number
            - bbox: bounding box
            - type: block type (text, image, table)
            - text: content
            - img_path, image_caption: for images

        """
        all_blocks: list[ContentBlock] = []
        all_tables: list[Table] = []
        all_figures: list[Figure] = []

        import uuid

        for block_data in content_list:
            block_type = block_data.get("type", "text")
            content = block_data.get("text", "")
            bbox = block_data.get("bbox", [])
            page_num = block_data.get("page_idx", 0)
            text_level = block_data.get("text_level")  # Hierarchy info!

            # Generate UUID-based unique IDs
            block_id = str(uuid.uuid4())

            # Build metadata dict with hierarchy info
            metadata = {}
            if text_level is not None:
                metadata["text_level"] = text_level  # Preserve heading level
                metadata["is_heading"] = True

            # Map content_list types to internal types
            if block_type == "image":
                # Images/figures
                figure = self._parse_figure({
                    "id": block_id,
                    "content": content,
                    "bbox": bbox,
                    "page_number": page_num,
                    "metadata": {
                        **metadata,
                        "mineru_type": block_type,
                        "img_path": block_data.get("img_path"),
                        "image_caption": block_data.get("image_caption", []),
                        "image_footnote": block_data.get("image_footnote", []),
                    }
                })
                all_figures.append(figure)
            elif block_type == "table":
                # Tables
                table = self._parse_table({
                    "id": block_id,
                    "content": content,
                    "bbox": bbox,
                    "page_number": page_num,
                    "metadata": {**metadata, "mineru_type": block_type}
                })
                all_tables.append(table)
            else:
                # Text blocks (with or without heading level)
                block = self._parse_block({
                    "id": block_id,
                    "block_type": "header" if text_level else "text",
                    "text": content,
                    "bbox": bbox,
                    "page": page_num,
                    "metadata": metadata,
                })
                all_blocks.append(block)

        # Generate document_id from content hash
        content_str = "".join(b.text or "" for b in all_blocks[:10] if b.text)
        if not content_str and all_blocks:
            content_str = str(len(all_blocks))

        import hashlib
        document_id = f"mineru_{hashlib.md5(content_str.encode()).hexdigest()[:12]}"

        parsed = ParsedDocument(
            document_id=document_id,
            blocks=all_blocks,
            tables=all_tables,
            figures=all_figures,
            equations=[],
            metadata={"format": "content_list", "total_blocks": len(content_list)},
        )

        logger.debug(
            "mineru.output.parsed_content_list",
            document_id=document_id,
            blocks=len(all_blocks),
            tables=len(all_tables),
            figures=len(all_figures),
            headings=sum(1 for b in all_blocks if b.metadata.get("is_heading")),
        )
        return parsed

    def _parse_pages_format(self, pages: list[list[dict[str, Any]]]) -> ParsedDocument:
        """Parse MinerU v2.5.4+ output format (list of pages).

        Args:
            pages: List of pages, where each page is a list of content blocks

        Returns:
            Parsed document with all extracted content

        Note:
            MinerU v2.5.4 outputs: [[{block}, {block}], [{block}]]
            Each block has: type, content, bbox, angle

        """
        all_blocks: list[ContentBlock] = []
        all_tables: list[Table] = []
        all_figures: list[Figure] = []

        import uuid

        for page_num, page_blocks in enumerate(pages):
            for block_data in page_blocks:
                block_type = block_data.get("type", "text")
                content = block_data.get("content", "")
                bbox = block_data.get("bbox", [])

                # Generate UUID-based unique IDs for each block
                block_id = str(uuid.uuid4())

                # Map MinerU v2.5.4 types to our internal types
                if block_type in ("table", "table_caption"):
                    # Tables in new format
                    table = self._parse_table({
                        "id": block_id,
                        "content": content,
                        "bbox": bbox,
                        "page_number": page_num,
                        "metadata": {"mineru_type": block_type}
                    })
                    all_tables.append(table)
                elif block_type in ("image", "figure", "figure_caption"):
                    # Figures in new format
                    figure = self._parse_figure({
                        "id": block_id,
                        "content": content,
                        "bbox": bbox,
                        "page_number": page_num,
                        "metadata": {"mineru_type": block_type}
                    })
                    all_figures.append(figure)
                else:
                    # Everything else is a content block
                    block = self._parse_block({
                        "id": block_id,
                        "block_type": block_type,  # header, title, text, etc.
                        "text": content,  # Use 'text' instead of 'content' for ParsedBlock
                        "bbox": bbox,
                        "page": page_num,
                        "metadata": {"angle": block_data.get("angle", 0)}
                    })
                    all_blocks.append(block)

        # Generate a document_id from content hash if not provided
        import hashlib
        content_str = "".join(b.text or "" for b in all_blocks[:10] if b.text)  # First 10 blocks with text
        if not content_str and all_blocks:
            content_str = str(len(all_blocks))  # Fallback to block count
        document_id = f"mineru_{hashlib.md5(content_str.encode()).hexdigest()[:12]}"

        parsed = ParsedDocument(
            document_id=document_id,
            blocks=all_blocks,
            tables=all_tables,
            figures=all_figures,
            equations=[],  # v2.5.4 doesn't separate equations
            metadata={"format": "mineru_v2.5.4", "pages": len(pages)}
        )

        logger.debug(
            "mineru.output.parsed_v2.5.4",
            document_id=document_id,
            pages=len(pages),
            blocks=len(all_blocks),
            tables=len(all_tables),
            figures=len(all_figures),
        )
        return parsed

    def parse_markdown(self, document_id: str, markdown: str) -> ParsedDocument:
        """Parse Markdown text into a document structure.

        Args:
            document_id: Unique identifier for the document
            markdown: Markdown text content to parse

        Returns:
            Parsed document with blocks created from Markdown lines

        Example:
            >>> parser = MineruOutputParser()
            >>> markdown = "# Title\\n\\nParagraph text"
            >>> document = parser.parse_markdown("doc-1", markdown)
            >>> assert len(document.blocks) == 2

        """
        blocks: list[ParsedBlock] = []
        for idx, line in enumerate(markdown.splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            blocks.append(
                ParsedBlock(
                    id=f"blk-md-{idx}",
                    page=1,
                    type="paragraph",
                    text=text,
                    bbox=None,
                    confidence=1.0,
                    reading_order=idx,
                    metadata={"source": "markdown"},
                )
            )
        return ParsedDocument(
            document_id=document_id,
            blocks=blocks,
            tables=[],
            figures=[],
            equations=[],
            metadata={"format": "markdown"},
        )

    def _parse_block(self, payload: dict[str, Any]) -> ParsedBlock:
        """Parse a single block from MinerU output.

        Args:
            payload: Dictionary containing block data

        Returns:
            Parsed block with validated data

        Example:
            >>> parser = MineruOutputParser()
            >>> payload = {"id": "blk-1", "page": 1, "type": "paragraph", "text": "Hello"}
            >>> block = parser._parse_block(payload)
            >>> assert block.id == "blk-1"

        """
        bbox_value = payload.get("bbox")
        bbox = tuple(float(value) for value in bbox_value) if bbox_value else None
        cleaned_bbox = bbox if bbox and len(bbox) == 4 else None
        return ParsedBlock(
            id=str(payload.get("id")),
            page=int(payload.get("page", 1)),
            type=str(payload.get("type", "paragraph")),
            text=payload.get("text"),
            bbox=cleaned_bbox,
            confidence=self._maybe_float(payload.get("confidence")),
            reading_order=self._maybe_int(payload.get("reading_order")),
            metadata=self._ensure_dict(payload.get("metadata", {})),
            table_id=payload.get("table_id"),
            figure_id=payload.get("figure_id"),
            equation_id=payload.get("equation_id"),
        )

    def _parse_table(self, payload: dict[str, Any]) -> Table:
        """Parse a table from MinerU output.

        Args:
            payload: Dictionary containing table data

        Returns:
            Parsed table with cells and metadata

        Example:
            >>> parser = MineruOutputParser()
            >>> payload = {"id": "tbl-1", "page": 1, "cells": [], "headers": []}
            >>> table = parser._parse_table(payload)
            >>> assert table.id == "tbl-1"

        """
        cells = tuple(self._parse_table_cell(cell) for cell in payload.get("cells", []))
        headers = tuple(str(value) for value in payload.get("headers", []))
        bbox_value = payload.get("bbox")
        bbox = tuple(float(value) for value in bbox_value) if bbox_value else None
        return Table(
            id=str(payload.get("id")),
            page=int(payload.get("page", 1)),
            caption=payload.get("caption"),
            cells=cells,
            headers=headers,
            bbox=bbox if bbox and len(bbox) == 4 else None,
            metadata=self._ensure_dict(payload.get("metadata", {})),
        )

    def _parse_table_cell(self, payload: dict[str, Any]) -> TableCell:
        """Parse a table cell from MinerU output.

        Args:
            payload: Dictionary containing cell data

        Returns:
            Parsed table cell with position and content

        Example:
            >>> parser = MineruOutputParser()
            >>> payload = {"row": 0, "column": 0, "content": "Cell text"}
            >>> cell = parser._parse_table_cell(payload)
            >>> assert cell.content == "Cell text"

        """
        bbox_value = payload.get("bbox")
        bbox = tuple(float(value) for value in bbox_value) if bbox_value else None
        return TableCell(
            row=int(payload.get("row", 0)),
            column=int(payload.get("column", 0)),
            content=str(payload.get("content", "")),
            rowspan=int(payload.get("rowspan", 1)),
            colspan=int(payload.get("colspan", 1)),
            bbox=bbox if bbox and len(bbox) == 4 else None,
            confidence=self._maybe_float(payload.get("confidence")),
        )

    def _parse_figure(self, payload: dict[str, Any]) -> Figure:
        """Parse a figure from MinerU output.

        Args:
            payload: Dictionary containing figure data

        Returns:
            Parsed figure with image path and metadata

        Example:
            >>> parser = MineruOutputParser()
            >>> payload = {"id": "fig-1", "page": 1, "image_path": "path/to/image.png"}
            >>> figure = parser._parse_figure(payload)
            >>> assert figure.id == "fig-1"

        """
        bbox_value = payload.get("bbox")
        bbox = tuple(float(value) for value in bbox_value) if bbox_value else None
        return Figure(
            id=str(payload.get("id")),
            page=int(payload.get("page", 1)),
            image_path=str(payload.get("image_path", "")),
            caption=payload.get("caption"),
            bbox=bbox if bbox and len(bbox) == 4 else None,
            figure_type=payload.get("figure_type"),
            mime_type=payload.get("mime_type"),
            width=self._maybe_int(payload.get("width")),
            height=self._maybe_int(payload.get("height")),
            metadata=self._ensure_dict(payload.get("metadata", {})),
        )

    def _parse_equation(self, payload: dict[str, Any]) -> Equation:
        """Parse an equation from MinerU output.

        Args:
            payload: Dictionary containing equation data

        Returns:
            Parsed equation with LaTeX and MathML

        Example:
            >>> parser = MineruOutputParser()
            >>> payload = {"id": "eq-1", "page": 1, "latex": "x^2 + y^2 = z^2"}
            >>> equation = parser._parse_equation(payload)
            >>> assert equation.id == "eq-1"

        """
        bbox_value = payload.get("bbox")
        bbox = tuple(float(value) for value in bbox_value) if bbox_value else None
        return Equation(
            id=str(payload.get("id")),
            page=int(payload.get("page", 1)),
            latex=str(payload.get("latex", "")),
            mathml=payload.get("mathml"),
            bbox=bbox if bbox and len(bbox) == 4 else None,
            display=bool(payload.get("display", True)),
            metadata=self._ensure_dict(payload.get("metadata", {})),
        )

    def _ensure_dict(self, value: Any) -> dict[str, Any]:
        """Ensure value is a dictionary.

        Args:
            value: Value to convert to dictionary

        Returns:
            Dictionary representation of the value

        Example:
            >>> parser = MineruOutputParser()
            >>> result = parser._ensure_dict({"key": "value"})
            >>> assert result == {"key": "value"}
            >>> result = parser._ensure_dict("string")
            >>> assert result == {"value": "string"}

        """
        if isinstance(value, dict):
            return value
        return {"value": value}

    def _maybe_int(self, value: Any) -> int | None:
        """Convert value to integer if possible.

        Args:
            value: Value to convert

        Returns:
            Integer value or None if conversion fails

        Example:
            >>> parser = MineruOutputParser()
            >>> result = parser._maybe_int("123")
            >>> assert result == 123
            >>> result = parser._maybe_int("invalid")
            >>> assert result is None

        """
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _maybe_float(self, value: Any) -> float | None:
        """Convert value to float if possible.

        Args:
            value: Value to convert

        Returns:
            Float value or None if conversion fails

        Example:
            >>> parser = MineruOutputParser()
            >>> result = parser._maybe_float("123.45")
            >>> assert result == 123.45
            >>> result = parser._maybe_float("invalid")
            >>> assert result is None

        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


# ==============================================================================
# EXPORTS
# ==============================================================================


__all__ = [
    "MineruOutputParser",
    "MineruOutputParserError",
    "ParsedBlock",
    "ParsedDocument",
]
