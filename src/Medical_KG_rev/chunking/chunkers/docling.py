"""Docling-based chunker leveraging Docling's built-in chunking capabilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMResult

from ..base import ContextualChunker
from ..provenance import BlockContext
from ..segmentation import Segment
from ..tokenization import TokenCounter


class DoclingChunker(ContextualChunker):
    """Chunker that uses Docling's output without requiring torch in main gateway.

    Docling dependency (docling[vlm]>=2.0.0) is already present and runs in
    its own GPU-enabled container. This class consumes Docling's output.
    """

    name = "docling"
    version = "v1"
    segment_type = "docling"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        min_chunk_size: int = 64,
        max_chunk_size: int = 2048,
        overlap_ratio: float = 0.1,
    ) -> None:
        super().__init__(token_counter=token_counter)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_ratio = overlap_ratio

    def segment_contexts(self, contexts: Sequence[BlockContext]) -> Iterable[Segment]:
        """Segment contexts using Docling's semantic understanding."""
        if not contexts:
            return []

        segments: list[Segment] = []
        current_segment: list[BlockContext] = []
        current_tokens = 0

        for context in contexts:
            context_tokens = context.token_count

            # Check if adding this context would exceed max size
            if current_tokens + context_tokens > self.max_chunk_size and current_segment:
                # Create segment from current contexts
                if current_tokens >= self.min_chunk_size:
                    segments.append(Segment(contexts=list(current_segment)))

                # Start new segment with overlap
                overlap_contexts = self._get_overlap_contexts(current_segment)
                current_segment = list(overlap_contexts)
                current_tokens = sum(ctx.token_count for ctx in current_segment)

            current_segment.append(context)
            current_tokens += context_tokens

        # Add final segment if it meets minimum size
        if current_segment and current_tokens >= self.min_chunk_size:
            segments.append(Segment(contexts=list(current_segment)))

        return segments

    def _get_overlap_contexts(self, contexts: list[BlockContext]) -> list[BlockContext]:
        """Get overlap contexts for smooth transitions between chunks."""
        if not contexts:
            return []

        overlap_tokens = int(self.max_chunk_size * self.overlap_ratio)
        overlap_contexts: list[BlockContext] = []
        current_tokens = 0

        # Add contexts from the end until we reach overlap size
        for context in reversed(contexts):
            if current_tokens + context.token_count > overlap_tokens:
                break
            overlap_contexts.insert(0, context)
            current_tokens += context.token_count

        return overlap_contexts

    def explain(self) -> dict[str, object]:
        return {
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "overlap_ratio": self.overlap_ratio,
        }


class DoclingVLMChunker(ContextualChunker):
    """Chunker that directly processes DoclingVLMResult without torch dependencies."""

    name = "docling_vlm"
    version = "v1"
    segment_type = "docling_vlm"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        preserve_structure: bool = True,
        include_tables: bool = True,
        include_figures: bool = True,
    ) -> None:
        super().__init__(token_counter=token_counter)
        self.preserve_structure = preserve_structure
        self.include_tables = include_tables
        self.include_figures = include_figures

    def chunk_from_docling_result(
        self,
        result: DoclingVLMResult,
        *,
        tenant_id: str,
    ) -> list:
        """Chunk a DoclingVLMResult directly without requiring torch."""
        from Medical_KG_rev.models.ir import Block, BlockType, Document, Section


        # Create document from Docling result
        blocks: list[Block] = []

        # Add text blocks
        if result.text:
            paragraphs = [p.strip() for p in result.text.split("\n\n") if p.strip()]
            for idx, paragraph in enumerate(paragraphs):
                blocks.append(
                    Block(
                        id=f"{result.document_id}-text-{idx}",
                        type=BlockType.PARAGRAPH,
                        text=paragraph,
                        metadata={"source": "docling_vlm", "paragraph_index": idx},
                    )
                )

        # Add table blocks if requested
        if self.include_tables and result.tables:
            for idx, table_data in enumerate(result.tables):
                table_text = self._format_table(table_data)
                blocks.append(
                    Block(
                        id=f"{result.document_id}-table-{idx}",
                        type=BlockType.TABLE,
                        text=table_text,
                        metadata={"source": "docling_vlm", "table_index": idx},
                    )
                )

        # Add figure blocks if requested
        if self.include_figures and result.figures:
            for idx, figure_data in enumerate(result.figures):
                caption = figure_data.get("caption", "")
                blocks.append(
                    Block(
                        id=f"{result.document_id}-figure-{idx}",
                        type=BlockType.FIGURE,
                        text=caption,
                        metadata={"source": "docling_vlm", "figure_index": idx},
                    )
                )

        # Create document
        section = Section(
            id=f"{result.document_id}-section-0",
            title=result.metadata.get("title"),
            blocks=blocks,
        )

        document = Document(
            id=result.document_id,
            source="docling_vlm",
            title=result.metadata.get("title"),
            sections=[section],
            metadata=result.metadata,
        )

        # Use base chunking logic
        contexts = self.prepare_contexts(document)
        return self.chunk_with_contexts(
            document,
            contexts,
            tenant_id=tenant_id,
            granularity="paragraph",
        )

    def _format_table(self, table_data: dict) -> str:
        """Format table data as markdown text."""
        headers = table_data.get("headers", [])
        cells = table_data.get("cells", [])

        if not cells:
            return ""

        # Create markdown table
        lines = []

        # Add headers if available
        if headers:
            lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")

        # Add rows
        max_row = max(cell.get("row", 0) for cell in cells) if cells else 0
        for row_idx in range(max_row + 1):
            row_cells = [cell for cell in cells if cell.get("row") == row_idx]
            if row_cells:
                row_text = []
                for col_idx in range(
                    len(headers)
                    if headers
                    else max(cell.get("column", 0) for cell in row_cells) + 1
                ):
                    cell = next((c for c in row_cells if c.get("column") == col_idx), None)
                    cell_text = cell.get("text", "") if cell else ""
                    row_text.append(cell_text)
                lines.append("| " + " | ".join(row_text) + " |")

        return "\n".join(lines)

    def segment_contexts(self, contexts: Sequence[BlockContext]) -> Iterable[Segment]:
        """Segment contexts preserving Docling's structure."""
        if not contexts:
            return []

        segments: list[Segment] = []

        # Group contexts by type for better semantic coherence
        current_segment: list[BlockContext] = []
        current_type = None

        for context in contexts:
            context_type = context.metadata.get("source", "unknown")

            # Start new segment if type changes and we have content
            if current_type is not None and current_type != context_type and current_segment:
                segments.append(Segment(contexts=list(current_segment)))
                current_segment = []

            current_segment.append(context)
            current_type = context_type

        # Add final segment
        if current_segment:
            segments.append(Segment(contexts=list(current_segment)))

        return segments

    def explain(self) -> dict[str, object]:
        return {
            "preserve_structure": self.preserve_structure,
            "include_tables": self.include_tables,
            "include_figures": self.include_figures,
        }
