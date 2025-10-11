"""Hybrid chunker with tokenizer alignment for SPLADE compatibility."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import structlog
from Medical_KG_rev.chunking.base import ContextualChunker
from Medical_KG_rev.chunking.models import Chunk, Granularity
from Medical_KG_rev.models.ir import Block, Document

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class ChunkSegment:
    """Represents a segment within a chunk for tokenizer alignment."""

    text: str
    start_char: int
    end_char: int
    token_count: int
    segment_type: str  # "title", "section_header", "paragraph", "table", "caption"


class HybridChunker(ContextualChunker):
    """Hybrid chunker with tokenizer alignment for SPLADE compatibility.

    This chunker implements hierarchy-first segmentation (titles, sections,
    paragraphs, tables) with tokenizer-aware split/merge using SPLADE tokenizer.
    It ensures "measured tokens" in chunking match SPLADE's 512 cap and
    configures chunk sizes: 350-500 tokens, up to 700 when structure warrants.
    """

    default_granularity: Granularity = "paragraph"
    segment_type: str = "hybrid"
    include_tables: bool = True

    def __init__(
        self,
        *,
        max_tokens: int = 512,  # SPLADE limit
        target_chunk_tokens: int = 450,  # Target chunk size
        max_chunk_tokens: int = 700,  # Maximum chunk size
        tokenizer_name: str = "naver/splade-v3",
        **kwargs: Any,
    ) -> None:
        """Initialize hybrid chunker with tokenizer alignment.

        Args:
            max_tokens: Maximum tokens per SPLADE segment (default: 512)
            target_chunk_tokens: Target tokens per chunk (default: 450)
            max_chunk_tokens: Maximum tokens per chunk (default: 700)
            tokenizer_name: Name of the tokenizer to use for alignment
            **kwargs: Additional arguments passed to parent class

        """
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.target_chunk_tokens = target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.tokenizer_name = tokenizer_name
        self._tokenizer: Any = None

    def _get_tokenizer(self) -> Any:
        """Get or initialize the SPLADE tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                logger.info(
                    "hybrid_chunker.tokenizer_loaded",
                    tokenizer_name=self.tokenizer_name,
                )
            except Exception as exc:
                logger.warning(
                    "hybrid_chunker.tokenizer_failed",
                    tokenizer_name=self.tokenizer_name,
                    error=str(exc),
                )
                # Fallback to simple word counting
                self._tokenizer = "fallback"
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the configured tokenizer."""
        tokenizer = self._get_tokenizer()
        if tokenizer == "fallback":
            # Simple word-based estimation (rough approximation)
            return int(len(text.split()) * 1.3)  # Rough token-to-word ratio
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            # Fallback to word counting
            return int(len(text.split()) * 1.3)

    def _create_deterministic_chunk_id(
        self,
        doc_id: str,
        page_no: int,
        element_path: str,
        char_span: tuple[int, int],
    ) -> str:
        """Create deterministic chunk ID from document components.

        Args:
            doc_id: Document identifier
            page_no: Page number
            element_path: Path to element (e.g., "Introduction > Methods")
            char_span: Character span (start, end)

        Returns:
            Deterministic chunk identifier

        """
        # Create deterministic hash from components
        components = f"{doc_id}:{page_no}:{element_path}:{char_span[0]}:{char_span[1]}"
        hash_obj = hashlib.md5(components.encode())
        return f"{doc_id}_chunk_{hash_obj.hexdigest()[:8]}"

    def _segment_chunk_for_splade(self, chunk_text: str) -> list[ChunkSegment]:
        """Split chunk text into ≤512-token segments using SPLADE tokenizer.

        Args:
            chunk_text: Text to segment

        Returns:
            List of segments with token counts

        """
        tokenizer = self._get_tokenizer()
        segments = []

        if tokenizer == "fallback":
            # Simple fallback segmentation
            words = chunk_text.split()
            current_segment: list[str] = []
            current_tokens = 0.0

            for word in words:
                word_tokens = len(word.split()) * 1.3  # Rough estimation
                if current_tokens + word_tokens > self.max_tokens and current_segment:
                    # Finish current segment
                    segment_text = " ".join(current_segment)
                    segments.append(
                        ChunkSegment(
                            text=segment_text,
                            start_char=0,  # Simplified for fallback
                            end_char=len(segment_text),
                            token_count=int(current_tokens),
                            segment_type="paragraph",
                        )
                    )
                    current_segment = [word]
                    current_tokens = word_tokens
                else:
                    current_segment.append(word)
                    current_tokens += word_tokens

            # Add final segment
            if current_segment:
                segment_text = " ".join(current_segment)
                segments.append(
                    ChunkSegment(
                        text=segment_text,
                        start_char=0,
                        end_char=len(segment_text),
                        token_count=int(current_tokens),
                        segment_type="paragraph",
                    )
                )
        else:
            # Use actual tokenizer for precise segmentation
            try:
                tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
                current_segment_tokens: list[int] = []
                current_text = ""
                start_pos = 0

                for token_id in tokens:
                    token_text = tokenizer.decode([token_id])
                    current_segment_tokens.append(token_id)
                    current_text += token_text

                    if len(current_segment_tokens) >= self.max_tokens:
                        # Finish current segment
                        segments.append(
                            ChunkSegment(
                                text=current_text.strip(),
                                start_char=start_pos,
                                end_char=start_pos + len(current_text),
                                token_count=len(current_segment_tokens),
                                segment_type="paragraph",
                            )
                        )
                        start_pos += len(current_text)
                        current_segment_tokens = []
                        current_text = ""

                # Add final segment if any tokens remain
                if current_segment_tokens:
                    segments.append(
                        ChunkSegment(
                            text=current_text.strip(),
                            start_char=start_pos,
                            end_char=start_pos + len(current_text),
                            token_count=len(current_segment_tokens),
                            segment_type="paragraph",
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "hybrid_chunker.segmentation_failed",
                    error=str(exc),
                )
                # Fallback to simple segmentation
                return self._segment_chunk_for_splade(chunk_text)

        return segments

    def _create_contextualized_text(
        self,
        chunk_text: str,
        section_path: str | None = None,
        caption: str | None = None,
    ) -> str:
        """Create contextualized text with section path and caption.

        Args:
            chunk_text: Base chunk text
            section_path: Path to section (e.g., "Introduction > Methods")
            caption: Caption text if available

        Returns:
            Contextualized text for dense embeddings

        """
        parts = []

        if section_path:
            parts.append(f"Section: {section_path}")

        if caption:
            parts.append(f"Caption: {caption}")

        if parts:
            context_prefix = " | ".join(parts) + "\n\n"
            return context_prefix + chunk_text

        return chunk_text

    def _create_content_only_text(
        self,
        chunk_text: str,
        section_path: str | None = None,
        caption: str | None = None,
    ) -> str:
        """Create content-only text without synthetic prefixes.

        Args:
            chunk_text: Base chunk text
            section_path: Path to section (ignored for content-only)
            caption: Caption text if available

        Returns:
            Content-only text for BM25/SPLADE

        """
        # For content-only, we include caption but not section path
        if caption:
            return f"{caption}\n\n{chunk_text}"

        return chunk_text

    def chunk_document(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: list[Block] | None = None,
    ) -> list[Chunk]:
        """Chunk document using hybrid approach with tokenizer alignment.

        Args:
            document: Document to chunk
            tenant_id: Tenant identifier
            granularity: Chunking granularity (ignored for hybrid)
            blocks: Specific blocks to chunk (if None, chunks all blocks)

        Returns:
            List of chunks with tokenizer-aligned segmentation

        """
        if blocks is None:
            # Get blocks from document - adapt to actual Document structure
            if hasattr(document, "body") and hasattr(document.body, "body"):
                blocks = document.body.body
            elif hasattr(document, "blocks"):
                blocks = document.blocks
            else:
                blocks = []

        chunks = []
        current_chunk_text = ""
        current_chunk_tokens = 0
        current_section_path = ""
        current_page_no = 1
        current_char_start = 0

        for block in blocks:
            # Determine block type and processing strategy
            block_type = self._get_block_type(block)
            block_text = self._extract_block_text(block)
            block_tokens = self._count_tokens(block_text)

            # Update section path for hierarchical context
            if block_type in ["title", "section_header"]:
                current_section_path = self._update_section_path(current_section_path, block_text)

            # Check if we should start a new chunk
            should_start_new_chunk = (
                block_type in ["title", "section_header", "table"]
                or current_chunk_tokens + block_tokens > self.max_chunk_tokens
                or (current_chunk_tokens > self.target_chunk_tokens and block_type == "paragraph")
            )

            if should_start_new_chunk and current_chunk_text:
                # Finish current chunk
                chunk = self._create_chunk(
                    doc_id=getattr(document, "document_id", "unknown"),
                    tenant_id=tenant_id,
                    chunk_text=current_chunk_text,
                    section_path=current_section_path,
                    page_no=current_page_no,
                    char_start=current_char_start,
                    char_end=current_char_start + len(current_chunk_text),
                )
                chunks.append(chunk)

                # Reset for new chunk
                current_chunk_text = ""
                current_chunk_tokens = 0
                current_char_start += len(current_chunk_text)

            # Add block to current chunk
            if current_chunk_text:
                current_chunk_text += "\n\n"
            current_chunk_text += block_text
            current_chunk_tokens += block_tokens

            # Update page number
            if hasattr(block, "page") and block.page:
                current_page_no = block.page

        # Add final chunk if any text remains
        if current_chunk_text:
            chunk = self._create_chunk(
                doc_id=getattr(document, "document_id", "unknown"),
                tenant_id=tenant_id,
                chunk_text=current_chunk_text,
                section_path=current_section_path,
                page_no=current_page_no,
                char_start=current_char_start,
                char_end=current_char_start + len(current_chunk_text),
            )
            chunks.append(chunk)

        logger.info(
            "hybrid_chunker.document_chunked",
            doc_id=getattr(document, "document_id", "unknown"),
            chunk_count=len(chunks),
            total_tokens=sum(self._count_tokens(chunk.body) for chunk in chunks),
        )

        return chunks

    def _get_block_type(self, block: Block) -> str:
        """Determine the type of a block for processing strategy."""
        if hasattr(block, "kind") and block.kind:
            return str(block.kind)
        if hasattr(block, "label") and block.label:
            return str(block.label)
        return "paragraph"

    def _extract_block_text(self, block: Block) -> str:
        """Extract text content from a block."""
        if hasattr(block, "text") and block.text:
            return str(block.text)
        if hasattr(block, "content") and block.content:
            return str(block.content)
        return ""

    def _update_section_path(self, current_path: str, header_text: str) -> str:
        """Update section path based on header text."""
        if not current_path:
            return str(header_text)
        return f"{current_path} > {header_text}"

    def _create_chunk(
        self,
        doc_id: str,
        tenant_id: str,
        chunk_text: str,
        section_path: str,
        page_no: int,
        char_start: int,
        char_end: int,
    ) -> Chunk:
        """Create a chunk with deterministic ID and contextualized text."""
        # Create deterministic chunk ID
        chunk_id = self._create_deterministic_chunk_id(
            doc_id=doc_id,
            page_no=page_no,
            element_path=section_path,
            char_span=(char_start, char_end),
        )

        # Create contextualized and content-only text
        contextualized_text = self._create_contextualized_text(chunk_text, section_path)
        content_only_text = self._create_content_only_text(chunk_text, section_path)

        # Store both versions in metadata
        meta = {
            "contextualized_text": contextualized_text,
            "content_only_text": content_only_text,
            "section_path": section_path,
            "token_count": self._count_tokens(chunk_text),
            "chunker_type": "hybrid",
            "tokenizer_name": self.tokenizer_name,
        }

        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            tenant_id=tenant_id,
            body=chunk_text,  # Use original text as body
            title_path=tuple(section_path.split(" > ") if section_path else []),
            section=section_path,
            start_char=char_start,
            end_char=char_end,
            granularity="paragraph",
            chunker="hybrid_chunker",
            chunker_version="1.0.0",
            page_no=page_no,
            meta=meta,
        )
