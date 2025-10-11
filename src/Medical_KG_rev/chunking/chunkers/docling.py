"""Docling-based chunking strategies."""

from __future__ import annotations

from collections.abc import Iterable

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter


class DoclingChunker(BaseChunker):
    """Chunker that uses Docling VLM for document processing."""

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize the Docling chunker."""
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        """Chunk a document using Docling VLM processing."""
        try:
            from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMService
        except ImportError as exc:
            raise ChunkerConfigurationError("Docling VLM service not available") from exc

        # Extract text content from document
        contexts = [ctx for ctx in self.normalizer.iter_block_contexts(document) if ctx.text]
        if not contexts:
            return []

        # Process with Docling VLM
        docling_service = DoclingVLMService()

        # Create a temporary document for processing
        temp_doc = Document(
            id=document.id,
            title=document.title or "Untitled",
            content=[ctx.text for ctx in contexts],
            metadata=document.metadata,
        )

        try:
            result = docling_service.process_document(temp_doc)

            # Create blocks from Docling result
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

            # Add table blocks
            if result.tables:
                for idx, table in enumerate(result.tables):
                    blocks.append(
                        Block(
                            id=f"{result.document_id}-table-{idx}",
                            type=BlockType.TABLE,
                            text=table.text,
                            metadata={"source": "docling_vlm", "table_index": idx, "table": table.model_dump()},
                        )
                    )

            # Add figure blocks
            if result.figures:
                for idx, figure in enumerate(result.figures):
                    blocks.append(
                        Block(
                            id=f"{result.document_id}-figure-{idx}",
                            type=BlockType.FIGURE,
                            text=figure.caption or "",
                            metadata={"source": "docling_vlm", "figure_index": idx, "figure": figure.model_dump()},
                        )
                    )

            # Group blocks into chunks
            chunks = []
            current_chunk = []
            current_size = 0

            for block in blocks:
                block_size = self.counter.count_tokens(block.text)
                if current_size + block_size > self.chunk_size and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = [block]
                    current_size = block_size
                else:
                    current_chunk.append(block)
                    current_size += block_size

            if current_chunk:
                chunks.append(current_chunk)

            # Create chunk objects
            assembler = ChunkAssembler(
                document,
                tenant_id=tenant_id,
                chunker_name="docling",
                chunker_version="v1",
                granularity=granularity or "paragraph",
                token_counter=self.counter,
            )

            result_chunks = []
            for chunk_blocks in chunks:
                chunk_text = " ".join(block.text for block in chunk_blocks)
                chunk_meta = {
                    "segment_type": "docling",
                    "block_count": len(chunk_blocks),
                    "token_count": self.counter.count_tokens(chunk_text),
                    "blocks": [block.model_dump() for block in chunk_blocks],
                }

                # Create a simple context for the chunk
                from ..provenance import BlockContext
                context = BlockContext(
                    text=chunk_text,
                    block_id=f"docling-chunk-{len(result_chunks)}",
                    block_type="docling_chunk",
                    metadata=chunk_meta,
                )

                result_chunks.append(assembler.build([context], metadata=chunk_meta))

            return result_chunks

        except Exception as exc:
            raise ChunkerConfigurationError(f"Docling processing failed: {exc}") from exc

    def explain(self) -> dict[str, object]:
        """Explain the chunking strategy."""
        return {
            "strategy": "docling",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
