"""Hybrid chunking strategies combining multiple approaches."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from transformers import AutoTokenizer

from Medical_KG_rev.models.ir import Document

from .assembly import ChunkAssembler
from .exceptions import ChunkerConfigurationError
from .models import Chunk, Granularity
from .ports import BaseChunker
from .provenance import ProvenanceNormalizer
from .tokenization import TokenCounter, default_token_counter

logger = logging.getLogger(__name__)


class HybridChunker(BaseChunker):
    """Chunker that combines multiple chunking strategies."""

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer_name: str = "bert-base-uncased",
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize the hybrid chunker."""
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer_name = tokenizer_name
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

        # Initialize tokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            logger.info(
                "hybrid_chunker.tokenizer_loaded",
                tokenizer_name=self.tokenizer_name,
            )
        except Exception as exc:
            logger.error(
                "hybrid_chunker.tokenizer_failed",
                tokenizer_name=self.tokenizer_name,
                error=str(exc),
            )
            raise ChunkerConfigurationError(
                f"Failed to load tokenizer '{self.tokenizer_name}' for hybrid chunker"
            ) from exc

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        """Chunk a document using hybrid strategies."""
        contexts = [ctx for ctx in self.normalizer.iter_block_contexts(document) if ctx.text]
        if not contexts:
            return []

        # Extract text content
        text_content = " ".join(ctx.text for ctx in contexts)

        # Apply hybrid chunking strategies
        chunks = self._hybrid_chunking(text_content)

        # Create chunk objects
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name="hybrid",
            chunker_version="v1",
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )

        result_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_meta = {
                "segment_type": "hybrid",
                "chunk_index": i,
                "token_count": self.counter.count_tokens(chunk_text),
                "strategy": "hybrid",
            }

            # Find contexts that correspond to this chunk
            chunk_contexts = []
            for ctx in contexts:
                if ctx.text in chunk_text:
                    chunk_contexts.append(ctx)

            if chunk_contexts:
                result_chunks.append(assembler.build(chunk_contexts, metadata=chunk_meta))

        return result_chunks

    def _hybrid_chunking(self, text: str) -> list[str]:
        """Apply hybrid chunking strategies."""
        chunks = []

        # Strategy 1: Sentence-based chunking
        sentences = self._split_into_sentences(text)

        # Strategy 2: Token-based chunking
        token_chunks = self._token_based_chunking(text)

        # Strategy 3: Semantic chunking (simplified)
        semantic_chunks = self._semantic_chunking(text)

        # Combine strategies and select best chunks
        all_chunks = sentences + token_chunks + semantic_chunks

        # Remove duplicates and merge overlapping chunks
        chunks = self._merge_chunks(all_chunks)

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = self.counter.count_tokens(sentence)
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _token_based_chunking(self, text: str) -> list[str]:
        """Chunk text based on token count."""
        try:
            tokens = self._tokenizer.tokenize(text)
            chunks = []
            current_chunk = []
            current_size = 0

            for token in tokens:
                if current_size >= self.chunk_size and current_chunk:
                    chunks.append(self._tokenizer.convert_tokens_to_string(current_chunk))
                    current_chunk = [token]
                    current_size = 1
                else:
                    current_chunk.append(token)
                    current_size += 1

            if current_chunk:
                chunks.append(self._tokenizer.convert_tokens_to_string(current_chunk))

            return chunks
        except Exception as exc:
            logger.error("hybrid_chunker.token_chunk_failed", error=str(exc))
            raise ChunkerConfigurationError("Token-based chunking failed") from exc

    def _semantic_chunking(self, text: str) -> list[str]:
        """Simplified semantic chunking based on paragraph breaks."""
        paragraphs = text.split('\n\n')
        chunks = []

        for paragraph in paragraphs:
            if paragraph.strip():
                chunks.append(paragraph.strip())

        return chunks

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        """Merge overlapping chunks and remove duplicates."""
        if not chunks:
            return []

        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)

        return unique_chunks

    def explain(self) -> dict[str, object]:
        """Explain the chunking strategy."""
        return {
            "strategy": "hybrid",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "tokenizer_name": self.tokenizer_name,
        }
