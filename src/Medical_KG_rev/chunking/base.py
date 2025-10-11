"""Shared abstractions for contextual chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

from Medical_KG_rev.models.ir import Block, Document

if TYPE_CHECKING:
    import numpy as np

from .assembly import ChunkAssembler
from .exceptions import ChunkerConfigurationError
from .models import Chunk, Granularity
from .ports import BaseChunker
from .provenance import BlockContext, ProvenanceNormalizer
from .segmentation import Segment, Segmenter
from .tokenization import TokenCounter, default_token_counter


class ContextualChunker(BaseChunker, ABC):
    """Base class with common assembly logic for block-context chunkers."""

    default_granularity: Granularity = "paragraph"
    segment_type: str | None = None
    include_tables: bool = False

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        segmenter: Segmenter | None = None,
        context_filter: Callable[[BlockContext], bool] | None = None,
    ) -> None:
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)
        self._segmenter = segmenter
        self._context_filter = context_filter

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Sequence[Block] | None = None,
    ) -> list[Chunk]:
        contexts = self.prepare_contexts(document, blocks=blocks)
        return self.chunk_with_contexts(
            document,
            contexts,
            tenant_id=tenant_id,
            granularity=granularity,
            blocks=blocks,
        )

    def prepare_contexts(
        self,
        document: Document,
        *,
        blocks: Sequence[Block] | None = None,
    ) -> list[BlockContext]:
        allowed_blocks: set[str] | None = None
        if blocks is not None:
            allowed_blocks = {
                str(getattr(block, "id", index) or index) for index, block in enumerate(blocks)
            }
        contexts: list[BlockContext] = []
        for context in self.normalizer.iter_block_contexts(document):
            if allowed_blocks is not None and str(context.block.id) not in allowed_blocks:
                continue
            if self._accept_context(context):
                contexts.append(context)
        return contexts

    # Backwards compatibility for subclasses/tests that call collect_contexts.
    def collect_contexts(
        self,
        document: Document,
        *,
        blocks: Sequence[Block] | None = None,
    ) -> list[BlockContext]:
        return self.prepare_contexts(document, blocks=blocks)

    def chunk_with_contexts(
        self,
        document: Document,
        contexts: Sequence[BlockContext],
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Sequence[Block] | None = None,
    ) -> list[Chunk]:
        if not contexts:
            return []
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or self.default_granularity,
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        for segment in self.segment_document(document, contexts, blocks=blocks):
            if not segment.contexts:
                continue
            metadata = self._merge_metadata(segment.metadata)
            chunks.append(assembler.build(list(segment.contexts), metadata=metadata))
        return chunks

    def segment_document(
        self,
        document: Document,
        contexts: Sequence[BlockContext],
        *,
        blocks: Sequence[Block] | None = None,
    ) -> Iterable[Segment]:
        """Segment the document after context normalization."""
        if self._segmenter is not None:
            return self._segmenter.plan(contexts)
        return self.segment_contexts(contexts)

    def _accept_context(self, context: BlockContext) -> bool:
        if not context.text:
            return False
        if not self.include_tables and context.is_table:
            return False
        if self._context_filter is not None and not self._context_filter(context):
            return False
        return True

    def _merge_metadata(self, metadata: dict[str, object] | None) -> dict[str, object]:
        merged: dict[str, object] = {}
        if metadata:
            merged.update(metadata)
        if self.segment_type and "segment_type" not in merged:
            merged["segment_type"] = self.segment_type
        return merged

    def explain(self) -> dict[str, object]:  # pragma: no cover - overridable default
        return {}

    @abstractmethod
    def segment_contexts(self, contexts: Sequence[BlockContext]) -> Iterable[Segment]:
        """Yield contiguous segments of contexts for chunk assembly."""


def resolve_sentence_encoder(
    model_name: str,
    *,
    gpu_semantic_checks: bool = False,  # Deprecated parameter, kept for compatibility
    encoder: object | None,
) -> object:
    """Resolve a sentence transformer encoder without GPU placement.

    Note: GPU semantic checks are no longer supported in the torch-free architecture.
    Use Docling's built-in chunking capabilities instead.
    """
    if encoder is not None:
        return encoder
    if gpu_semantic_checks:
        raise ChunkerConfigurationError(
            "GPU semantic checks are no longer supported. Use Docling's built-in chunking capabilities instead."
        )
    try:  # pragma: no cover - optional dependency
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        message = "sentence-transformers must be installed for semantic chunkers"

        class _MissingEncoder:
            def encode(self, *_args, **_kwargs):
                raise ChunkerConfigurationError(message) from exc

        return _MissingEncoder()
    resolved = SentenceTransformer(model_name)
    return resolved


class EmbeddingContextualChunker(ContextualChunker, ABC):
    """Contextual chunker base class that provides embedding utilities."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        gpu_semantic_checks: bool = False,  # Deprecated parameter, kept for compatibility
        encoder: object | None = None,
        segmenter: Segmenter | None = None,
        context_filter: Callable[[BlockContext], bool] | None = None,
    ) -> None:
        super().__init__(
            token_counter=token_counter,
            segmenter=segmenter,
            context_filter=context_filter,
        )
        self.model = resolve_sentence_encoder(
            model_name,
            gpu_semantic_checks=gpu_semantic_checks,
            encoder=encoder,
        )

    def encode_contexts(self, contexts: Sequence[BlockContext]) -> np.ndarray:
        import numpy as np

        sentences = [ctx.text for ctx in contexts]
        if not sentences:
            return np.empty((0, 1))
        encode = getattr(self.model, "encode", None)
        if encode is None:
            raise ChunkerConfigurationError("Encoder does not expose an encode() method")
        embeddings = encode(sentences, convert_to_numpy=True)
        return np.asarray(embeddings)
