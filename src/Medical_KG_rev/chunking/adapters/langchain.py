"""LangChain-based chunker wrappers."""

from __future__ import annotations

from collections.abc import Iterable

from langchain.text_splitter import (  # type: ignore
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
)

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from .mapping import OffsetMapper


def _create_text_splitter(class_name: str, **kwargs):
    """Create a LangChain text splitter instance."""
    try:
        from langchain.text_splitter import (
            HTMLHeaderTextSplitter,
            MarkdownHeaderTextSplitter,
            NLTKTextSplitter,
            RecursiveCharacterTextSplitter,
            SpacyTextSplitter,
            TokenTextSplitter,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ChunkerConfigurationError(
            "langchain must be installed to use LangChain chunkers"
        ) from exc
    mapping = {
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
        "TokenTextSplitter": TokenTextSplitter,
        "MarkdownHeaderTextSplitter": MarkdownHeaderTextSplitter,
        "HTMLHeaderTextSplitter": HTMLHeaderTextSplitter,
        "NLTKTextSplitter": NLTKTextSplitter,
        "SpacyTextSplitter": SpacyTextSplitter,
    }
    splitter_cls = mapping.get(class_name)
    if splitter_cls is None:
        raise ChunkerConfigurationError(f"Unsupported LangChain splitter '{class_name}'")
    return splitter_cls(**kwargs)


class _BaseLangChainChunker(BaseChunker):
    framework = "langchain"
    version = "v1"

    def __init__(
        self,
        *,
        splitter,
        name: str,
        splitter_name: str,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.name = name
        self.splitter = splitter
        self.splitter_name = splitter_name
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def _split(self, text: str) -> list[tuple[str, dict[str, object]]]:
        split = getattr(self.splitter, "create_documents", None)
        if callable(split):
            documents = split([text])
            segments: list[tuple[str, dict[str, object]]] = []
            for doc in documents:
                body = getattr(doc, "page_content", None) or getattr(doc, "text", "")
                if not body:
                    continue
                metadata = getattr(doc, "metadata", {}) or {}
                segments.append((str(body), dict(metadata)))
            return segments
        split = getattr(self.splitter, "split_text", None)
        if callable(split):
            pieces = split(text)
            return [(piece, {}) for piece in pieces if piece]
        raise ChunkerConfigurationError("LangChain splitter does not expose split method")

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        contexts = [ctx for ctx in self.normalizer.iter_block_contexts(document) if ctx.text]
        if not contexts:
            return []
        mapper = OffsetMapper(contexts, token_counter=self.counter)
        aggregated_text = mapper.aggregated_text
        segments = self._split(aggregated_text)
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        cursor = 0
        for text_segment, metadata in segments:
            projection = mapper.project(text_segment, start_hint=cursor)
            cursor = projection.end_offset
            if not projection.contexts:
                continue
            chunk_meta = {
                "segment_type": "framework",
                "framework": self.framework,
                "splitter": self.splitter_name,
            }
            if metadata:
                chunk_meta.update(metadata)
            chunks.append(assembler.build(projection.contexts, metadata=chunk_meta))
        return chunks

    def explain(self) -> dict[str, object]:
        return {
            "framework": self.framework,
            "splitter": self.splitter_name,
        }


class LangChainSplitterChunker(_BaseLangChainChunker):
    def __init__(
        self,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        splitter = _create_text_splitter(
            "RecursiveCharacterTextSplitter",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        super().__init__(
            splitter=splitter,
            name="langchain.recursive_character",
            splitter_name="RecursiveCharacterTextSplitter",
            token_counter=token_counter,
        )


class LangChainTokenSplitterChunker(_BaseLangChainChunker):
    def __init__(
        self,
        *,
        encoding_name: str = "cl100k_base",
        chunk_size: int = 256,
        chunk_overlap: int = 20,
        token_counter: TokenCounter | None = None,
    ) -> None:
        splitter = _create_text_splitter(
            "TokenTextSplitter",
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        super().__init__(
            splitter=splitter,
            name="langchain.token",
            splitter_name="TokenTextSplitter",
            token_counter=token_counter,
        )


class LangChainMarkdownChunker(_BaseLangChainChunker):
    def __init__(
        self,
        *,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        splitter = _create_text_splitter(
            "MarkdownHeaderTextSplitter",
            headers_to_split_on=headers_to_split_on
            or [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
        )
        super().__init__(
            splitter=splitter,
            name="langchain.markdown",
            splitter_name="MarkdownHeaderTextSplitter",
            token_counter=token_counter,
        )


class LangChainHTMLChunker(_BaseLangChainChunker):
    def __init__(
        self,
        *,
        headings: tuple[str, ...] = ("h1", "h2", "h3"),
        token_counter: TokenCounter | None = None,
    ) -> None:
        splitter = _create_text_splitter(
            "HTMLHeaderTextSplitter",
            headings=list(headings),
        )
        super().__init__(
            splitter=splitter,
            name="langchain.html",
            splitter_name="HTMLHeaderTextSplitter",
            token_counter=token_counter,
        )


class LangChainNLTKChunker(_BaseLangChainChunker):
    def __init__(
        self,
        *,
        language: str = "english",
        token_counter: TokenCounter | None = None,
    ) -> None:
        splitter = _create_text_splitter("NLTKTextSplitter", language=language)
        super().__init__(
            splitter=splitter,
            name="langchain.nltk",
            splitter_name="NLTKTextSplitter",
            token_counter=token_counter,
        )


class LangChainSpacyChunker(_BaseLangChainChunker):
    def __init__(
        self,
        *,
        pipeline: str = "en_core_web_sm",
        token_counter: TokenCounter | None = None,
    ) -> None:
        splitter = _create_text_splitter("SpacyTextSplitter", pipeline=pipeline)
        super().__init__(
            splitter=splitter,
            name="langchain.spacy",
            splitter_name="SpacyTextSplitter",
            token_counter=token_counter,
        )
