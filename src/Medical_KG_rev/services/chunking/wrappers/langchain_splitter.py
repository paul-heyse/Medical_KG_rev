"""Wrapper around langchain text splitters."""

from __future__ import annotations

from typing import Any

from Medical_KG_rev.models.ir import Document

from ..port import Chunk
from .base import BaseProfileChunker


def _ensure_langchain_dependencies() -> tuple[Any, Any]:  # pragma: no cover - import side effects tested separately
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "langchain-text-splitters is required for LangChainChunker"
        ) from exc
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for LangChainChunker") from exc
    return RecursiveCharacterTextSplitter, AutoTokenizer


class LangChainChunker(BaseProfileChunker):
    """Recursive chunker backed by LangChain text splitters."""

    name = "langchain_recursive"

    def __init__(self, *, profile: dict[str, Any]) -> None:
        super().__init__(profile=profile)
        splitter_cls, tokenizer_cls = _ensure_langchain_dependencies()
        model_id = profile.get("metadata", {}).get(
            "tokenizer_model", "Qwen/Qwen2.5-Coder-1.5B"
        )
        self._tokenizer = tokenizer_cls.from_pretrained(model_id)
        self._splitter = splitter_cls(
            chunk_size=profile.get("target_tokens", 512) * 4,
            chunk_overlap=profile.get("overlap_tokens", 50) * 4,
            length_function=self._count_tokens,
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def chunk(self, document: Document, *, profile: str) -> list[Chunk]:
        filtered_document, groups = self._prepare_groups(document)
        chunk_texts: list[str] = []
        chunk_to_group: list[int] = []
        for index, group in enumerate(groups):
            combined = "\n\n".join(ctx.text for ctx in group if ctx.text)
            if not combined:
                continue
            splits = self._splitter.split_text(combined)
            chunk_texts.extend(splits)
            chunk_to_group.extend([index] * len(splits))
        return self._assemble(
            document=filtered_document,
            groups=groups,
            chunk_texts=chunk_texts,
            chunk_to_group_index=chunk_to_group,
        )


def register() -> None:
    from ..port import register_chunker

    register_chunker(
        LangChainChunker.name,
        lambda *, profile: LangChainChunker(profile=profile),
    )
