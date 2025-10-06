"""Semantic chunking strategies for documents."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, object]


@dataclass(slots=True)
class ChunkingOptions:
    strategy: str = "section"
    max_tokens: int = 512
    overlap: float = 0.1


class ChunkingService:
    SECTION_PATTERN = re.compile(r"^(#+\s+|[A-Z][A-Za-z\s]+:)")

    def chunk(self, document_id: str, text: str, options: ChunkingOptions | None = None) -> List[Chunk]:
        opts = options or ChunkingOptions()
        strategy = opts.strategy.lower()
        if strategy == "paragraph":
            parts = self._paragraph_chunks(text)
        elif strategy == "section":
            parts = self._section_chunks(text)
        elif strategy == "table":
            parts = self._table_chunks(text)
        elif strategy == "sliding-window":
            parts = self._sliding_window_chunks(text, opts.max_tokens, opts.overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        return self._limit_tokens(document_id, parts, opts.max_tokens)

    def _paragraph_chunks(self, text: str) -> List[str]:
        paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
        return paragraphs or [text.strip()]

    def _section_chunks(self, text: str) -> List[str]:
        sections: List[str] = []
        current: List[str] = []
        for line in text.splitlines():
            if self.SECTION_PATTERN.match(line) and current:
                sections.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append("\n".join(current).strip())
        return [section for section in sections if section]

    def _table_chunks(self, text: str) -> List[str]:
        chunks: List[str] = []
        current: List[str] = []
        in_table = False
        for line in text.splitlines():
            if "|" in line or "\t" in line:
                in_table = True
                current.append(line)
            else:
                if in_table:
                    chunks.append("\n".join(current).strip())
                    current = []
                    in_table = False
                if line.strip():
                    chunks.append(line.strip())
        if current:
            chunks.append("\n".join(current).strip())
        return chunks or [text.strip()]

    def _sliding_window_chunks(self, text: str, max_tokens: int, overlap: float) -> List[str]:
        words = text.split()
        if not words:
            return []
        window_size = max(1, max_tokens)
        step = max(1, int(window_size * (1 - overlap)))
        chunks = []
        for start in range(0, len(words), step):
            segment = words[start : start + window_size]
            if not segment:
                break
            chunks.append(" ".join(segment))
            if start + window_size >= len(words):
                break
        return chunks

    def _limit_tokens(self, document_id: str, parts: Sequence[str], max_tokens: int) -> List[Chunk]:
        chunks: List[Chunk] = []
        for part in parts:
            tokens = part.split()
            if not tokens:
                continue
            for start in range(0, len(tokens), max_tokens):
                window = tokens[start : start + max_tokens]
                chunk_text = " ".join(window)
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{start}:{chunk_text}"))
                metadata = {
                    "document_id": document_id,
                    "start_token": start,
                    "end_token": start + len(window),
                }
                chunks.append(Chunk(id=chunk_id, text=chunk_text, metadata=metadata))
        return chunks

    def chunk_sections(self, document_id: str, text: str) -> List[Chunk]:
        return self.chunk(document_id, text, ChunkingOptions(strategy="section"))

    def chunk_paragraphs(self, document_id: str, text: str) -> List[Chunk]:
        return self.chunk(document_id, text, ChunkingOptions(strategy="paragraph"))

    def chunk_tables(self, document_id: str, text: str) -> List[Chunk]:
        return self.chunk(document_id, text, ChunkingOptions(strategy="table"))

    def sliding_window(self, document_id: str, text: str, max_tokens: int, overlap: float) -> List[Chunk]:
        return self.chunk(
            document_id,
            text,
            ChunkingOptions(strategy="sliding-window", max_tokens=max_tokens, overlap=overlap),
        )
