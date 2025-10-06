"""Semantic chunking strategies for documents."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import tiktoken


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, object]
    token_count: int


@dataclass(slots=True)
class ChunkingOptions:
    strategy: str = "section"
    max_tokens: int = 512
    overlap: float = 0.1


class ChunkingService:
    SECTION_PATTERN = re.compile(r"^(#+\s+|[A-Z][A-Za-z\s]+:)")

    def __init__(self, *, encoding: str = "cl100k_base") -> None:
        try:
            self._encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except KeyError:
            self._encoder = tiktoken.get_encoding(encoding)

    def chunk(self, document_id: str, text: str, options: ChunkingOptions | None = None) -> List[Chunk]:
        opts = options or ChunkingOptions()
        strategy = opts.strategy.lower()
        if strategy in {"semantic", "section"}:
            segments = self._section_chunks(text)
        elif strategy == "paragraph":
            segments = self._paragraph_chunks(text)
        elif strategy == "table":
            segments = self._table_chunks(text)
        elif strategy == "sliding-window":
            segments = self._sliding_window_chunks(text, opts.max_tokens, opts.overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        return self._materialize_chunks(document_id, segments, opts)

    def _paragraph_chunks(self, text: str) -> List[Tuple[str, dict[str, object]]]:
        paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]
        return [(paragraph, {"segment_type": "paragraph"}) for paragraph in paragraphs]

    def _section_chunks(self, text: str) -> List[Tuple[str, dict[str, object]]]:
        sections: List[Tuple[str, dict[str, object]]] = []
        current_lines: List[str] = []
        current_title = ""
        for line in text.splitlines():
            if self.SECTION_PATTERN.match(line):
                if current_lines:
                    sections.append(("\n".join(current_lines).strip(), {"segment_type": "section", "section_title": current_title}))
                current_title = line.strip().strip(":")
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            sections.append(("\n".join(current_lines).strip(), {"segment_type": "section", "section_title": current_title}))
        return sections or [(text.strip(), {"segment_type": "section", "section_title": ""})]

    def _table_chunks(self, text: str) -> List[Tuple[str, dict[str, object]]]:
        chunks: List[Tuple[str, dict[str, object]]] = []
        current: List[str] = []
        in_table = False
        for line in text.splitlines():
            if "|" in line or "\t" in line:
                in_table = True
                current.append(line)
            else:
                if in_table:
                    chunks.append(("\n".join(current).strip(), {"segment_type": "table"}))
                    current = []
                    in_table = False
                if line.strip():
                    chunks.append((line.strip(), {"segment_type": "text"}))
        if current:
            chunks.append(("\n".join(current).strip(), {"segment_type": "table"}))
        return chunks or [(text.strip(), {"segment_type": "text"})]

    def _sliding_window_chunks(
        self, text: str, max_tokens: int, overlap: float
    ) -> List[Tuple[str, dict[str, object]]]:
        tokens = self._encoder.encode(text)
        if not tokens:
            return []
        step = max(1, int(max_tokens * (1 - overlap)))
        windows: List[Tuple[str, dict[str, object]]] = []
        for index in range(0, len(tokens), step):
            window = tokens[index : index + max_tokens]
            if not window:
                break
            chunk_text = self._encoder.decode(window)
            windows.append((chunk_text, {"segment_type": "window", "token_start": index}))
            if index + max_tokens >= len(tokens):
                break
        return windows

    def _materialize_chunks(
        self,
        document_id: str,
        segments: Sequence[Tuple[str, dict[str, object]]],
        options: ChunkingOptions,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        for segment_index, (segment_text, metadata) in enumerate(segments):
            tokens = self._encoder.encode(segment_text)
            if not tokens:
                continue
            windows = list(self._token_windows(tokens, options.max_tokens, options.overlap))
            for window_index, token_window in enumerate(windows):
                chunk_text = self._encoder.decode(token_window)
                token_count = len(token_window)
                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{document_id}:{segment_index}:{window_index}:{hash(chunk_text)}",
                    )
                )
                chunk_metadata = {
                    "document_id": document_id,
                    "strategy": options.strategy,
                    "chunk_index": len(chunks),
                    "segment_index": segment_index,
                    "window_index": window_index,
                    "token_count": token_count,
                }
                chunk_metadata.update(metadata)
                chunks.append(Chunk(id=chunk_id, text=chunk_text, metadata=chunk_metadata, token_count=token_count))
        return chunks

    def _token_windows(
        self, tokens: Sequence[int], max_tokens: int, overlap: float
    ) -> Iterable[Sequence[int]]:
        if len(tokens) <= max_tokens:
            yield tokens
            return
        step = max(1, int(max_tokens * (1 - overlap)))
        for start in range(0, len(tokens), step):
            window = tokens[start : start + max_tokens]
            if not window:
                break
            yield window
            if start + max_tokens >= len(tokens):
                break

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
