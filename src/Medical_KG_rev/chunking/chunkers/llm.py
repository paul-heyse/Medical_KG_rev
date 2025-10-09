"""LLM assisted chunker implementations."""

from __future__ import annotations

"""LLM assisted chunker implementations."""

from dataclasses import dataclass, field
import json
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..coherence import SemanticDriftDetector
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import BlockContext, ProvenanceNormalizer, make_chunk_id
from ..tokenization import TokenCounter, default_token_counter
from ..exceptions import ChunkerConfigurationError
from .semantic import SemanticSplitterChunker


class SupportsLLMGeneration(Protocol):
    """Protocol implemented by lightweight LLM client wrappers."""

    def generate(self, *, prompt: str, text: str) -> dict[str, object]: ...


@dataclass(slots=True)
class _TemplateLLM:
    """Deterministic template backed LLM used for tests and offline evaluation."""

    seed: int = 0
    _cache: dict[str, dict[str, object]] = field(init=False, repr=False, default_factory=dict)

    def generate(self, *, prompt: str, text: str) -> dict[str, object]:
        cache_key = json.dumps({"prompt": prompt, "text": text})
        if cache_key in self._cache:
            return self._cache[cache_key]
        lowered = text.lower()
        boundary_terms = [
            ("introduction", 0.0),
            ("background", 0.0),
            ("methods", 0.0),
            ("results", 0.0),
            ("discussion", 0.0),
            ("conclusion", 0.0),
        ]
        boundaries: list[dict[str, object]] = []
        for term, _weight in boundary_terms:
            index = lowered.find(term)
            if index == -1:
                continue
            boundaries.append({"offset": index, "label": term})
        payload: dict[str, object] = {"boundaries": boundaries}
        self._cache[cache_key] = payload
        return payload


class _HashingEncoder:
    """Simple encoder that approximates embeddings via hashing.

    The semantic splitter fallback expects a ``encode`` method that returns a
    numpy array.  By hashing n-grams we can obtain deterministic vectors without
    depending on heavyweight models in unit tests.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def encode(self, sentences: Sequence[str], convert_to_numpy: bool = True) -> np.ndarray:
        vectors = []
        for sentence in sentences:
            vec = np.zeros(self.dimensions, dtype=float)
            tokens = sentence.split()
            if not tokens:
                vectors.append(vec)
                continue
            for token in tokens:
                bucket = hash(token) % self.dimensions
                vec[bucket] += 1.0
            norm = np.linalg.norm(vec)
            if norm:
                vec /= norm
            vectors.append(vec)
        return np.vstack(vectors) if convert_to_numpy else vectors


class LLMChapteringChunker(BaseChunker):
    """Chunker that leverages LLM prompted boundaries with semantic validation."""

    name = "llm_chaptering"
    version = "v1"

    def __init__(
        self,
        *,
        prompt_version: str = "v1",
        llm_client: SupportsLLMGeneration | None = None,
        coherence_threshold: float = 0.78,
        min_tokens: int = 160,
        token_counter: TokenCounter | None = None,
        fallback_chunker: SemanticSplitterChunker | None = None,
    ) -> None:
        self.prompt_version = prompt_version
        self.llm = llm_client or _TemplateLLM()
        self.coherence_threshold = coherence_threshold
        self.min_tokens = min_tokens
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)
        self._fallback = fallback_chunker or SemanticSplitterChunker(
            encoder=_HashingEncoder(), token_counter=self.counter
        )
        self._boundary_cache: dict[str, list[int]] = {}

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        contexts = [
            ctx
            for ctx in self.normalizer.iter_block_contexts(document)
            if ctx.text and not ctx.is_table
        ]
        if not contexts:
            return []
        boundaries = self._fetch_boundaries(document, contexts)
        validated = self._validate_boundaries(contexts, boundaries)
        if len(validated) <= 1:
            return self._fallback.chunk(
                document, tenant_id=tenant_id, granularity=granularity or "section"
            )
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "section",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        for start, end in zip(validated[:-1], validated[1:], strict=False):
            window = contexts[start:end]
            if not window:
                continue
            metadata = {"segment_type": "llm", "prompt_version": self.prompt_version}
            chunk = assembler.build(window, metadata=metadata)
            if not chunk.chunk_id.startswith(f"{document.id}:"):
                chunk = chunk.model_copy(
                    update={
                        "chunk_id": make_chunk_id(
                            document.id, self.name, chunk.granularity, len(chunks)
                        )
                    }
                )
            chunks.append(chunk)
        return chunks

    def explain(self) -> dict[str, object]:
        return {
            "prompt_version": self.prompt_version,
            "coherence_threshold": self.coherence_threshold,
            "min_tokens": self.min_tokens,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch_boundaries(self, document: Document, contexts: Sequence[BlockContext]) -> list[int]:
        cache_key = f"{document.id}:{self.prompt_version}"
        if cache_key in self._boundary_cache:
            return list(self._boundary_cache[cache_key])
        prompt = self._prompt_template(document)
        combined = "\n\n".join(ctx.text for ctx in contexts)
        response = self.llm.generate(prompt=prompt, text=combined) or {}
        raw_boundaries = response.get("boundaries", [])
        if not isinstance(raw_boundaries, Sequence):
            raise ChunkerConfigurationError("LLM boundary payload must be a sequence")
        offsets: list[int] = []
        for candidate in raw_boundaries:
            if isinstance(candidate, Mapping):
                value = candidate.get("offset")
                if isinstance(value, (int, float)):
                    offsets.append(int(value))
                continue
            if isinstance(candidate, (int, float)):
                offsets.append(int(candidate))
        mapped = self._map_offsets_to_contexts(contexts, offsets)
        self._boundary_cache[cache_key] = mapped
        return list(mapped)

    def _map_offsets_to_contexts(
        self, contexts: Sequence[BlockContext], offsets: Sequence[int]
    ) -> list[int]:
        if not offsets:
            return [0, len(contexts)]
        candidates: list[int] = [0]
        for offset in sorted(set(offsets)):
            for index, ctx in enumerate(contexts):
                if ctx.start_char <= offset < ctx.end_char:
                    if index not in candidates:
                        candidates.append(index)
                    break
        if candidates[-1] != len(contexts):
            candidates.append(len(contexts))
        return sorted(set(candidates))

    def _validate_boundaries(
        self, contexts: Sequence[BlockContext], boundaries: Sequence[int]
    ) -> list[int]:
        if len(boundaries) <= 1:
            return [0, len(contexts)]
        embeddings = _HashingEncoder().encode([ctx.text for ctx in contexts])
        similarities: list[float] = []
        for idx in range(1, len(contexts)):
            left = embeddings[idx - 1]
            right = embeddings[idx]
            denom = np.linalg.norm(left) * np.linalg.norm(right)
            if denom == 0:
                similarities.append(1.0)
            else:
                cos_sim = float(np.dot(left, right) / denom)
                similarities.append(max(min(cos_sim, 1.0), -1.0))
        validated = [0]
        token_budget = contexts[0].token_count if contexts else 0
        for boundary in boundaries[1:]:
            if boundary >= len(contexts):
                break
            token_budget += contexts[boundary].token_count
            sim = similarities[boundary - 1] if boundary - 1 < len(similarities) else 1.0
            if token_budget >= self.min_tokens and sim <= self.coherence_threshold:
                validated.append(boundary)
                token_budget = 0
        drift = SemanticDriftDetector(
            threshold=self.coherence_threshold,
            min_tokens=self.min_tokens,
            token_counter=self.counter,
        ).detect(contexts, similarities)
        candidates = sorted(set(validated) | set(drift))
        if candidates[0] != 0:
            candidates.insert(0, 0)
        if candidates[-1] != len(contexts):
            candidates.append(len(contexts))
        if len(candidates) <= 1:
            return [0, len(contexts)]
        return candidates

    def _prompt_template(self, document: Document) -> str:
        examples = (
            "Document: Clinical Trial Report\n"
            "Sections: Introduction -> Methods -> Results -> Discussion\n"
            "Return JSON with `boundaries` measured as character offsets."
        )
        if self.prompt_version == "v2":
            examples = (
                "Document: Drug Label\n"
                "Sections: Highlights -> Indications -> Dosage -> Warnings\n"
                "Return boundary offsets keyed by section labels."
            )
        return (
            "You are an expert biomedical editor. "
            "Identify major section boundaries in the provided document. "
            f"Document title: {document.title or 'Untitled'}.\n\n"
            f"Examples:\n{examples}\n\nRespond with a JSON object"
        )


__all__ = ["LLMChapteringChunker"]
