"""Semantic splitter chunker based on embedding coherence."""

from __future__ import annotations

from math import inf
from typing import Iterable

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..provenance import BlockContext, ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from ..ports import BaseChunker


class SemanticSplitterChunker(BaseChunker):
    name = "semantic_splitter"
    version = "v1"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        tau_coh: float = 0.82,
        min_tokens: int = 200,
        gpu_semantic_checks: bool = False,
        encoder: object | None = None,
    ) -> None:
        if encoder is None:
            if SentenceTransformer is None:
                raise ChunkerConfigurationError(
                    "sentence-transformers must be installed for SemanticSplitterChunker"
                )
            encoder = SentenceTransformer(model_name)
            if gpu_semantic_checks:
                if torch is None or not torch.cuda.is_available():
                    raise RuntimeError("GPU semantic checks requested but CUDA is not available")
                encoder = encoder.to("cuda")
        self.counter = token_counter or default_token_counter()
        self.model = encoder
        self.tau_coh = tau_coh
        self.min_tokens = min_tokens
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

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
        embeddings = self._encode(contexts)
        boundaries = self._find_boundaries(contexts, embeddings)
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        start = 0
        for boundary in boundaries:
            window = contexts[start:boundary]
            if window:
                chunks.append(
                    assembler.build(window, metadata={"segment_type": "semantic"})
                )
            start = boundary
        tail = contexts[start:]
        if tail:
            chunks.append(assembler.build(tail, metadata={"segment_type": "semantic"}))
        return chunks

    def explain(self) -> dict[str, object]:
        return {"tau_coh": self.tau_coh, "min_tokens": self.min_tokens}

    def _encode(self, contexts: list[BlockContext]) -> np.ndarray:
        sentences = [ctx.text for ctx in contexts]
        if not sentences:
            return np.empty((0, 1))
        encode = getattr(self.model, "encode", None)
        if encode is None:
            raise ChunkerConfigurationError("Encoder does not expose an encode() method")
        result = encode(sentences, convert_to_numpy=True)  # type: ignore[arg-type]
        return np.asarray(result)

    def _find_boundaries(self, contexts: list[BlockContext], embeddings: np.ndarray) -> list[int]:
        if embeddings.size == 0:
            return [len(contexts)]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.clip(norms, a_min=1e-9, a_max=inf)
        sims = np.sum(normalized[1:] * normalized[:-1], axis=1)
        boundaries = []
        token_budget = 0
        for idx, (ctx, sim) in enumerate(zip(contexts[1:], sims, strict=False), start=1):
            token_budget += ctx.token_count
            if token_budget >= self.min_tokens and sim < self.tau_coh:
                boundaries.append(idx)
                token_budget = 0
        boundaries.append(len(contexts))
        return boundaries
