"""Clinical role aware chunker using lightweight keyword heuristics."""

from __future__ import annotations

from typing import Iterable

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..models import Chunk, Granularity
from ..provenance import BlockContext, ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from ..ports import BaseChunker


ROLE_KEYWORDS = {
    "pico_population": {"population", "patients", "subjects"},
    "pico_intervention": {"intervention", "treatment", "drug", "dose"},
    "pico_outcome": {"outcome", "efficacy", "response", "result"},
    "eligibility": {"eligibility", "inclusion", "exclusion"},
    "adverse_event": {"adverse", "safety", "serious", "ae"},
    "dose_regimen": {"dosage", "dose", "regimen", "administration"},
    "endpoint": {"endpoint", "primary", "secondary"},
}


class ClinicalRoleChunker(BaseChunker):
    name = "clinical_role"
    version = "v1"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        min_tokens: int = 120,
    ) -> None:
        self.counter = token_counter or default_token_counter()
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
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        buffer: list[BlockContext] = []
        current_role = "general"
        token_total = 0
        for ctx in contexts:
            role = self._detect_role(ctx)
            if role != current_role and buffer:
                chunks.append(
                    assembler.build(
                        buffer,
                        metadata={"segment_type": "clinical", "facet_type": current_role},
                    )
                )
                buffer = []
                token_total = 0
            buffer.append(ctx)
            current_role = role
            token_total += ctx.token_count
            if token_total >= self.min_tokens:
                chunks.append(
                    assembler.build(
                        buffer,
                        metadata={"segment_type": "clinical", "facet_type": current_role},
                    )
                )
                buffer = []
                token_total = 0
        if buffer:
            chunks.append(
                assembler.build(
                    buffer,
                    metadata={"segment_type": "clinical", "facet_type": current_role},
                )
            )
        return chunks

    def explain(self) -> dict[str, object]:
        return {"min_tokens": self.min_tokens, "roles": sorted(ROLE_KEYWORDS)}

    def _detect_role(self, context: BlockContext) -> str:
        text = context.text.lower()
        for role, keywords in ROLE_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return role
        return "general"
