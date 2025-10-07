from __future__ import annotations

from typing import Iterable

from ..base import ContextualChunker
from ..data import load_json_resource
from ..provenance import BlockContext
from ..segmentation import Segment
from ..tokenization import TokenCounter


_TAXONOMY = load_json_resource("clinical_sections.json")


def _build_role_keywords() -> dict[str, set[str]]:
    keywords: dict[str, set[str]] = {
        "pico_population": {"population", "patients", "subjects", "participants"},
        "pico_intervention": {"intervention", "treatment", "drug", "dose", "therapy"},
        "pico_outcome": {"outcome", "efficacy", "response", "result", "effect"},
        "eligibility": {"eligibility", "inclusion", "exclusion"},
        "adverse_event": {"adverse", "safety", "serious", "ae", "toxicity"},
        "dose_regimen": {"dosage", "dose", "regimen", "administration", "schedule"},
        "endpoint": {"endpoint", "primary", "secondary", "objective"},
        "effect_magnitude": {"effect", "improvement", "reduction", "increase"},
    }
    for family in _TAXONOMY.values():
        for role_name, labels in family.items():
            normalized = role_name.replace(" ", "_")
            keywords.setdefault(normalized, set()).update(
                {label.lower() for label in labels}
            )
    return keywords


ROLE_KEYWORDS = _build_role_keywords()
PAIRING_ROLES = {"endpoint": {"pico_outcome", "effect_magnitude"}}


class ClinicalRoleChunker(ContextualChunker):
    name = "clinical_role"
    version = "v1"
    segment_type = "clinical"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        min_tokens: int = 120,
    ) -> None:
        super().__init__(token_counter=token_counter)
        self.min_tokens = min_tokens

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        buffer: list[BlockContext] = []
        current_role = "general"
        token_total = 0
        pending_pair = False
        segments: list[Segment] = []
        for ctx in contexts:
            role = self._detect_role(ctx)
            if pending_pair and role in PAIRING_ROLES.get(current_role, set()):
                current_role = role
                pending_pair = False
            elif role != current_role and buffer:
                segments.append(
                    Segment(
                        contexts=list(buffer),
                        metadata={"facet_type": current_role},
                    )
                )
                buffer = []
                token_total = 0
            buffer.append(ctx)
            if role == "endpoint":
                pending_pair = True
            current_role = role
            token_total += ctx.token_count
            if token_total >= self.min_tokens:
                segments.append(
                    Segment(
                        contexts=list(buffer),
                        metadata={"facet_type": current_role},
                    )
                )
                buffer = []
                token_total = 0
        if buffer:
            segments.append(
                Segment(contexts=list(buffer), metadata={"facet_type": current_role})
            )
        return segments

    def explain(self) -> dict[str, object]:
        return {
            "min_tokens": self.min_tokens,
            "roles": sorted(ROLE_KEYWORDS),
            "pairing_roles": {key: sorted(value) for key, value in PAIRING_ROLES.items()},
        }

    def _detect_role(self, context: BlockContext) -> str:
        text = context.text.lower()
        for role, keywords in ROLE_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return role
        return "general"
