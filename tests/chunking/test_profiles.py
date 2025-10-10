from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.services.chunking.profile_chunkers import (
    CTGovRegistryChunker,
    GuidelineChunker,
    SPLLabelChunker,
)
from Medical_KG_rev.services.chunking.wrappers import langchain_splitter


def build_profile(name: str, chunker_type: str, *, metadata: dict | None = None):
    return {
        "name": name,
        "domain": name,
        "chunker_type": chunker_type,
        "target_tokens": 256,
        "overlap_tokens": 0,
        "respect_boundaries": ["section"],
        "sentence_splitter": "syntok",
        "preserve_tables_as_html": True,
        "filters": [],
        "metadata": metadata or {},
    }


class _StubTokenizer:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, text: str) -> list[str]:
        self.seen.append(text)
        return text.split()

    @classmethod
    def from_pretrained(cls, model_id: str):
        return cls()


class _StubSplitter:
    def __init__(self, *, chunk_size: int, chunk_overlap: int, length_function):
        self.params = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        self._length_function = length_function

    def split_text(self, text: str) -> list[str]:
        # Split on pipe characters to emulate deterministic chunk boundaries.
        return [part.strip() for part in text.split("|") if part.strip()]


def _stub_langchain(monkeypatch):
    monkeypatch.setattr(
        langchain_splitter,
        "_ensure_langchain_dependencies",
        lambda: (_StubSplitter, _StubTokenizer),
    )


def _ctgov_document() -> Document:
    return Document(
        id="NCT-1",
        source="ctgov",
        sections=[
            Section(
                id="eligibility",
                title="Eligibility Criteria",
                metadata={"intent": "eligibility"},
                blocks=[
                    Block(
                        id="elig-1",
                        type=BlockType.PARAGRAPH,
                        text="Include adults aged 18-65.",
                    )
                ],
            ),
            Section(
                id="outcomes",
                title="Outcome Measures",
                blocks=[
                    Block(
                        id="outcome-primary",
                        type=BlockType.PARAGRAPH,
                        text="Primary outcome measuring blood pressure.",
                        metadata={"title": "Primary Outcome", "time_frame": "12 weeks"},
                    ),
                    Block(
                        id="outcome-secondary",
                        type=BlockType.PARAGRAPH,
                        text="Secondary outcome evaluating quality of life.",
                        metadata={"title": "Secondary Outcome", "measure_type": "Secondary"},
                    ),
                ],
            ),
            Section(
                id="adverse",
                title="Adverse Events",
                blocks=[
                    Block(
                        id="ae-table",
                        type=BlockType.TABLE,
                        text="",
                        metadata={"html": "<table><tr><td>AE</td></tr></table>"},
                    )
                ],
            ),
            Section(
                id="results",
                title="Results",
                metadata={"intent": "results"},
                blocks=[
                    Block(
                        id="result-1",
                        type=BlockType.PARAGRAPH,
                        text="Observed systolic pressure decreased by 10mmHg.",
                    ),
                    Block(
                        id="result-2",
                        type=BlockType.PARAGRAPH,
                        text="Quality of life scores improved by 15%.",
                    ),
                ],
            ),
        ],
    )


def _spl_document() -> Document:
    return Document(
        id="SPL-1",
        source="spl",
        sections=[
            Section(
                id="indications",
                title="Indications",
                blocks=[
                    Block(
                        id="ind-1",
                        type=BlockType.PARAGRAPH,
                        text="This medication treats hypertension.",
                        metadata={"loinc_code": "34089-3"},
                    )
                ],
            ),
            Section(
                id="dosage",
                title="Dosage",
                blocks=[
                    Block(
                        id="dosage-1",
                        type=BlockType.PARAGRAPH,
                        text="Initial dose is 10mg once daily.",
                    )
                ],
            ),
            Section(
                id="warnings",
                title="Warnings",
                blocks=[
                    Block(
                        id="warn-1",
                        type=BlockType.PARAGRAPH,
                        text="Monitor for dizziness in elderly patients.",
                    )
                ],
            ),
            Section(
                id="adverse",
                title="Adverse Reactions",
                blocks=[
                    Block(
                        id="ae-1",
                        type=BlockType.PARAGRAPH,
                        text="Common reactions include headache and nausea.",
                    )
                ],
            ),
        ],
    )


def _guideline_document() -> Document:
    return Document(
        id="GUIDE-1",
        source="guideline",
        sections=[
            Section(
                id="recommendations",
                title="Recommendations",
                blocks=[
                    Block(
                        id="rec-1",
                        type=BlockType.PARAGRAPH,
                        text="Recommend initiating therapy for adults with stage 2 hypertension.",
                        metadata={
                            "recommendation_id": "R1",
                            "strength": "strong",
                            "certainty": "high",
                        },
                    ),
                    Block(
                        id="rec-2",
                        type=BlockType.PARAGRAPH,
                        text="Consider lifestyle modifications for prehypertensive patients.",
                        metadata={
                            "recommendation_id": "R2",
                            "strength": "conditional",
                            "certainty": "moderate",
                        },
                    ),
                ],
            ),
            Section(
                id="evidence",
                title="Evidence Summary",
                blocks=[
                    Block(
                        id="ev-1",
                        type=BlockType.PARAGRAPH,
                        text="Meta-analysis across 12 RCTs shows significant benefit.",
                    ),
                    Block(
                        id="ev-2",
                        type=BlockType.TABLE,
                        text="",
                        metadata={"html": "<table><tr><td>Evidence</td></tr></table>"},
                    ),
                ],
            ),
        ],
    )


def _imrad_document() -> Document:
    return Document(
        id="PMC-1",
        source="pmc",
        sections=[
            Section(
                id="intro",
                title="Introduction",
                metadata={"imrad": "introduction"},
                blocks=[
                    Block(
                        id="intro-1",
                        type=BlockType.PARAGRAPH,
                        text="Background context.|Study rationale.",
                    )
                ],
            ),
            Section(
                id="methods",
                title="Methods",
                metadata={"imrad": "methods"},
                blocks=[
                    Block(
                        id="methods-1",
                        type=BlockType.PARAGRAPH,
                        text="Participants were randomized to two arms.",
                    )
                ],
            ),
            Section(
                id="results",
                title="Results",
                metadata={"imrad": "results"},
                blocks=[
                    Block(
                        id="results-1",
                        type=BlockType.PARAGRAPH,
                        text="Outcome differences reached statistical significance.",
                    )
                ],
            ),
            Section(
                id="discussion",
                title="Discussion",
                metadata={"imrad": "discussion"},
                blocks=[
                    Block(
                        id="discussion-1",
                        type=BlockType.PARAGRAPH,
                        text="Findings align with prior observational evidence.",
                    )
                ],
            ),
        ],
    )


def _assert_ctgov_metadata(chunks):
    outcome_chunks = [chunk for chunk in chunks if chunk.intent_hint == "outcome"]
    assert outcome_chunks[0].metadata["ctgov_title"] == "Primary Outcome"
    assert outcome_chunks[0].metadata["registry_unit"] == "outcome"
    assert outcome_chunks[0].metadata["ctgov_time_frame"] == "12 weeks"
    assert outcome_chunks[1].metadata["ctgov_measure_type"] == "Secondary"
    ae_chunk = next(chunk for chunk in chunks if chunk.intent_hint == "ae")
    assert ae_chunk.metadata["registry_unit"] == "ae"
    assert ae_chunk.metadata["table_html"].startswith("<table")


def _assert_spl_metadata(chunks):
    labels = [chunk.section_label for chunk in chunks]
    assert labels == [
        "LOINC:34089-3 Indications",
        "LOINC:42348-3 Dosage",
        "LOINC:39245-5 Warnings",
        "LOINC:43995-0 Adverse Reactions",
    ]
    assert all(chunk.metadata["chunker_version"] == "SPLLabelChunker" for chunk in chunks)
    assert any(chunk.metadata.get("loinc_code") == "42348-3" for chunk in chunks)


def _assert_guideline_metadata(chunks):
    recommendations = [
        chunk for chunk in chunks if chunk.metadata["guideline_unit"] == "recommendation"
    ]
    assert recommendations[0].metadata["recommendation_id"] == "R1"
    assert recommendations[0].metadata["strength"] == "strong"
    evidence_chunks = [chunk for chunk in chunks if chunk.metadata["guideline_unit"] == "evidence"]
    assert any("table_html" in chunk.metadata for chunk in evidence_chunks)


def _assert_imrad_metadata(chunks):
    intro_chunks = [chunk for chunk in chunks if chunk.section_label == "Introduction"]
    assert len(intro_chunks) == 2
    assert all(chunk.intent_hint == "background" for chunk in intro_chunks)
    assert intro_chunks[0].metadata["section_metadata"]["imrad"] == "introduction"
    discussion = chunks[-1]
    assert discussion.section_label == "Discussion"
    assert discussion.metadata["section_metadata"]["imrad"] == "discussion"


@dataclass(slots=True)
class ProfileScenario:
    id: str
    chunker_cls: type
    profile: dict[str, object]
    document_factory: Callable[[], Document]
    expected_sections: Sequence[str]
    expected_intents: Sequence[str]
    metadata_assertions: Callable[[Sequence], None]
    requires_langchain: bool = False


SCENARIOS = [
    ProfileScenario(
        id="ctgov-registry",
        chunker_cls=CTGovRegistryChunker,
        profile=build_profile(
            "ctgov-registry",
            "ctgov_registry",
            metadata={
                "intent_hints": {
                    "Eligibility Criteria": "eligibility",
                    "Outcome Measures": "outcome",
                    "Adverse Events": "ae",
                    "Results": "results",
                }
            },
        ),
        document_factory=_ctgov_document,
        expected_sections=[
            "Eligibility Criteria",
            "Outcome Measures",
            "Outcome Measures",
            "Adverse Events",
            "Results",
            "Results",
        ],
        expected_intents=["eligibility", "outcome", "outcome", "ae", "results", "results"],
        metadata_assertions=_assert_ctgov_metadata,
    ),
    ProfileScenario(
        id="spl-label",
        chunker_cls=SPLLabelChunker,
        profile=build_profile(
            "spl-label",
            "spl_label",
            metadata={
                "intent_hints": {
                    "Indications": "narrative",
                    "Dosage": "dose",
                    "Warnings": "safety",
                    "Adverse Reactions": "ae",
                },
                "loinc_map": {
                    "Indications": "34089-3",
                    "Dosage": "42348-3",
                    "Warnings": "39245-5",
                    "Adverse Reactions": "43995-0",
                },
            },
        ),
        document_factory=_spl_document,
        expected_sections=[
            "LOINC:34089-3 Indications",
            "LOINC:42348-3 Dosage",
            "LOINC:39245-5 Warnings",
            "LOINC:43995-0 Adverse Reactions",
        ],
        expected_intents=["narrative", "dose", "safety", "ae"],
        metadata_assertions=_assert_spl_metadata,
    ),
    ProfileScenario(
        id="guideline",
        chunker_cls=GuidelineChunker,
        profile=build_profile(
            "guideline",
            "guideline_recommendation",
            metadata={
                "intent_hints": {
                    "Recommendations": "recommendation",
                    "Evidence Summary": "evidence",
                }
            },
        ),
        document_factory=_guideline_document,
        expected_sections=[
            "Recommendations",
            "Recommendations",
            "Evidence Summary",
            "Evidence Summary",
        ],
        expected_intents=["recommendation", "recommendation", "evidence", "evidence"],
        metadata_assertions=_assert_guideline_metadata,
    ),
    ProfileScenario(
        id="pmc-imrad",
        chunker_cls=langchain_splitter.LangChainChunker,
        profile=build_profile(
            "pmc-imrad",
            "langchain_recursive",
            metadata={
                "intent_hints": {
                    "Introduction": "background",
                    "Methods": "methods",
                    "Results": "results",
                    "Discussion": "discussion",
                },
                "chunker_version": "langchain-test",
                "tokenizer_model": "stub-model",
            },
        ),
        document_factory=_imrad_document,
        expected_sections=[
            "Introduction",
            "Introduction",
            "Methods",
            "Results",
            "Discussion",
        ],
        expected_intents=["background", "background", "methods", "results", "discussion"],
        metadata_assertions=_assert_imrad_metadata,
        requires_langchain=True,
    ),
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda case: case.id)
def test_profile_chunkers_preserve_annotations(scenario: ProfileScenario, monkeypatch):
    if scenario.requires_langchain:
        _stub_langchain(monkeypatch)
    chunker = scenario.chunker_cls(profile=scenario.profile)
    document = scenario.document_factory()
    chunks = chunker.chunk(document, profile=scenario.profile["name"])

    assert [chunk.section_label for chunk in chunks] == list(scenario.expected_sections)
    assert [chunk.intent_hint for chunk in chunks] == list(scenario.expected_intents)
    assert all(
        chunk.metadata.get("chunking_profile") == scenario.profile["name"] for chunk in chunks
    )
    scenario.metadata_assertions(chunks)
