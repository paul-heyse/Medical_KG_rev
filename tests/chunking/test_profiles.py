from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.services.chunking.profile_chunkers import (
    CTGovRegistryChunker,
    GuidelineChunker,
    SPLLabelChunker,
)


def build_profile(name: str, chunker_type: str, *, metadata: dict | None = None):
    profile = {
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
    return profile


def test_ctgov_registry_chunker():
    profile = build_profile(
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
    )
    document = Document(
        id="NCT-1",
        source="ctgov",
        sections=[
            Section(
                id="eligibility",
                title="Eligibility Criteria",
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
    chunker = CTGovRegistryChunker(profile=profile)
    chunks = chunker.chunk(document, profile="ctgov-registry")

    assert [chunk.intent_hint for chunk in chunks] == [
        "eligibility",
        "outcome",
        "outcome",
        "ae",
        "results",
        "results",
    ]
    outcome_metadata = [chunk.metadata for chunk in chunks if chunk.intent_hint == "outcome"]
    assert outcome_metadata[0]["ctgov_title"] == "Primary Outcome"
    assert outcome_metadata[0]["ctgov_time_frame"] == "12 weeks"
    assert outcome_metadata[1]["ctgov_measure_type"] == "Secondary"
    ae_chunk = next(chunk for chunk in chunks if chunk.intent_hint == "ae")
    assert ae_chunk.metadata["table_html"].startswith("<table")


def test_spl_label_chunker_formats_loinc_labels():
    profile = build_profile(
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
    )
    document = Document(
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

    chunker = SPLLabelChunker(profile=profile)
    chunks = chunker.chunk(document, profile="spl-label")

    labels = {chunk.section_label for chunk in chunks}
    assert "LOINC:34089-3 Indications" in labels
    assert any(chunk.metadata.get("loinc_code") == "42348-3" for chunk in chunks)


def test_guideline_chunker_creates_recommendation_units():
    profile = build_profile(
        "guideline",
        "guideline_recommendation",
        metadata={
            "intent_hints": {
                "Recommendations": "recommendation",
                "Evidence Summary": "evidence",
            }
        },
    )
    document = Document(
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

    chunker = GuidelineChunker(profile=profile)
    chunks = chunker.chunk(document, profile="guideline")

    assert [chunk.intent_hint for chunk in chunks] == [
        "recommendation",
        "recommendation",
        "evidence",
        "evidence",
    ]
    recommendation_chunks = [chunk for chunk in chunks if chunk.metadata["guideline_unit"] == "recommendation"]
    assert recommendation_chunks[0].metadata["recommendation_id"] == "R1"
    table_chunk = next(chunk for chunk in chunks if chunk.metadata["guideline_unit"] == "evidence" and "table_html" in chunk.metadata)
    assert table_chunk.metadata["table_html"].startswith("<table")
