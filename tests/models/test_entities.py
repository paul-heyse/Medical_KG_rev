import pytest

from Medical_KG_rev.models.entities import Claim, Entity, Evidence
from Medical_KG_rev.models.ir import Span
from Medical_KG_rev.models.provenance import DataSource, ExtractionActivity


def _activity() -> ExtractionActivity:
    return ExtractionActivity(
        id="act-1",
        actor="tester",
        data_source=DataSource(id="clinicaltrials", name="ClinicalTrials"),
    )


def test_entity_alias_deduplication():
    entity = Entity(
        id="ent-1",
        type="Condition",
        canonical_name="Diabetes",
        aliases=["diabetes", "Diabetes"],
    )
    assert len(entity.aliases) == 1


def test_claim_roundtrip():
    claim = Claim(id="c1", subject_id="s", predicate="treats", object_id="o", extraction=_activity())
    assert claim.predicate == "treats"


def test_evidence_confidence_bounds():
    evidence = Evidence(
        id="e1",
        document_id="doc1",
        span=Span(start=0, end=4, text="test"),
        extraction=_activity(),
    )
    assert 0.0 <= evidence.confidence <= 1.0

    with pytest.raises(ValueError):
        Evidence(
            id="e2",
            document_id="doc1",
            span=Span(start=0, end=2),
            extraction=_activity(),
            confidence=2.0,
        )


def test_extraction_activity_requires_timezone():
    with pytest.raises(ValueError):
        ExtractionActivity(
            id="act-2",
            actor="tester",
            data_source=DataSource(id="clinicaltrials", name="ClinicalTrials"),
            performed_at=_activity().performed_at.replace(tzinfo=None),
        )
