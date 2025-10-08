from Medical_KG_rev.services.retrieval.boosting import (
    ClinicalIntent,
    ClinicalIntentAnalyzer,
    infer_document_intents,
)


def test_analyzer_detects_adverse_event_keywords() -> None:
    analyzer = ClinicalIntentAnalyzer()
    analysis = analyzer.analyse("What adverse events were observed in the trial?")
    assert analysis.primary is not None
    assert analysis.primary.intent is ClinicalIntent.ADVERSE_EVENTS
    assert analysis.primary.confidence > 0


def test_analyzer_supports_multi_intents() -> None:
    analyzer = ClinicalIntentAnalyzer()
    analysis = analyzer.analyse("Eligibility dose titration schedule")
    intents = {score.intent for score in analysis.intents}
    assert ClinicalIntent.ELIGIBILITY in intents
    assert ClinicalIntent.DOSAGE in intents
    assert sum(score.confidence for score in analysis.intents) <= 1.0


def test_analyzer_manual_override() -> None:
    analyzer = ClinicalIntentAnalyzer()
    override = analyzer.resolve_overrides(["results", ClinicalIntent.DOSAGE])
    analysis = analyzer.analyse("Anything", override=override)
    assert {score.intent for score in analysis.intents} == {
        ClinicalIntent.RESULTS,
        ClinicalIntent.DOSAGE,
    }
    assert all(score.confidence == 1.0 for score in analysis.intents)
    assert tuple(analysis.override) == override


def test_infer_document_intents_matches_metadata() -> None:
    metadata = {
        "section_label": "Eligibility Criteria",
        "intent_hint": "eligibility",
        "metadata": {"section": "Study Design"},
    }
    inferred = infer_document_intents(metadata)
    assert ClinicalIntent.ELIGIBILITY in inferred
    assert ClinicalIntent.METHODS in inferred

