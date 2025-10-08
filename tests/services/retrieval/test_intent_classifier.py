from Medical_KG_rev.services.retrieval.routing import IntentClassifier, QueryIntent


def test_intent_classifier_detects_tabular_keywords() -> None:
    classifier = IntentClassifier()
    result = classifier.classify("What are the adverse events results table for pembrolizumab?")
    assert result.intent is QueryIntent.TABULAR
    assert result.confidence >= 0.8
    assert any("adverse events" in pattern for pattern in result.matched_patterns)


def test_intent_classifier_manual_override() -> None:
    classifier = IntentClassifier()
    result = classifier.classify("Eligibility criteria", override=QueryIntent.NARRATIVE)
    assert result.intent is QueryIntent.NARRATIVE
    assert result.override is QueryIntent.NARRATIVE
    assert result.confidence == 1.0


def test_intent_classifier_benchmark_accuracy() -> None:
    classifier = IntentClassifier()
    dataset = {
        "pembrolizumab adverse events table": QueryIntent.TABULAR,
        "describe the mechanism of action": QueryIntent.NARRATIVE,
    }
    accuracy = classifier.benchmark(dataset)
    assert accuracy == 1.0
