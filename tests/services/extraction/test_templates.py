import pytest

from Medical_KG_rev.services.extraction.templates import TemplateValidationError, validate_template


def test_validate_pico_template_success():
    text = (
        "Population: Adults with hypertension. Intervention: ACE inhibitor. "
        "Comparison: Placebo. Outcome: Reduced systolic blood pressure."
    )
    payload = {
        "population": {
            "description": "Adults with hypertension",
            "span": {"text": "Adults with hypertension", "start": 12, "end": 35},
        },
        "interventions": [
            {
                "name": "ACE inhibitor",
                "span": {"text": "ACE inhibitor", "start": 49, "end": 62},
            }
        ],
        "comparison": {
            "description": "Placebo",
            "span": {"text": "Placebo", "start": 76, "end": 83},
        },
        "outcomes": [
            {
                "name": "Reduced systolic blood pressure",
                "span": {
                    "text": "Reduced systolic blood pressure",
                    "start": 93,
                    "end": 122,
                },
            }
        ],
    }
    result = validate_template("pico", payload, text)
    assert result["population"]["description"] == "Adults with hypertension"


def test_invalid_span_raises():
    text = "Example text"
    payload = {
        "population": {
            "description": "Example",
            "span": {"text": "Mismatch", "start": 0, "end": 8},
        },
        "interventions": [],
        "outcomes": [],
    }
    with pytest.raises(TemplateValidationError):
        validate_template("pico", payload, text)
