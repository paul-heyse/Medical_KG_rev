import pytest

from Medical_KG_rev.validation.fhir import FHIRValidationError, FHIRValidator


def _evidence_resource():
    return {
        "resourceType": "Evidence",
        "status": "active",
        "description": "Hypertension reduction evidence",
        "outcome": {"reference": "Observation/1"},
        "characteristic": [
            {
                "code": {
                    "coding": [
                        {"system": "http://loinc.org", "code": "1234-5", "display": "Systolic"}
                    ]
                }
            }
        ],
    }


def test_valid_evidence_resource():
    validator = FHIRValidator()
    validator.validate(_evidence_resource())


def test_invalid_missing_required_field():
    validator = FHIRValidator()
    resource = _evidence_resource()
    resource.pop("description")
    with pytest.raises(FHIRValidationError):
        validator.validate(resource)


def test_invalid_coding_system():
    validator = FHIRValidator()
    resource = _evidence_resource()
    resource["characteristic"][0]["code"]["coding"][0].pop("system")
    with pytest.raises(FHIRValidationError) as exc:
        validator.validate(resource)
    assert "Coding.system" in "; ".join(exc.value.errors)
