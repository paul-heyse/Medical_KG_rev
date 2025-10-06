import pytest

from Medical_KG_rev.validation.ucum import UCUMValidator, UnitValidationError


def test_validate_measurement_normalises_units():
    validator = UCUMValidator()
    result = validator.validate_measurement("20 milligrams per deciliter", context="lab")
    assert result.normalized_unit == "mg/dL"
    assert result.normalized_value == pytest.approx(20)


def test_invalid_unit_raises():
    validator = UCUMValidator()
    with pytest.raises(UnitValidationError):
        validator.validate_value(10, "bananas", context="dose")


def test_range_validation():
    validator = UCUMValidator()
    with pytest.raises(UnitValidationError):
        validator.validate_value(6000, "mg", context="dose")


def test_missing_unit():
    validator = UCUMValidator()
    with pytest.raises(UnitValidationError):
        validator.validate_measurement("20", context="dose")
