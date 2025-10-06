import pytest

from Medical_KG_rev.utils.validation import validate_doi, validate_nct_id, validate_pmcid, validate_pmid


def test_validate_nct_id():
    assert validate_nct_id("NCT12345678") == "NCT12345678"
    with pytest.raises(ValueError):
        validate_nct_id("INVALID")


def test_validate_doi():
    assert validate_doi("10.1000/xyz") == "10.1000/XYZ"
    with pytest.raises(ValueError):
        validate_doi("1234")


def test_validate_pmcid_pmid():
    assert validate_pmcid("pmc123") == "PMC123"
    assert validate_pmid("12345") == "12345"
    with pytest.raises(ValueError):
        validate_pmid("abc")
