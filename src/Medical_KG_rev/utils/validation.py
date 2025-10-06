"""Domain specific validation helpers."""
from __future__ import annotations

import re
from typing import Pattern

NCT_ID_PATTERN: Pattern[str] = re.compile(r"^NCT\d{8}$", re.IGNORECASE)
DOI_PATTERN: Pattern[str] = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
PMCID_PATTERN: Pattern[str] = re.compile(r"^PMC\d+$", re.IGNORECASE)
PMID_PATTERN: Pattern[str] = re.compile(r"^\d{1,8}$")


def _validate(value: str, pattern: Pattern[str], label: str) -> str:
    if not pattern.match(value):
        raise ValueError(f"Invalid {label}: {value}")
    return value.upper()


def validate_nct_id(value: str) -> str:
    return _validate(value, NCT_ID_PATTERN, "NCT identifier")


def validate_doi(value: str) -> str:
    return _validate(value, DOI_PATTERN, "DOI")


def validate_pmcid(value: str) -> str:
    return _validate(value, PMCID_PATTERN, "PMCID")


def validate_pmid(value: str) -> str:
    return _validate(value, PMID_PATTERN, "PMID")
