"""Domain specific validation helpers."""
from __future__ import annotations

import re
from typing import Pattern

NCT_ID_PATTERN: Pattern[str] = re.compile(r"^NCT\d{8}$", re.IGNORECASE)
DOI_PATTERN: Pattern[str] = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
PMCID_PATTERN: Pattern[str] = re.compile(r"^PMC\d+$", re.IGNORECASE)
PMID_PATTERN: Pattern[str] = re.compile(r"^\d{1,8}$")
NDC_PATTERN: Pattern[str] = re.compile(
    r"^(\d{4}-\d{3}-\d{2}|\d{5}-\d{3}-\d{1}|\d{4}-\d{4}-\d{2}|\d{5}-\d{4}-\d{1}|\d{11})$",
    re.IGNORECASE,
)
SETID_PATTERN: Pattern[str] = re.compile(r"^[0-9A-F]{32}$", re.IGNORECASE)
RXCUI_PATTERN: Pattern[str] = re.compile(r"^\d+$")
ICD11_PATTERN: Pattern[str] = re.compile(r"^[0-9A-Z]{3,8}$", re.IGNORECASE)
MESH_ID_PATTERN: Pattern[str] = re.compile(r"^D\d{6}$", re.IGNORECASE)
CHEMBL_ID_PATTERN: Pattern[str] = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)


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


def validate_ndc(value: str) -> str:
    return _validate(value, NDC_PATTERN, "NDC code")


def validate_set_id(value: str) -> str:
    return _validate(value, SETID_PATTERN, "set ID")


def validate_rxcui(value: str) -> str:
    return _validate(value, RXCUI_PATTERN, "RxCUI")


def validate_icd11(value: str) -> str:
    return _validate(value, ICD11_PATTERN, "ICD-11 code")


def validate_mesh_id(value: str) -> str:
    return _validate(value, MESH_ID_PATTERN, "MeSH descriptor ID")


def validate_chembl_id(value: str) -> str:
    return _validate(value, CHEMBL_ID_PATTERN, "ChEMBL identifier")
