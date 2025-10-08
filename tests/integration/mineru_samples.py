"""Sample MinerU documents used for integration testing."""

from __future__ import annotations

from textwrap import dedent

SAMPLE_DOCUMENTS: dict[str, bytes] = {
    "doc-basic": dedent(
        """
        Clinical Study Report
        Introduction
        Results Summary
        Conclusions
        """
    ).encode("utf-8"),
    "doc-table": dedent(
        """
        Arm | Dose | Outcome
        Control | 5mg | Stable
        Treatment | 10mg | Improved
        """
    ).encode("utf-8"),
    "doc-multi": dedent(
        """
        Background

        Methods

        Findings
        """
    ).encode("utf-8"),
}

E2E_PRIMARY_DOCUMENT_ID = "doc-table"
