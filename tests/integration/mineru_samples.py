"""Sample MinerU documents used for integration testing."""

from __future__ import annotations

from textwrap import dedent

SAMPLE_DOCUMENTS: dict[str, bytes] = {
    "doc-01-overview": dedent(
        """
        Study Overview
        Background
        Objectives
        Methods
        Results
        Conclusions
        """
    ).encode("utf-8"),
    "doc-02-cohort-table": dedent(
        """
        Cohort | Count | Response
        Placebo | 42 | Stable
        Treatment | 44 | Improved
        Observation | 12 | Withdrawn
        """
    ).encode("utf-8"),
    "doc-03-multi-page": (
        b"Baseline Characteristics\nAge Distribution\n\fOutcome Measures\nPrimary Endpoint\nSecondary Endpoint\n"
    ),
    "doc-04-adverse-events": dedent(
        """
        Adverse Events Summary
        Mild headache in 10% of participants
        Injection site pain in 5% of participants
        No serious adverse events reported
        """
    ).encode("utf-8"),
    "doc-05-lab-results": dedent(
        """
        Analyte | Visit | Value
        Hemoglobin | Baseline | 12.5
        Hemoglobin | Week 12 | 13.2
        Platelets | Baseline | 220
        """
    ).encode("utf-8"),
    "doc-06-trial-design": dedent(
        """
        Randomised controlled trial
        Double blind design
        Parallel assignment
        Stratified by site
        Conducted across 12 locations
        """
    ).encode("utf-8"),
    "doc-07-population-table": dedent(
        """
        Group | Age | BMI
        Control | 57 | 26.4
        Active | 58 | 25.8
        Extension | 55 | 27.1
        """
    ).encode("utf-8"),
    "doc-08-efficacy": dedent(
        """
        Efficacy Outcomes
        Significant reduction in symptom scores
        Sustained response through week 24
        Benefits observed across subgroups
        """
    ).encode("utf-8"),
    "doc-09-safety": dedent(
        """
        Safety Monitoring
        Weekly laboratory assessments
        Monthly ECG evaluations
        No dose reductions required
        """
    ).encode("utf-8"),
    "doc-10-methods": dedent(
        """
        Methods Overview
        Inclusion criteria outlined
        Exclusion criteria listed
        Statistical analysis plan described
        """
    ).encode("utf-8"),
    "doc-11-endpoints": dedent(
        """
        Primary endpoint: Change in pain score
        Secondary endpoint: Mobility index improvement
        Exploratory endpoint: Biomarker response
        """
    ).encode("utf-8"),
    "doc-12-timeline-table": dedent(
        """
        Visit | Window | Assessments
        Screening | -14 to 0 | Eligibility, Labs
        Baseline | Day 0 | Randomisation, Labs
        Follow-up | Day 28 | Physical exam
        """
    ).encode("utf-8"),
    "doc-13-quality": dedent(
        """
        Quality Assurance Measures
        Source data verification performed
        Protocol adherence reviewed weekly
        Deviations documented and resolved
        """
    ).encode("utf-8"),
    "doc-14-subgroup-table": dedent(
        """
        Subgroup | N | Outcome
        Male | 48 | Responder
        Female | 50 | Responder
        Non-binary | 4 | Partial responder
        """
    ).encode("utf-8"),
    "doc-15-biomarkers": dedent(
        """
        Biomarker Analysis
        CRP reduction observed in majority
        IL-6 levels decreased post-treatment
        Gene expression changes noted
        """
    ).encode("utf-8"),
    "doc-16-imaging": dedent(
        """
        Imaging Review
        MRI indicates lesion reduction
        CT corroborates findings
        Blinded radiologist assessment
        """
    ).encode("utf-8"),
    "doc-17-follow-up": dedent(
        """
        Follow-up Schedule
        Telephone visit at week 2
        Clinic visit at week 6
        Final assessment at week 12
        """
    ).encode("utf-8"),
    "doc-18-dose-response": dedent(
        """
        Dose | Exposure | Response
        5mg | 1.2 | Minimal
        10mg | 2.4 | Moderate
        15mg | 3.6 | Robust
        """
    ).encode("utf-8"),
    "doc-19-compliance": dedent(
        """
        Compliance Tracking
        Dosing diaries reviewed
        Electronic monitoring utilised
        Overall adherence above 92%
        """
    ).encode("utf-8"),
    "doc-20-summary": dedent(
        """
        Executive Summary
        Therapy met primary endpoint
        Secondary endpoints trending positive
        Safety profile acceptable
        Future studies recommended
        """
    ).encode("utf-8"),
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
        Control | 5 mg | Stable
        Treatment | 10 mg | Improved
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

E2E_PRIMARY_DOCUMENT_ID = "doc-05-lab-results"
