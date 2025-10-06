# Data Models Guide

This guide describes the federated intermediate representation (IR) and related
models defined in `Medical_KG_rev.models`.

## Core IR

- `Document` → Top-level container with sections, metadata and provenance.
- `Section` → Grouping of `Block` instances.
- `Block` → Atomic textual unit such as paragraph, table or figure.
- `Span` → Character offsets within blocks used to maintain traceability back to
  source documents.

Validation is strict: blocks require unique identifiers, spans must be ordered
and constrained by the owning block content, and documents enforce UTC
timestamps.

## Entities and Evidence

- `Entity` → Canonical concept linked to spans.
- `Claim` → Semantic assertion relating entities.
- `Evidence` → Supporting snippet tying claims/entities back to the document.
- `ExtractionActivity` → Provenance metadata indicating how an artifact was
  derived.

## Domain Overlays

- `MedicalDocument` → Adds `ResearchStudy` metadata and `EvidenceAssessment`
  alignments with FHIR resources.
- `FinancialDocument` → Encapsulates XBRL contexts and facts.
- `LegalDocument` → Captures legal clauses and references consistent with
  LegalDocML.

Each overlay enforces domain-specific rules while remaining compatible with the
core IR.
