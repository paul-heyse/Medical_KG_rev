# Change Proposal: Biomedical Data Source Adapters

## Why

Implement plug-in adapters for ingesting data from 10+ biomedical APIs and databases including clinical trials, research literature, drug labels, ontologies, and chemistry databases. These adapters transform diverse external formats into our unified IR (Intermediate Representation) for downstream processing.

## What Changes

- **Clinical Trials**: ClinicalTrials.gov API v2 adapter (NCT IDs)
- **Drug Data**: OpenFDA (drug labels, adverse events, devices), DailyMed SPL adapter
- **Literature**: OpenAlex (with pyalex), PubMed Central/Europe PMC, Unpaywall, Crossref, CORE PDF access
- **Ontologies**: RxNorm (drug normalization), ICD-11 (diagnoses), MeSH terms
- **Chemistry**: ChEMBL adapter for molecular data
- **Enrichment**: Semantic Scholar for citations
- YAML-based adapter configurations for simple REST APIs
- Python adapter implementations for complex sources (PDFs, SOAP)
- Rate limiting and retry logic per source
- Adapter registry for dynamic discovery
- Comprehensive adapter tests with fixtures

## Impact

- **Affected specs**: NEW capability `biomedical-adapters`
- **Affected code**:
  - `src/Medical_KG_rev/adapters/clinicaltrials.py`
  - `src/Medical_KG_rev/adapters/openfda.py` (drug labels, events, devices)
  - `src/Medical_KG_rev/adapters/openalex.py`
  - `src/Medical_KG_rev/adapters/pmc.py` (PubMed Central + Europe PMC)
  - `src/Medical_KG_rev/adapters/unpaywall.py`
  - `src/Medical_KG_rev/adapters/crossref.py`
  - `src/Medical_KG_rev/adapters/core.py` (PDF access)
  - `src/Medical_KG_rev/adapters/rxnorm.py`
  - `src/Medical_KG_rev/adapters/icd11.py`
  - `src/Medical_KG_rev/adapters/chembl.py`
  - `src/Medical_KG_rev/adapters/semantic_scholar.py`
  - `adapters/configs/*.yaml` - Declarative adapter configs
  - `tests/adapters/` - Adapter tests with mocked API responses
