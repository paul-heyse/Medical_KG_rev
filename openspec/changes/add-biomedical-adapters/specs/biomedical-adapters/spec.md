# Biomedical Data Source Adapters Specification

## ADDED Requirements

### Requirement: ClinicalTrials.gov Adapter

The system SHALL provide an adapter for ingesting clinical trial data from ClinicalTrials.gov API v2.

#### Scenario: Fetch trial by NCT ID

- **WHEN** given valid NCT ID (e.g., "NCT04267848")
- **THEN** adapter MUST fetch from API v2 and create Document with trial metadata

#### Scenario: NCT ID validation

- **WHEN** given invalid NCT ID format
- **THEN** adapter MUST reject immediately without API call

#### Scenario: Structured field extraction

- **WHEN** parsing CT.gov response
- **THEN** adapter MUST extract phase, status, interventions, outcomes, eligibility into metadata

### Requirement: OpenFDA Adapters

The system SHALL provide adapters for OpenFDA drug labels, adverse events, and medical devices.

#### Scenario: Drug label by NDC

- **WHEN** OpenFDADrugLabelAdapter fetches by NDC code
- **THEN** SPL document MUST be retrieved and parsed to IR

#### Scenario: Adverse event data

- **WHEN** OpenFDADrugEventAdapter queries by drug name
- **THEN** adverse event reports MUST be returned with structured data

#### Scenario: Medical device data

- **WHEN** OpenFDADeviceAdapter queries by device ID
- **THEN** device information MUST be retrieved and normalized

### Requirement: Literature Adapters (OpenAlex, PMC, Unpaywall, Crossref, CORE)

The system SHALL provide adapters for ingesting open-access research literature with metadata and full text.

#### Scenario: OpenAlex search query

- **WHEN** OpenAlexAdapter searches "lung cancer immunotherapy"
- **THEN** relevant papers MUST be returned with metadata (title, authors, DOI, abstract, OA status)

#### Scenario: Multi-source enrichment

- **WHEN** OpenAlex returns OA paper
- **THEN** system MUST fetch full text via Unpaywall → CORE → PDF download pipeline

#### Scenario: PubMed Central full text

- **WHEN** PMCAdapter fetches by PMCID
- **THEN** full-text XML MUST be retrieved from Europe PMC and parsed

#### Scenario: Crossref metadata lookup

- **WHEN** CrossrefAdapter fetches by DOI
- **THEN** citation metadata MUST be returned

### Requirement: Ontology Normalization Adapters

The system SHALL provide adapters for normalizing terms to standard ontologies.

#### Scenario: Drug name to RxNorm

- **WHEN** RxNormAdapter lookups "atorvastatin"
- **THEN** RxCUI code MUST be returned with term details

#### Scenario: Diagnosis to ICD-11

- **WHEN** ICD11Adapter searches "Type 2 Diabetes"
- **THEN** ICD-11 code and URI MUST be returned

#### Scenario: MeSH term lookup

- **WHEN** MeSHAdapter queries medical concept
- **THEN** MeSH descriptor MUST be returned with tree numbers

### Requirement: ChEMBL Chemistry Adapter

The system SHALL provide an adapter for ChEMBL molecular and compound data.

#### Scenario: Molecule by ChEMBL ID

- **WHEN** ChEMBLAdapter fetches "CHEMBL25"
- **THEN** molecular structure, targets, and activity data MUST be returned

#### Scenario: SMILES search

- **WHEN** searching by SMILES string
- **THEN** matching compounds MUST be returned

### Requirement: Semantic Scholar Citation Adapter

The system SHALL provide an adapter for enriching literature with citation data.

#### Scenario: Citation count by DOI

- **WHEN** SemanticScholarAdapter fetches paper
- **THEN** citation count and reference list MUST be returned

### Requirement: Rate Limiting Per Adapter

Each adapter SHALL respect source-specific rate limits using token bucket algorithm.

#### Scenario: OpenAlex rate limit compliance

- **WHEN** making requests to OpenAlex
- **THEN** adapter MUST not exceed 100k requests/day (polite pool)

#### Scenario: Rate limit exceeded handling

- **WHEN** rate limit is hit
- **THEN** adapter MUST delay requests to stay within limits

### Requirement: YAML Adapter Configuration

The system SHALL support defining simple REST API adapters declaratively in YAML without Python code.

#### Scenario: YAML config parsing

- **WHEN** adapter defined in clinicaltrials.yaml
- **THEN** system MUST generate adapter class automatically

#### Scenario: Config validation

- **WHEN** YAML config is loaded
- **THEN** schema MUST be validated for required fields

### Requirement: Adapter Resilience

All adapters SHALL implement retry logic with exponential backoff and circuit breaking.

#### Scenario: Transient error retry

- **WHEN** API returns 503
- **THEN** adapter MUST retry with exponential backoff up to 3 attempts

#### Scenario: Circuit breaker opening

- **WHEN** endpoint fails 5 times consecutively
- **THEN** adapter MUST open circuit breaker and fail fast for 60 seconds

#### Scenario: Respect Retry-After header

- **WHEN** API returns 429 with Retry-After
- **THEN** adapter MUST wait specified duration before retry
