# Design Document: Biomedical Data Source Adapters

## Context

The system must ingest data from 10+ heterogeneous biomedical sources with different APIs (REST, SOAP), authentication schemes (none, API key, OAuth), and data formats (JSON, XML, PDF). Each source has unique rate limits and availability constraints.

## Goals / Non-Goals

### Goals

- Unified adapter interface for all sources
- Declarative YAML configs for simple REST APIs
- Rate limiting per source to respect API limits
- Resilience (retry, backoff, circuit breaking)
- Comprehensive test coverage with mocked APIs

### Non-Goals

- Not building a general-purpose ETL tool (domain-specific)
- Not supporting real-time streaming (batch/on-demand only)
- Not storing raw API responses (only normalized IR)

## Decisions

### Decision 1: Adapter Class Hierarchy

```python
BaseAdapter (abstract)
├── RESTAdapter (for JSON APIs)
│   ├── ClinicalTrialsAdapter
│   ├── OpenFDAAdapter
│   └── OpenAlexAdapter
├── SOAPAdapter (for SOAP/XML)
│   └── EuropePMCAdapter
└── PDFAdapter (for document processing)
    └── COREAdapter
```

### Decision 2: YAML Config Format (Singer/Airbyte-inspired)

```yaml
source: "ClinicalTrialsAPI"
base_url: "https://clinicaltrials.gov/api/v2"
rate_limit:
  requests: 10
  per_seconds: 1
auth:
  type: "none"
endpoints:
  get_study:
    path: "/studies/{nct_id}"
    method: "GET"
    params:
      format: "json"
    mapping:
      document_id: "$.protocolSection.identificationModule.nctId"
      title: "$.protocolSection.identificationModule.briefTitle"
```

### Decision 3: Rate Limiting Strategy

- Token bucket algorithm per adapter
- Configurable per source (OpenAlex: 10/sec, OpenFDA: varies by key)
- Shared rate limiter state (Redis for multi-instance)
- 429 responses trigger exponential backoff

### Decision 4: Multi-Source Literature Pipeline

OpenAlex → Unpaywall (OA check) → CORE (PDF) → MinerU (parse)

- Metadata first, then full-text enrichment
- Merge metadata + content into single Document

### Decision 5: Ontology Adapter Pattern

- Normalization adapters (RxNorm, ICD-11) return mappings, not Documents
- Used in extraction/mapping phase, not ingestion
- Cache ontology lookups (high reuse)

## Data Flow Example: Literature Ingestion

```
1. OpenAlexAdapter.fetch("lung cancer")
   → Returns 50 paper metadata Documents
2. For each with is_oa=true:
   3a. UnpaywallAdapter.get_pdf_url(doi)
   3b. If PDF URL: COREAdapter.download_pdf(doi)
   3c. Store PDF in S3, trigger MinerU
4. MinerU parses PDF → content Blocks
5. Merge metadata + content → final Document
```

## Risks / Trade-offs

### Risk 1: API Rate Limit Violations

**Mitigation**: Conservative rate limits, exponential backoff, respect Retry-After headers

### Risk 2: API Schema Changes

**Mitigation**: Version adapter configs, add schema validation, alerting on parse failures

### Risk 3: Large Volume PDF Downloads

**Mitigation**: Async downloads, queue-based processing, storage quotas

## Migration Plan

New capability. Adapters are pluggable; adding new sources doesn't affect existing ones.

## Open Questions

1. **Q**: Cache API responses to reduce redundant calls?
   **A**: Yes, with TTL (e.g., 24h for metadata, no cache for real-time data)

2. **Q**: Support pagination for large result sets?
   **A**: Yes, adapters handle pagination internally, expose as iterator
