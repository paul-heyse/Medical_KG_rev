# Ingestion Spec Delta

## ADDED Requirements

### Requirement: Adapter Plugin Stage Wrapper

The ingestion system MUST provide an `IngestStage` implementation that wraps the existing adapter plugin manager. The `PluginIngestStage` class SHALL:

- Implement the `IngestStage` Protocol: `execute(ctx: StageContext, request: AdapterRequest) -> list[RawPayload]`
- Use `get_plugin_manager()` to discover and execute adapters
- Apply resilience policies (retries, circuit breakers, rate limits) at the adapter call boundary
- Emit CloudEvents for adapter start/finish/failure
- Preserve existing adapter execution semantics (fetch → parse → validate flow)

#### Scenario: ClinicalTrials adapter executed via IngestStage

- **GIVEN** a StageContext with tenant_id="tenant-123" and AdapterRequest for NCT04267848
- **WHEN** `PluginIngestStage.execute(ctx, request)` is called
- **THEN** the plugin manager discovers the `clinicaltrials` adapter
- **AND** the adapter's `fetch()` method is invoked with the NCT ID
- **AND** raw payloads are returned as `list[RawPayload]`
- **AND** a CloudEvent `stage.completed` with `data.output_count=1` is emitted

#### Scenario: Adapter failure triggers retry policy

- **GIVEN** an IngestStage with resilience policy "polite-api" (10 retries, linear backoff)
- **WHEN** the adapter call fails with HTTP 429 (rate limit)
- **THEN** the stage retries with 1-second backoff intervals
- **AND** CloudEvents `stage.retrying` are emitted for each attempt
- **AND** if all retries fail, a `stage.failed` event is emitted with error details

---

### Requirement: Parse Stage for IR Conversion

The ingestion system MUST provide a `ParseStage` implementation that converts raw adapter payloads to IR Documents. The `IRParseStage` class SHALL:

- Implement the `ParseStage` Protocol: `execute(ctx: StageContext, payloads: list[RawPayload]) -> Document`
- Use existing adapter `parse()` methods to transform payloads
- Validate the resulting IR Document against Pydantic schema
- Attach provenance metadata (adapter_name, adapter_version, parsed_at timestamp)
- Fail-fast if IR validation fails (no partial documents)

#### Scenario: OpenAlex payload parsed to IR Document

- **GIVEN** a raw OpenAlex API response with work metadata
- **WHEN** `IRParseStage.execute(ctx, payloads)` is called
- **THEN** the payload is parsed into an IR Document with title, authors, abstract
- **AND** the Document has `source="openalex"`, `id="openalex:W2741809807"`
- **AND** metadata includes `adapter_version="1.0.0"`, `parsed_at="2025-01-15T10:30:00Z"`

#### Scenario: IR validation failure aborts pipeline

- **GIVEN** a malformed payload missing required fields (e.g., no `id`)
- **WHEN** `IRParseStage.execute(ctx, payloads)` is called
- **THEN** Pydantic validation raises `ValidationError`
- **AND** the stage catches the error and emits `stage.failed` CloudEvent
- **AND** the job is marked `parse_failed` in Job Ledger with error details
- **AND** no downstream stages execute

---

## MODIFIED Requirements

### Requirement: Adapter Plugin Execution

The adapter plugin manager MUST be invoked through stage contracts rather than directly by the orchestrator. The `AdapterPluginManager` SHALL:

- Be wrapped by `PluginIngestStage` for Dagster ops
- Continue to support direct invocation for backward compatibility (feature flag `MK_USE_DAGSTER=false`)
- Return structured results (`AdapterResponse`) that can be converted to `list[RawPayload]` by the stage wrapper
- Preserve existing plugin discovery, registration, and execution semantics

#### Scenario: Legacy orchestrator invokes plugins directly

- **GIVEN** feature flag `MK_USE_DAGSTER=false`
- **WHEN** `Orchestrator.execute_pipeline(job_id)` is called
- **THEN** the orchestrator invokes `plugin_manager.run(adapter_name, request)` directly
- **AND** no Dagster ops are created
- **AND** no CloudEvents are emitted (existing Kafka topics used instead)

#### Scenario: Dagster orchestration invokes plugins via stage wrapper

- **GIVEN** feature flag `MK_USE_DAGSTER=true`
- **WHEN** `submit_to_dagster(job_id, "auto")` is called
- **THEN** the `ingest_op` wraps `PluginIngestStage.execute()`
- **AND** the plugin manager is invoked inside the stage wrapper
- **AND** CloudEvents are emitted for stage lifecycle
- **AND** resilience policies are applied at the stage boundary

---

## ADDED Requirements

### Requirement: Download Stage for PDF Retrieval

For PDF-bound sources, the ingestion system MUST provide a `DownloadStage` that retrieves full-text PDFs. The `PDFDownloadStage` class SHALL:

- Implement a custom stage contract: `execute(ctx: StageContext, document: Document) -> PDFFile`
- Extract PDF URLs from document metadata (e.g., `document.metadata["pdf_url"]`)
- Download PDFs with streaming to handle large files (>10MB)
- Store PDFs in MinIO with key pattern `tenants/{tenant_id}/pdfs/{doc_id}.pdf`
- Update Job Ledger with `pdf_downloaded=true` on success
- Apply resilience policy "polite-api" with rate limiting to respect OA repositories

#### Scenario: Unpaywall PDF downloaded and stored

- **GIVEN** an IR Document with `metadata["pdf_url"]="https://unpaywall.org/..."`
- **WHEN** `PDFDownloadStage.execute(ctx, document)` is called
- **THEN** the PDF is downloaded via streaming HTTP request
- **AND** the file is stored in MinIO at `tenants/tenant-123/pdfs/unpaywall:10.1234.pdf`
- **AND** the Job Ledger is updated with `pdf_downloaded=true`
- **AND** the pipeline pauses at the PDF gate, awaiting MinerU processing

#### Scenario: PDF download failure retries with backoff

- **GIVEN** a PDF URL that returns HTTP 503 (service unavailable)
- **WHEN** the download stage executes
- **THEN** the stage retries 10 times with linear backoff (polite-api policy)
- **AND** CloudEvents `stage.retrying` are emitted
- **AND** if all retries fail, the job is marked `pdf_download_failed` with URL in error details
