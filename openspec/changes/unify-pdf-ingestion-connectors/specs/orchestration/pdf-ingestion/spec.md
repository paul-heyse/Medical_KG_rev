## ADDED Requirements

### Requirement: PDF-capable adapters emit standardized manifests
Adapters that declare PDF capabilities SHALL implement the shared PDF manifest contract, exposing `pdf_assets`, normalized `pdf_urls`, checksum hints, and polite-pool metadata.

#### Scenario: Adapter provides manifest for downloadable work
- **WHEN** a PDF-capable adapter fetches a work that includes one or more downloadable assets
- **THEN** the adapter SHALL return a manifest containing at least the source URL, inferred license, and optional checksum hints for each asset
- **AND** the adapter SHALL mark the document metadata with `document_type="pdf"` so the pipeline can route it to the PDF flow

### Requirement: Connector configuration enforces polite pool compliance
The system SHALL provide connector-specific settings (contact email, user-agent, rate limits, timeouts) and automatically apply them to outbound HTTP clients for PDF-capable adapters.

#### Scenario: Adapter uses configured polite headers
- **WHEN** operators set polite-pool configuration for a connector (e.g., OpenAlex, Unpaywall, Crossref, PMC)
- **THEN** the adapter SHALL issue outbound requests with the configured contact email and user-agent
- **AND** the rate limiter SHALL enforce the configured requests-per-second budget

### Requirement: Storage-aware download stage persists PDF assets
The PDF download stage SHALL use the storage client to persist downloaded assets, emit metrics, and update the job ledger with storage receipts for each successful download.

#### Scenario: Stage stores PDF and records ledger state
- **WHEN** the download stage receives a manifest containing a valid PDF URL
- **THEN** it SHALL download the asset with retries, reject files that exceed configured limits, and store successful downloads via the object storage client
- **AND** it SHALL record the resulting storage URI, checksum, and byte size in both the pipeline state and the job ledger

### Requirement: PDF-capable adapters route through the two-phase pipeline
The orchestration layer SHALL automatically route any adapter declaring PDF capabilities to the PDF two-phase pipeline while leaving non-PDF adapters on their existing topologies.

#### Scenario: Capability-aware routing selects PDF pipeline
- **WHEN** an ingest request targets an adapter whose manifest declares the `pdf` capability
- **THEN** the pipeline builder SHALL enqueue the `pdf-two-phase` topology (ingest → download → gate → chunk …)
- **AND** adapters without the capability SHALL continue to use their configured non-PDF pipelines

### Requirement: PDF pipeline gating validates MinerU readiness
The two-phase PDF pipeline SHALL require both successful downloads and MinerU readiness signals before resuming downstream stages.

#### Scenario: Gate blocks pipeline until MinerU completes
- **WHEN** the pipeline executes the PDF gate stage before chunking
- **THEN** the gate SHALL confirm the job ledger flags `pdf_downloaded=true` and `pdf_ir_ready=true`
- **AND** if either flag is false within the configured timeout, the gate SHALL block execution and emit an operational alert

### Requirement: Download failures emit telemetry without advancing the pipeline
The system SHALL surface failed downloads via structured logs and metrics, refrain from marking the ledger as downloaded, and leave the pipeline in a blocked state until remediation.

#### Scenario: Oversized PDF triggers failure telemetry
- **WHEN** the download stage encounters a PDF exceeding the configured max file size
- **THEN** it SHALL log a failure event, increment the failure metric with reason `size_limit`, and skip persisting storage metadata
- **AND** the job ledger SHALL keep `pdf_downloaded=false`, causing the downstream gate to block progression
