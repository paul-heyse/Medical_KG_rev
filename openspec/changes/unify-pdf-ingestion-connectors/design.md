## Context
The existing PDF ingestion story spans multiple layers: adapters emit ad-hoc `pdf_urls`, the orchestration pipeline uses stub download stages, and MinerU gating depends on manual ledger toggles. Only the OpenAlex adapter leverages `pyalex`, and PDF assets are not persisted consistently. We need a unified contract and pipeline that supports all PDF-capable connectors, surfaces polite-pool settings, and records storage outcomes for downstream MinerU processing. The change must also preserve backward compatibility for consumers that still read `pdf_urls` directly while we migrate.

## Goals / Non-Goals
- Goals:
  - Define a capability contract that lets any adapter advertise PDF assets in a consistent manifest.
  - Ensure the two-phase pipeline downloads, stores, and gates PDFs via the storage-aware stage.
  - Centralise polite-pool configuration and rate limiting for adapters that make outbound HTTP calls.
  - Provide observability (metrics/logs) and automated tests that cover OpenAlex and at least one additional connector.
- Non-Goals:
  - Replacing MinerU itself or altering its interface.
  - Introducing real network calls in test suites; use mocks/fakes for upstream APIs and storage.
  - Changing non-PDF pipelines or adapters lacking PDF capabilities.

## Decisions
- **Adapter Protocol**: Introduce `PdfCapableAdapter` (ABC or Protocol) exposing `iter_pdf_candidates()` and `pdf_capabilities` metadata. Adapters use a mixin to normalize manifests and attach polite headers.
  - *Alternatives*: Continue relying on metadata dicts; rejected due to lack of type guarantees and discoverability.
- **Manifest Schema**: Define `PdfAssetManifest` with fields `{url, landing_page_url, license, version, source, checksum_hint, is_open_access, content_type}` plus top-level metadata (`retrieved_at`, `connector`, `polite_headers`). Store canonical representation on the `Document.metadata["pdf_manifest"]` key.
  - *Alternatives*: Multiple ad-hoc dictionaries; rejected because maintenance cost grows with each connector.
- **Polite Pool Configuration**: Extend connector settings to include email, user-agent, rate limits, and HTTP timeouts. Inject via factories so stage plugins and adapters share config.
  - *Alternatives*: Environment variables per adapter; rejected because it scatters configuration.
- **Download Stage**: Reuse `StorageAwarePdfDownloadStage` for both sync and async flows after fixing the current implementation. Register it via stage plugins guarded by `capabilities=("pdf",)`.
  - *Alternatives*: Build a brand-new stage; rejected since existing stage already covers storage needs once repaired.
- **Manifest Persistence**: Pipeline state stores the manifest and storage receipts, and the job ledger records `pdf_downloaded`, `pdf_checksum`, and `storage_uri`.
  - *Alternatives*: Rely solely on storage metadata; rejected because ledger gating needs quick lookups.
- **Observability**: Emit Prometheus counters/histograms and structured logs with per-connector labels.
  - *Alternatives*: Logging only; rejected because QA needs quantitative signals.
- **Feature Flag Rollout**: Guard the new manifest-to-pipeline wiring behind `ENABLE_PDF_MANIFEST_PIPELINE` until all connectors pass staging verification.
  - *Alternatives*: Big-bang deployment; rejected due to migration risk.

## Detailed Design

### Adapter Contract
- Introduce `PdfCapableAdapter` protocol exposing:
  - `pdf_capabilities: tuple[str, ...]` (must include `"pdf"`)
  - `iter_pdf_candidates(self, context: AdapterContext) -> Iterable[PdfAssetManifest]`
  - `polite_headers(self) -> Mapping[str, str]`
- `PdfManifestMixin` responsibilities:
  - Normalise URLs (strip whitespace, enforce HTTPS when possible).
  - Flatten OpenAlex/Unpaywall structures into `PdfAssetManifest`.
  - Deduplicate by URL + version.
  - Provide helper `with_pdf_manifest(documents: Sequence[Document]) -> Sequence[Document]` that injects `pdf_manifest` into metadata.
- Backward compatibility:
  - Continue populating `metadata["pdf_urls"]` for existing consumers.
  - Emit deprecation warning via structlog when manifest is available but legacy fields are accessed.

### Configuration Surface
- Add `ConnectorPdfSettings` in `config/settings.py` with:
  - `contact_email`, `user_agent`, `requests_per_second`, `burst`, `timeout_seconds`, `max_file_size_mb`, `retry_attempts`, `retry_backoff_seconds`.
- Update `OpenAlexSettings`, `UnpaywallSettings`, `CrossrefSettings`, `PMCSettings` to inherit or include `pdf: ConnectorPdfSettings`.
- Extend `StagePluginResources` with `pdf_settings: dict[str, ConnectorPdfSettings]` and `rate_limiters: dict[str, Limiter]`.
- Document env vars: `MK_OPENALEX__PDF__CONTACT_EMAIL`, etc.

### Storage-aware Download Stage
- Split into `PdfDownloadWorker` (pure helper) shared by sync/async stage classes.
- Use `httpx` streaming to write into memory buffer with size guard; fallback to chunked streaming for large files.
- Compute checksum (SHA256) once and reuse for storage + ledger.
- Metrics:
  - Histogram `pdf_download_latency_seconds` labelled by connector & outcome.
  - Counter `pdf_download_failures_total` labelled by reason (`timeout`, `http_status`, `content_type`, `size_limit`).
  - Counter `pdf_download_bytes_total`.
- Logging:
  - `pdf_download.retry` with attempt count.
  - `pdf_download.skipped` when manifest flagged but download not attempted (e.g., unsupported protocol).

### Pipeline Integration
- Ingest stage obtains documents from adapter; for manifest-aware adapters, attach manifest to pipeline state via `state.set_pdf_manifest`.
- `PipelineState` gains `pdf_manifest: PdfManifest | None` property with serializer to persist in checkpoints if needed.
- Ledger updates:
  - Extend `JobLedgerEntry` with `pdf_checksum`, `pdf_storage_uri`, `pdf_last_download_attempt`, `pdf_connector`.
  - Provide atomic setter `set_pdf_storage(job_id, checksum, uri, size_bytes)`.
- Gate Stage:
  - Accepts `required_flags=("pdf_downloaded", "pdf_ir_ready")`.
  - Surfaces gating failures with reason codes for observability.

### Observability & Rate Limiting
- `aiolimiter.AsyncLimiter` per connector stored in `Adapter` instance.
- For sync flows, use `rate_limiter.acquire()` context manager via `anyio`.
- Trace integration: attach polite headers and download latency to existing OpenTelemetry spans.
- Alerts:
  - Fire on `pdf_download_failures_total` > threshold within 10 minutes.
  - Gate timeout triggers structlog warning `pdf_gate.timeout`.

### Testing
- Unit tests mock httpx to simulate streaming, retries, and oversize.
- Integration tests use `FakePdfStorageClient` storing bytes in dict.
- Regression tests ensure connectors without manifest remain unaffected (capability flag false).

## Risks / Trade-offs
- **Complex adapter refactor** → Mitigation: Introduce protocol and mixin incrementally, keep legacy fields until connectors migrate.
- **Increased stage complexity** → Mitigation: Centralize retry/backoff behavior in helper methods and cover with unit tests.
- **Configuration sprawl** → Mitigation: Document new settings, supply safe defaults, and validate via Pydantic schemas.
- **Performance regressions** → Mitigation: Provide rate limits and streaming downloads where possible; include metrics for monitoring throughput.

## Migration Plan
1. Land adapter protocol/mixin with feature flag leaving old metadata untouched.
2. Update OpenAlex adapter and pipeline to consume the new manifest behind flag; run integration tests.
3. Migrate additional connectors one by one, verifying tests and metrics.
4. Switch pipeline default to the new storage-aware stages; remove stub implementations.
5. Update documentation and announce the new contract to downstream teams.

## Open Questions
- Do we need per-connector overrides for storage retention policies?
- Should we expose a CLI/tooling command to inspect stored PDF manifests per job?
- Is MinerU gate readiness inferred solely from storage success or does it need additional signals (e.g., checksum validation)?
- How do we phase out legacy `pdf_urls` once clients adopt the manifest?
- Should we push manifest schema to downstream storage/IR services for reuse?
