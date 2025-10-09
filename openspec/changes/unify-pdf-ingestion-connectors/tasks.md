## 0. Specification & Validation
- [ ] 0.1 Review `openspec/project.md`, `openspec list`, and relevant archived changes to avoid overlap.
- [ ] 0.2 Confirm baseline requirements in `openspec/specs/orchestration/pdf-ingestion` (once added) match current behaviour.
- [ ] 0.3 Draft and iterate on spec deltas in `openspec/changes/unify-pdf-ingestion-connectors/specs/orchestration/pdf-ingestion/spec.md`.
- [ ] 0.4 Run `openspec validate unify-pdf-ingestion-connectors --strict` after every update and capture issues.

## 1. Adapter Contract & Manifest
- [ ] 1.1 Add `PdfAssetManifest` dataclass + `PdfCapableAdapter` protocol under `src/Medical_KG_rev/adapters/interfaces/pdf.py`.
- [ ] 1.2 Create `PdfManifestMixin` in `src/Medical_KG_rev/adapters/mixins/` that normalises URLs, licenses, versions, and deduplicates assets.
- [ ] 1.3 Update OpenAlex, Unpaywall, Crossref, and PMC adapters to inherit the mixin, implement `iter_pdf_candidates`, and set `capabilities=("pdf",)`.
- [ ] 1.4 Ensure adapters populate polite-pool metadata (contact email, user-agent) on client initialisation.
- [ ] 1.5 Retain backward-compatible fields (`pdf_urls`, `document_type`) for consumers until deprecation notice.
- [ ] 1.6 Add adapter-specific unit tests validating manifest construction, duplicate removal, and polite header population (`tests/adapters/test_openalex_adapter.py`, etc.).

## 2. Configuration Surface
- [ ] 2.1 Introduce shared `ConnectorPdfSettings` model in `src/Medical_KG_rev/config/settings.py` with email, user-agent, rate limiter, timeout, and max file size options.
- [ ] 2.2 Map OpenAlex, Unpaywall, Crossref, and PMC settings onto the shared model; add defaults and env var bindings.
- [ ] 2.3 Update configuration loaders (`get_settings()`) and factory code so adapters receive the proper settings object.
- [ ] 2.4 Document new environment variables in `docs/api/adapters.md`, `docs/api/orchestration.md`, and `.env.example`.
- [ ] 2.5 Extend config tests (`tests/config/test_settings.py`) to cover validation of the new settings block.

## 3. Storage-aware Download Stage Hardening
- [ ] 3.1 Refactor `StorageAwarePdfDownloadStage` to share synchronous and asynchronous helpers, removing stray `await` usage in sync paths.
- [ ] 3.2 Implement streaming downloads with size enforcement and explicit checksum calculation prior to storage.
- [ ] 3.3 Add configurable retry policy via `tenacity`, and surface timeout / backoff derived from settings.
- [ ] 3.4 Emit structured logs and Prometheus metrics (`pdf_download_latency_seconds`, `pdf_download_bytes_total`, `pdf_download_failures_total`) tagged by connector.
- [ ] 3.5 Write unit tests for success, retry exhaustion, oversize rejection, and storage exceptions (`tests/orchestration/test_pdf_download_stage.py`).

## 4. Stage Plugin & Pipeline Integration
- [ ] 4.1 Replace stub `PdfDownloadStage` registration in `src/Medical_KG_rev/orchestration/dagster/stages.py` with the storage-aware implementation guarded by capability checks.
- [ ] 4.2 Extend `StagePluginResources` to inject pdf settings and rate limiters into stage builders.
- [ ] 4.3 Update pipeline state handling so ingest stages preserve `Document` objects and attach the PDF manifest to `PipelineState`.
- [ ] 4.4 Enhance job ledger entries (`src/Medical_KG_rev/orchestration/ledger.py`) with storage URI, checksum, and timestamp fields; migrate in-memory store accordingly.
- [ ] 4.5 Adjust gate stage logic to assert both `pdf_downloaded` and `pdf_ir_ready` with configurable timeout pulled from settings.
- [ ] 4.6 Update `config/orchestration/pipelines/pdf-two-phase.yaml` to include connector capability routing and feature flag toggles.

## 5. Observability & Rate Limiting
- [ ] 5.1 Integrate `aiolimiter` (or equivalent) per connector and ensure adapters respect the configured requests-per-second.
- [ ] 5.2 Register Prometheus collectors for download metrics and add dashboards/runbook entries.
- [ ] 5.3 Add structured alert hooks (e.g., logging keys) for repeated download failures or gate timeouts.
- [ ] 5.4 Extend tracing instrumentation via httpx event hooks to capture request IDs and polite headers.

## 6. Testing Strategy
- [ ] 6.1 Create integration test running the pdf-two-phase pipeline with OpenAlex manifest, mocked httpx transport, and in-memory storage/cache verifying ledger flags.
- [ ] 6.2 Add similar integration coverage for another connector (e.g., Unpaywall) to ensure manifest compatibility.
- [ ] 6.3 Provide end-to-end smoke test ensuring MinerU gate blocks until `pdf_ir_ready` flips; use fake MinerU worker.
- [ ] 6.4 Update CI configuration to run new adapter + orchestration test suites and capture coverage deltas.

## 7. Documentation & Rollout
- [ ] 7.1 Update developer docs (`docs/api/*`, `docs/storage-architecture.md`, runbooks) with the manifest contract, metrics, and migration guidance.
- [ ] 7.2 Add feature flag/rollout plan to `docs/adr/` or release notes outlining phased connector enablement.
- [ ] 7.3 Communicate migration steps to dependent teams; document fallback/rollback strategy.
- [ ] 7.4 After implementation, rerun `openspec validate unify-pdf-ingestion-connectors --strict` and attach validation output to the proposal review.
