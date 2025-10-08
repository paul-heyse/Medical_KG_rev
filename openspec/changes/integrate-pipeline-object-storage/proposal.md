## Why

The ingestion and orchestration pipeline now exposes PDF metadata (OpenAlex, Unpaywall, CORE, etc.) and MinerU produces post-processed artefacts, but bytes never leave the running job. Figures are persisted because the MinerU post-processor writes them via `FigureStorageClient`, while PDFs, documents, and chunk metadata remain transient. Likewise, orchestration relies on the job ledger alone—there is no shared cache for checksum lookups, resumable downloads, or cross-stage coordination.

Without a standard storage/caching layer:

- PDF downloads cannot be shared across stages or between retry attempts.
- MinerU cannot fetch PDFs from a durable location when the gate opens.
- Chunking/embedding pipelines lack an authoritative source of document metadata.
- Adapter authors have no pluggy-friendly way to emit artefacts that land in object storage.

We need a first-class integration with S3-compatible storage (via boto3) and Redis caching so every pipeline stage can rely on reproducible infrastructure.

## What Changes

1. **Storage bootstrap**
   - Add configuration (bucket, region, credentials, prefixes) and instantiate a shared `ObjectStore` (S3/boto3) during orchestrator/gateway bootstrap.
   - Provide a factory/pluggy resource that exposes typed helpers (e.g., `PdfStorageClient`, `DocumentStorageClient`).

2. **Redis cache integration**
   - Extend configuration for Redis endpoints and construct a reusable `CacheBackend` (async redis client).
   - Inject the cache into stage plugins for metadata memoization (e.g., checksum lookups, download dedupe).

3. **Pluggy resource wiring**
   - Enrich `StagePluginResources` with `object_store` and `cache` references.
   - Update `create_stage_plugin_manager` / `build_stage_factory` so both built-in stages and custom plugins can retrieve these dependencies.

4. **PDF download stage**
   - Replace the current ledger-only `_PdfDownloadStage` implementation with a pluggy-backed stage that:
     - Streams PDFs using httpx.
     - Persists them to S3 via the injected `ObjectStore`.
     - Records metadata (S3 key, checksum, size, content type) in Redis for quick access.
     - Returns enriched `PdfAsset` objects to `PipelineState`.

5. **MinerU integration**
   - Update MinerU processors to retrieve PDFs from S3 when the gate opens, using the cached metadata to locate object storage keys.
   - Ensure MinerU post-processing continues to write artefacts to the same storage namespace.

6. **Adapter support**
   - Provide optional helper mixins so adapters that already fetch PDFs (e.g., CORE, PMC) can upload assets during parse or emit metadata that the download stage understands.

7. **Documentation & tooling**
   - Document the storage/caching architecture in `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`.
   - Provide local development defaults (e.g., MinIO/S3 mock, redis server) via Docker Compose and env vars.
   - Add scripts/tests to validate S3/Redis integration (mocked in CI).

## Impact

- **Affected specs**: Update orchestration and adapter specs to mandate durable storage integration.
- **Affected code**:
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py`, `stages.py`, `stages/plugins/` for resource injection and PDF stage rewrite.
  - `src/Medical_KG_rev/services/mineru/` to read PDFs from S3.
  - `src/Medical_KG_rev/gateway/services.py` bootstrap for storage/cache configuration.
  - `src/Medical_KG_rev/config/settings.py` for new configuration blocks.
  - Adapter modules that emit PDF metadata (OpenAlex, Unpaywall, CORE, PMC) for optional storage helpers.
  - Tests under `tests/orchestration/`, `tests/services/mineru/`, `tests/adapters/`.
- **Systems**: Introduces required S3/Redis endpoints for ingestion environments; CI will rely on mocks/emulators.

## Success Criteria

- PDF assets downloaded by adapters are uploaded to S3 with deterministic keys and recorded in Redis.
- MinerU and downstream stages retrieve PDFs solely from object storage (no ad-hoc HTTP calls).
- PipelineState `pdf_assets` include storage URIs, checksums, and cache entries for quick lookup.
- All stage plugins (including custom ones) access storage/cache via pluggy resources without manual wiring.
- Local dev and CI have documented workflows (MinIO + redis container) to exercise the pipeline end-to-end.

## Tasks

1. **Configuration & bootstrap**
   - Add S3/Redis settings to `config/settings.py` and environment templates.
   - Instantiate shared `ObjectStore`/`CacheBackend` during orchestrator & gateway construction.

2. **Pluggy resource wiring**
   - Extend `StagePluginResources` and `create_stage_plugin_manager` to include storage/cache.
   - Update `build_stage_factory` to pass the enriched resources to all plugins.

3. **PDF storage stage**
   - Implement a new pluggy stage (or refactor `_PdfDownloadStage`) that downloads, stores, and caches PDFs.
   - Record checksum/metadata in Redis and return enriched `PdfAsset` instances.
   - Update ledger interactions to reflect stored assets.

4. **MinerU pipeline updates**
   - Modify MinerU processors to fetch PDFs from S3 using cached metadata.
   - Ensure post-processing artefacts align with the same storage namespace.

5. **Adapter enhancements**
   - Add optional upload helpers/mixins for adapters that already produce PDF bytes.
   - Update OpenAlex/Unpaywall/CORE adapters and tests to confirm compatibility with the new stage.

6. **Testing**
   - Add unit tests with moto/MinIO + redis-mock to cover storage and caching logic.
   - Create integration tests for the pdf-two-phase pipeline verifying S3 writes and MinerU reads.

7. **Documentation & tooling**
   - Document architecture, configuration, and local dev workflow.
  - Update docker-compose (MinIO, redis) and developer runbooks.
