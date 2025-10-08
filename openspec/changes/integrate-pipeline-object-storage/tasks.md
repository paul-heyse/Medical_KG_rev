## Gap Analysis

- **No spec delta tasks**: the checklist never instructs the implementer to update storage-related requirements or validate the change with `openspec`; add an explicit spec section (`openspec/changes/integrate-pipeline-object-storage/tasks.md:10`).
- **Missing validation/QA guidance**: there is no task ensuring the change passes `openspec validate --strict` or documenting success criteria; add validation tasks (`openspec/changes/integrate-pipeline-object-storage/tasks.md:16`).
- **Secrets/config coverage**: configuration tasks omit credential handling (AWS keys, Redis auth) and local overrides; extend bootstrap tasks to cover secrets and environment scaffolding (`openspec/changes/integrate-pipeline-object-storage/tasks.md:28`).
- **Failure-mode testing**: testing checklist lacks negative cases (e.g., S3 outage, cache miss eviction); include resilience test tasks (`openspec/changes/integrate-pipeline-object-storage/tasks.md:66`).
- **Cross-service consumers**: MinerU update tasks do not mention chunking/embedding services that rely on PDFs in later stages; add follow-up task to ensure downstream services resolve assets via storage (`openspec/changes/integrate-pipeline-object-storage/tasks.md:47`).
- **Operational runbooks**: documentation section omits monitoring/alert integration instructions; add explicit tasks for dashboards and alarms (`openspec/changes/integrate-pipeline-object-storage/tasks.md:82`).

## 0. Spec & Validation

- [ ] 0.1 Author storage/caching requirement deltas under the relevant specs (`specs/orchestration/spec.md`, `specs/ingestion/spec.md`) using ADDED/MODIFIED sections
- [ ] 0.2 Run `openspec validate integrate-pipeline-object-storage --strict` after drafting proposal/design/specs
- [ ] 0.3 Capture acceptance criteria and sign-off checklist in `tasks.md` once features are implemented

## 1. Configuration & Bootstrap

- [ ] 1.1 Define `ObjectStorageSettings` and `RedisCacheSettings` dataclasses in `config/settings.py` with fields for endpoint, region, bucket, access key, secret key, session token, TLS mode, redis url, password, db index, and TLS cert path
- [ ] 1.2 Wire new settings into `ApplicationSettings` (and `get_settings()`) so services can access them; add defaults that point to local MinIO (`http://minio:9000`, bucket `medical-kg-pdf`) and local Redis (`redis://redis:6379/0`)
- [ ] 1.3 Document exact environment variables for each setting (`OBJECT_STORAGE__BUCKET`, `OBJECT_STORAGE__ACCESS_KEY_ID`, `OBJECT_STORAGE__SECRET_ACCESS_KEY`, `OBJECT_STORAGE__ENDPOINT_URL`, `OBJECT_STORAGE__REGION`, `OBJECT_STORAGE__USE_TLS`, `OBJECT_STORAGE__SESSION_TOKEN`, `REDIS_CACHE__URL`, `REDIS_CACHE__PASSWORD`, `REDIS_CACHE__USE_TLS`, etc.) in both `.env.example` and deployment manifests
- [ ] 1.4 Provide secrets guidance: instructions for sourcing AWS credentials from Vault/Kubernetes secrets, required IAM permissions (GetObject/PutObject/DeleteObject/ListBucket), and Redis auth/TLS certificates; include references in `DEVOPS.md` or relevant ops guide
- [ ] 1.5 Instantiate shared `ObjectStore` and `CacheBackend` during orchestrator/gateway bootstrap with environment override detection (local MinIO fallback when credentials absent) and surface configuration errors with actionable messages
- [ ] 1.6 Provide helper factories (`PdfStorageClient`, `DocumentStorageClient`) exposed via pluggy resources, enforcing key prefix strategy (`pdf/{tenant}/{document_id}/{checksum}.pdf`); include checksum hashing algorithm selection and ability to override prefixes via config
- [ ] 1.7 Update deployment manifests (Docker Compose, Helm charts, CI secrets) with concrete examples: MinIO container credentials (`MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`), S3 credentials via secret references, Redis password/TLS settings, and instructions for rotating secrets
- [ ] 1.8 Add configuration validation tests to ensure missing/invalid settings produce descriptive errors and defaults resolve to functional local resources

## 2. Pluggy Resource Wiring

- [ ] 2.1 Extend `StagePluginResources` to include typed attributes (e.g., `storage`, `cache`, `settings`) instead of opaque extras; update constructor signatures and add type annotations/tests to prevent regressions
- [ ] 2.2 Update `create_stage_plugin_manager` / `build_stage_factory` to accept storage/cache instances, store them in `StagePluginContext`, and ensure legacy callers get migration shims with deprecation warnings
- [ ] 2.3 Publish resource key contract: update pluggy docs and add inline docstrings explaining how third-party plugins retrieve `context.require("storage")` / `"cache"`; include error messages when resources are missing
- [ ] 2.4 Audit all stage factory entry points (gateway bootstrap, `ChunkingService`, embedding service, tests) to inject the new resources and adjust existing fixtures/mocks to supply fake storage/cache implementations

## 3. Durable PDF Download Stage

- [ ] 3.1 Replace `_PdfDownloadStage` with a storage-aware implementation: stream HTTP downloads with retry/backoff, compute SHA-256 while streaming, enforce max size/content-type filters, and upload to S3 via the shared `ObjectStore`
- [ ] 3.2 Persist metadata (S3 key, ETag/checksum, content length, content-type, upload timestamp) to Redis with configurable TTL; design cache keys to include tenant/document/checksum for idempotency
- [ ] 3.3 Populate `PdfAsset` objects with storage URIs, checksum, cached metadata, and update ledger entries (`set_pdf_downloaded`, `ledger_reference`) to reference the stored asset
- [ ] 3.4 Emit structured logs and Prometheus metrics for download duration, S3 upload latency, cache hits/misses, and error categories; ensure alerts can be built from these series

## 4. MinerU Pipeline Updates

- [ ] 4.1 Update MinerU pdf-two-phase gate handling to resolve S3 keys via Redis/cache first, fall back to ledger metadata, and download PDFs through signed URLs or IAM-authenticated clients before invoking MinerU
- [ ] 4.2 Ensure MinerU post-processing writes JSON, extracted text, and diagrams back into the same storage prefix (`pdf/{tenant}/{doc}/...`), tagging metadata with references to the original PDF asset
- [ ] 4.3 Implement robust error handling: detect missing/stale cache entries, rebuild metadata from S3 when necessary, surface warnings in logs/metrics, and keep the job in a retryable state instead of failing silently
- [ ] 4.4 Verify downstream chunking/embedding services consume PDFs exclusively from storage helpers—update service wiring/tests to prove no component performs ad-hoc HTTP fetches once assets are persisted

## 5. Adapter Enhancements

- [ ] 5.1 Provide mixins/helpers (e.g., `PdfUploadMixin`) that adapters can invoke to push fetched bytes directly to storage and return pre-populated metadata for the download stage to reuse
- [ ] 5.2 Update OpenAlex/Unpaywall/CORE/PMC adapters to emit metadata compatible with the new stage (pdf URLs, hashes when available) and extend unit tests with storage mocks verifying the integration points
- [ ] 5.3 Document adapter configuration flags (e.g., eager upload vs deferred download) and add example YAML/env overrides illustrating how to enable/disable direct uploads per adapter

## 6. Testing

- [ ] 6.1 Introduce unit tests using moto/MinIO and redis-mock to cover storage client creation, upload/download helpers, cache writes, and recovery of metadata
- [ ] 6.2 Build integration tests that execute the pdf-two-phase pipeline end-to-end with simulated adapters, verifying PDFs land in S3, cache entries exist, and MinerU consumes stored assets
- [ ] 6.3 Validate retry/idempotency scenarios: repeated downloads should reuse cache, checksum mismatches should trigger re-upload, and concurrent jobs should avoid duplicate uploads
- [ ] 6.4 Simulate failure scenarios (S3 downtime, Redis eviction, credential errors) and assert the system falls back to safe behaviours with actionable error messages and metrics
- [ ] 6.5 Add load/performance smoke tests measuring upload throughput and cache latency under representative workloads; document thresholds and tune defaults if limits are exceeded

## 7. Documentation & Tooling

- [ ] 7.1 Document storage/caching architecture, configuration hierarchies, and key lifecycles in `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`
- [ ] 7.2 Update docker-compose/dev tooling to spin up MinIO + Redis with seeded buckets/users, provide helper scripts for credential setup, and ensure developers can run integration tests locally
- [ ] 7.3 Produce operational runbooks covering credential rotation, bucket lifecycle policies, Redis maintenance, and backup/restore procedures
- [ ] 7.4 Publish alerting/observability setup: Prometheus metrics catalogue, Grafana dashboard templates, and alert rules for failed uploads, cache misses, and credential expirations
- [ ] 7.5 Update README/onboarding materials with step-by-step instructions for configuring S3/Redis locally, in CI, and in production (including IAM policy examples and security considerations)
