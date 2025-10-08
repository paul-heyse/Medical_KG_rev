## Overview

This change introduces durable storage and caching primitives to the ingestion pipeline by wiring S3 (via boto3) and Redis into the pluggy infrastructure used by adapters and orchestration stages. The goal is to make PDFs, MinerU artefacts, and metadata reproducible across retries and environments while keeping the stage architecture extensible.

## Storage Architecture

### Object Store Abstraction

- Reuse `ObjectStore` interface in `storage/object_store.py`.
- Configure an `S3ObjectStore` instance at bootstrap using new settings (`bucket`, `region`, credentials).
- Expose typed helpers (`PdfStorageClient`, `DocumentStorageClient`) that normalise key structure:  
  `pdf/{tenant}/{document_id}/{checksum}.pdf`
- Ensure helpers support presigned URLs for downstream consumers.

### Redis Cache

- Instantiate `RedisCache` (async client) with configurable URL and namespace.
- Cache metadata per asset: checksum, S3 key, content length, content-type, download timestamp.
- Provide helper utility to combine ledger state + cache entries for quick gateway lookups.

## Pluggy Integration

### StagePluginResources Extension

- Add `extras` entries for `object_store` and `cache`.
- Update `create_stage_plugin_manager`/`build_stage_factory` to populate resources once and pass to all plugins.
- Document resource keys so third-party plugins can opt-in without new constructor arguments.

### PDF Download Stage

- Replace `_PdfDownloadStage` logic with a pluggy-aware builder that:
  1. Downloads via httpx with retry/timeout from stage config.
  2. Streams to S3 using the injected `ObjectStore`.
  3. Computes checksum (SHA256) while streaming.
  4. Stores metadata in Redis (`pdf:tenant:doc:checksum`).
  5. Returns `PdfAsset` objects referencing the S3 key and metadata.
- Maintain existing ledger updates (`set_pdf_downloaded`) and incorporate checksum reference.

### MinerU Flow

- MinerU gate stage retrieves S3 key/metadata from Redis; falls back to ledger metadata if cache misses.
- MinerU processor downloads the PDF from S3 (using presigned URL when IAM not available).
- Post-processing continues to place outputs under the same S3 namespace so artefacts co-locate.

## Adapter Enhancements

- Provide mixin that adapters can call to upload PDF bytes directly when they already fetch files (optional).
- For metadata-only adapters (OpenAlex, Unpaywall) the new download stage covers the heavy lifting; ensure payload fields align (`pdf_url`, `checksum` optional).

## Testing Strategy

- Use moto or MinIO in tests to emulate S3; redis-mock or embedded Redis for cache verification.
- Integration tests run the pdf-two-phase pipeline end-to-end: adapter -> download stage -> S3 -> MinerU fetch.
- Add property tests around checksum/idempotency (re-running stage should not duplicate uploads).

## Operations

- Provide docker-compose overrides bundling MinIO + Redis for local dev.
- Emit metrics for storage operations (upload duration, cache hits) and log errors with S3 keys/redacted URLs.
- Document backup/retention expectations and presigned URL TTL recommendations.
