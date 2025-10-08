## Why

The new `OpenAlexAdapter` now emits full PDF metadata, but the end-to-end pipeline is still missing critical integration pieces. Without a dedicated configuration change:

- Operators cannot reliably configure polite-pool credentials (email/user-agent) per environment.
- The gateway still routes OpenAlex ingest to the default pipeline, bypassing PDF download and MinerU processing.
- There is no standard way to pass adapter-specific parameters through orchestration or to verify the PDF pipeline end-to-end.

This change ensures OpenAlex is wired into the standardized PDF ingestion flow with reproducible configuration and validation.

## What Changes

- **Configuration surface**: add typed settings/env vars for OpenAlex contact email, user-agent, and rate limits. Provide secrets guidance.
- **Pipeline routing**: update gateway/orchestrator to send OpenAlex requests through the `pdf-two-phase` topology with adapter-specific parameters.
- **Adapter parameter wiring**: ensure the orchestrator passes polite-pool credentials to the adapter via pluggy resources/config.
- **End-to-end regression**: create integration tests proving PDFs are stored in S3 (via the new object storage work) and MinerU processes them successfully.

## Impact

- **Affected specs**: orchestration spec (PDF pipeline requirements), adapter spec (OpenAlex configuration).
- **Affected code**: settings configuration, gateway pipeline routing, orchestrator stage config, adapter tests, integration test suite.
- **Dependencies**: relies on the storage/caching change (`integrate-pipeline-object-storage`) for durable PDF handling.
