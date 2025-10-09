## Why
PDF-capable connectors (OpenAlex via pyalex, Unpaywall, Crossref, PMC) currently expose downloadable assets inconsistently and the orchestration pipeline still relies on stub download stages. This prevents end-to-end validation of MinerU processing and blocks storage observability across connectors.

## What Changes
- Establish a shared adapter contract and configuration surface for all PDF-capable connectors, including polite-pool settings, rate limiting, and structured asset manifests.
- Replace the stubbed PDF download/gate stages with the storage-aware implementation, complete with metrics, retries, alerting hooks, and ledger integration.
- Route PDF-capable adapters through the unified two-phase pipeline, persisting storage receipts, feature-flagged rollouts, and MinerU readiness state.
- Update configuration surfaces, developer documentation, and operational runbooks; add integration tests and observability assertions that cover multiple connectors.

## Impact
- Affected specs: orchestration/pdf-ingestion
- Affected code: `src/Medical_KG_rev/adapters/*`, `src/Medical_KG_rev/orchestration/stages/*`, `src/Medical_KG_rev/orchestration/dagster/*`, `src/Medical_KG_rev/config/settings.py`, `tests/adapters/*`, `tests/orchestration/*`
