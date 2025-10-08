## Overview

This change finalizes OpenAlex integration by introducing explicit configuration, routing, and orchestration plumbing so that OpenAlex-driven ingest flows leverage the standard PDF pipeline.

## Configuration

- Define an `OpenAlexSettings` block in `config/settings.py`, leveraging `pydantic-settings`.
- Fields: `contact_email`, `user_agent`, `max_results`, `requests_per_second`, `timeout_seconds`.
- Default contact email should be unset, forcing operators to provide one (polite pool requirement) with a clear error if missing.
- Expose environment variables under `OPENALEX__*` namespace; support JSON overrides for advanced scenarios.

## Gateway & Orchestration

- Gateway routing: update `_resolve_pipeline` (or equivalent) to route `dataset=openalex` to `pdf-two-phase`.
- Provide override knob (e.g., query parameter or config) to run metadata-only flows when needed.
- Orchestration pipeline config: add OpenAlex parameter block to `pdf-two-phase.yaml`, passing polite pool headers to the ingest stage.
- Adapter request: the ingest stage should merge user-provided parameters with settings before invoking `OpenAlexAdapter`.

## Pluggy Wiring

- Stage resources should expose `settings.openalex` so plugins/adapters can access configuration without direct imports.
- If the storage integration change introduces resource extras, reuse the same mechanism.
- Adapter instantiation: ensure `OpenAlexAdapter` receives settings via constructor, not hard-coded defaults.

## Testing Strategy

- Unit tests for configuration parsing and adapter usage.
- Integration test: run ingest with httpx mock verifying polite pool headers and confirm pipeline writes to S3 via existing storage integration.
- Document manual validation steps (CLI, API request) alongside expected storage keys (`pdf/{tenant}/{doc}/...`).
