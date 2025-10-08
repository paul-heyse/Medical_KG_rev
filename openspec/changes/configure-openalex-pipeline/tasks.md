## 0. Spec & Validation

- [ ] 0.1 Draft spec deltas (adapter + orchestration) describing required OpenAlex configuration, routing, and PDF pipeline usage
- [ ] 0.2 Run `openspec validate configure-openalex-pipeline --strict` after drafting proposal/tasks/specs
- [ ] 0.3 Capture acceptance criteria for end-to-end PDF ingestion in proposal/tasks

## 1. OpenAlex Configuration Surface

- [ ] 1.1 Add `OpenAlexSettings` to `config/settings.py` with fields for contact email, user-agent, max results/requests per second
- [ ] 1.2 Expose environment variables (`OPENALEX__EMAIL`, `OPENALEX__USER_AGENT`, etc.) in `.env.example`, Docker/Kubernetes manifests, and secrets guidance
- [ ] 1.3 Update adapter construction to pull settings via `get_settings()` and fall back to safe defaults
- [ ] 1.4 Document IAM/Vault guidance for storing polite-pool credentials and rotating them

## 2. Pipeline Routing & Parameters

- [ ] 2.1 Update `GatewayService` routing to send OpenAlex datasets to `pdf-two-phase` topology, with fallback logic for explicit overrides
- [ ] 2.2 Modify orchestration configuration (`config/orchestration/pipelines/*.yaml`) to include OpenAlex-specific stage parameters (search limits, polite pool email/user-agent)
- [ ] 2.3 Ensure adapter plugin receives settings via `AdapterRequest.parameters` or pluggy resources without hardcoding
- [ ] 2.4 Add rate-limit and polite-pool metadata to job submissions for observability

## 3. Pluggy Integration & Adapter Wiring

- [ ] 3.1 Inject OpenAlex settings into stage/plugin resources so the adapter and download stage can use them
- [ ] 3.2 Update adapter unit tests to assert settings are applied (custom email/user-agent)
- [ ] 3.3 Provide examples for other adapters to follow (documentation or mixin usage)

## 4. End-to-End Regression Testing

- [ ] 4.1 Add integration test that runs OpenAlex ingest through the PDF pipeline with mock storage/cache, asserting PDF stored in S3 and MinerU completes
- [ ] 4.2 Add smoke test verifying polite-pool headers (email + user-agent) are set on outgoing requests
- [ ] 4.3 Update documentation with validation steps (manual CLI invocation, expected storage keys, monitoring checks)
