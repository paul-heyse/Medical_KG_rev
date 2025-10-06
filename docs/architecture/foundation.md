# Foundation Infrastructure Architecture

The foundation layer establishes the shared building blocks used by every other
OpenSpec change. It provides:

- **Federated Intermediate Representation (IR)** built on Pydantic models.
- **Domain overlays** for medical (FHIR-aligned), finance (XBRL) and legal
  (LegalDocML) content.
- **Configuration management** backed by Pydantic Settings with optional Vault
  integration and feature flag support.
- **Shared utilities** for HTTP access, structured logging, telemetry, span
  manipulation and identifier generation.
- **Adapter SDK** enabling declarative data source integrations with lifecycle
  hooks and testing helpers.
- **Storage abstractions** that decouple application logic from concrete
  backends while supporting async usage patterns.

Subsequent architectural layers (gateway, adapters, orchestration, GPU
services, retrieval, security and observability) build on these primitives.
