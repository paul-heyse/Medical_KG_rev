# Tenant Isolation Penetration Test Report

**Date:** 2024-07-18  
**Scope:** Gateway REST, GraphQL, and gRPC embedding surfaces (multi-tenancy enforcement).

## Objectives
- Validate that tenant metadata is enforced across REST, GraphQL, and gRPC embedding endpoints.
- Ensure namespace access control prevents unauthorized namespace usage.
- Confirm observability alerts when cross-tenant attempts occur.

## Methodology
1. **Reconnaissance** – Enumerated namespaces through authorized tenant tokens to identify access surfaces.
2. **REST Attack Simulation** – Used `scripts/audit_tenant_isolation.py` to issue cross-tenant requests against `/v1/embed` with forged tenant IDs.
3. **GraphQL Mutation Fuzzing** – Leveraged `tests/gateway/test_graphql_embedding.py::test_cross_tenant_denied` as baseline, replayed with varying namespace/scope combinations.
4. **gRPC Channel Probing** – Executed `tests/contract/test_grpc_server.py::test_cross_tenant_denied` using mismatched tenant metadata.
5. **Telemetry Review** – Queried Prometheus for `medicalkg_cross_tenant_access_attempts_total` and verified audit logs.

## Findings
- All cross-tenant requests returned `403` with `Tenant not authorized for namespace` responses.
- Namespace registry correctly denied tenants outside `allowed_tenants` list.
- Middleware injected `validated_tenant_id` and prevented spoofed JWT tenant IDs.
- Metrics counter incremented for each blocked attempt, enabling alerting.
- No data exfiltration vectors observed via request replay or parameter fuzzing.

## Remediations
- None required. Existing safeguards met penetration test criteria.
- Scheduled quarterly re-test aligned with security calendar.

## Evidence Artifacts
- REST denial logs: `gateway-embedding-service-2024-07-18.log` (attached in internal ticket).
- Prometheus snapshot: `prom://medicalkg_cross_tenant_access_attempts_total?time=2024-07-18T02:10Z`.
- Test execution trace stored in CI artifact `tenant-isolation-penetration.zip`.

## Sign-off
- **Security Lead:** Ada McKenzie (`ada.mckenzie@medicalkg.example`)
- **Reviewer:** Ops On-call (Week 29)

> Penetration testing complete. Tenant isolation controls verified.
