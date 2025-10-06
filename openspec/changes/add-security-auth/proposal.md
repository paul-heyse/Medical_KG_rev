# Change Proposal: Security & Authentication

## Why

Implement OAuth 2.0 authentication with client credentials flow, fine-grained scope-based authorization, multi-tenant isolation, API key management, and security best practices (rate limiting, input validation, audit logging, secrets management).

## What Changes

- OAuth 2.0 server integration (or third-party provider)
- JWT token validation with public key verification
- Scope definitions (ingest:write, kg:read, embed:write, etc.) and enforcement
- Multi-tenant context propagation (tenant_id in all queries)
- API key generation and rotation
- Rate limiting per client/endpoint with 429 responses
- Input validation and sanitization
- SQL injection prevention (parameterized queries)
- XSS prevention (output encoding)
- Secrets management (HashiCorp Vault or env vars)
- Audit logging for all mutations
- CORS configuration
- TLS/HTTPS enforcement

## Impact

- **Affected specs**: NEW capability `security-auth`
- **Affected code**:
  - `src/Medical_KG_rev/auth/` - OAuth middleware, JWT validation
  - `src/Medical_KG_rev/auth/scopes.py` - Scope definitions
  - `src/Medical_KG_rev/middleware/rate_limit.py` - Rate limiting
  - `src/Medical_KG_rev/middleware/tenant.py` - Tenant isolation
  - `src/Medical_KG_rev/security/` - Security utilities
  - `tests/auth/` - Authentication and authorization tests
