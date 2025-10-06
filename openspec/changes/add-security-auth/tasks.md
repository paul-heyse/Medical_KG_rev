# Implementation Tasks: Security & Authentication

## 1. OAuth 2.0 Setup

- [x] 1.1 Choose OAuth provider (Auth0, Keycloak, or build minimal server)
- [x] 1.2 Configure client credentials flow
- [x] 1.3 Define scopes (ingest:write, kg:read, embed:write, etc.)
- [x] 1.4 Add OAuth configuration to settings
- [x] 1.5 Write OAuth integration tests

## 2. JWT Validation

- [x] 2.1 Implement JWT middleware for FastAPI
- [x] 2.2 Add public key retrieval (JWKS endpoint)
- [x] 2.3 Validate token signature, expiration, audience
- [x] 2.4 Extract claims (sub, scopes, tenant_id)
- [x] 2.5 Add token caching to avoid repeated validation
- [x] 2.6 Write JWT validation tests

## 3. Scope-Based Authorization

- [x] 3.1 Create scope definitions module
- [x] 3.2 Implement scope checking decorators
- [x] 3.3 Add scope requirements to all endpoints
- [x] 3.4 Return 403 Forbidden for insufficient scopes
- [x] 3.5 Write authorization tests

## 4. Multi-Tenant Isolation

- [x] 4.1 Extract tenant_id from JWT
- [x] 4.2 Add tenant context to all database queries
- [x] 4.3 Implement tenant filtering in Neo4j, OpenSearch, FAISS
- [x] 4.4 Add tenant validation (reject invalid tenant access)
- [x] 4.5 Write multi-tenant tests

## 5. Rate Limiting

- [x] 5.1 Implement token bucket rate limiter
- [x] 5.2 Add per-client rate limits (by token sub)
- [x] 5.3 Add per-endpoint rate limits
- [x] 5.4 Return 429 Too Many Requests with Retry-After
- [x] 5.5 Add rate limit metrics
- [x] 5.6 Write rate limit tests

## 6. API Key Management

- [x] 6.1 Implement API key generation
- [x] 6.2 Add API key storage (hashed)
- [x] 6.3 Support API key authentication (alternative to OAuth)
- [x] 6.4 Add key rotation functionality
- [x] 6.5 Write API key tests

## 7. Security Best Practices

- [x] 7.1 Add input validation to all endpoints
- [x] 7.2 Implement output encoding for XSS prevention
- [x] 7.3 Use parameterized queries (SQL injection prevention)
- [x] 7.4 Configure CORS properly
- [x] 7.5 Enforce HTTPS/TLS in production
- [x] 7.6 Add security headers (HSTS, CSP, etc.)
- [x] 7.7 Write security tests (OWASP Top 10)

## 8. Secrets Management

- [x] 8.1 Integrate HashiCorp Vault (or use env vars)
- [x] 8.2 Store API keys securely
- [x] 8.3 Implement secret rotation
- [x] 8.4 Add secret validation on startup
- [x] 8.5 Write secrets management tests

## 9. Audit Logging

- [x] 9.1 Implement audit log for all mutations
- [x] 9.2 Log user, action, resource, timestamp
- [x] 9.3 Store audit logs separately (immutable)
- [x] 9.4 Add audit log query API
- [x] 9.5 Write audit logging tests
