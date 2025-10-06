# Implementation Tasks: Security & Authentication

## 1. OAuth 2.0 Setup

- [ ] 1.1 Choose OAuth provider (Auth0, Keycloak, or build minimal server)
- [ ] 1.2 Configure client credentials flow
- [ ] 1.3 Define scopes (ingest:write, kg:read, embed:write, etc.)
- [ ] 1.4 Add OAuth configuration to settings
- [ ] 1.5 Write OAuth integration tests

## 2. JWT Validation

- [ ] 2.1 Implement JWT middleware for FastAPI
- [ ] 2.2 Add public key retrieval (JWKS endpoint)
- [ ] 2.3 Validate token signature, expiration, audience
- [ ] 2.4 Extract claims (sub, scopes, tenant_id)
- [ ] 2.5 Add token caching to avoid repeated validation
- [ ] 2.6 Write JWT validation tests

## 3. Scope-Based Authorization

- [ ] 3.1 Create scope definitions module
- [ ] 3.2 Implement scope checking decorators
- [ ] 3.3 Add scope requirements to all endpoints
- [ ] 3.4 Return 403 Forbidden for insufficient scopes
- [ ] 3.5 Write authorization tests

## 4. Multi-Tenant Isolation

- [ ] 4.1 Extract tenant_id from JWT
- [ ] 4.2 Add tenant context to all database queries
- [ ] 4.3 Implement tenant filtering in Neo4j, OpenSearch, FAISS
- [ ] 4.4 Add tenant validation (reject invalid tenant access)
- [ ] 4.5 Write multi-tenant tests

## 5. Rate Limiting

- [ ] 5.1 Implement token bucket rate limiter
- [ ] 5.2 Add per-client rate limits (by token sub)
- [ ] 5.3 Add per-endpoint rate limits
- [ ] 5.4 Return 429 Too Many Requests with Retry-After
- [ ] 5.5 Add rate limit metrics
- [ ] 5.6 Write rate limit tests

## 6. API Key Management

- [ ] 6.1 Implement API key generation
- [ ] 6.2 Add API key storage (hashed)
- [ ] 6.3 Support API key authentication (alternative to OAuth)
- [ ] 6.4 Add key rotation functionality
- [ ] 6.5 Write API key tests

## 7. Security Best Practices

- [ ] 7.1 Add input validation to all endpoints
- [ ] 7.2 Implement output encoding for XSS prevention
- [ ] 7.3 Use parameterized queries (SQL injection prevention)
- [ ] 7.4 Configure CORS properly
- [ ] 7.5 Enforce HTTPS/TLS in production
- [ ] 7.6 Add security headers (HSTS, CSP, etc.)
- [ ] 7.7 Write security tests (OWASP Top 10)

## 8. Secrets Management

- [ ] 8.1 Integrate HashiCorp Vault (or use env vars)
- [ ] 8.2 Store API keys securely
- [ ] 8.3 Implement secret rotation
- [ ] 8.4 Add secret validation on startup
- [ ] 8.5 Write secrets management tests

## 9. Audit Logging

- [ ] 9.1 Implement audit log for all mutations
- [ ] 9.2 Log user, action, resource, timestamp
- [ ] 9.3 Store audit logs separately (immutable)
- [ ] 9.4 Add audit log query API
- [ ] 9.5 Write audit logging tests
