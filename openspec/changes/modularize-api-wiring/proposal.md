## Why

The `create_app` function currently wires middleware, routers, docs pages, health checks, and a dozen exception handlers directly inside the factory. This creates a monolithic application setup that mixes infrastructure concerns with business logic and makes it difficult to understand how protocols integrate. The REST router repeats similar "fetch resource → if missing return presenter.error" patterns across adapter endpoints, creating code duplication and tight coupling between HTTP handling and domain services.

## What Changes

- **Extract composable setup functions**: Break down `create_app` into focused setup functions for middleware registration, exception mapping, router inclusion, and health check configuration
- **Create protocol plugin registry**: Introduce a plugin registry that enumerates enabled protocols and supplies their routers/middleware, replacing hardcoded `include_router` calls
- **Introduce shared presenter helpers**: Create reusable presenter utilities (e.g., `presenter.not_found(code, title, detail)`) to centralize error handling patterns
- **Add query object abstraction**: Create `AdapterQueries` class to centralize resource fetching and error handling logic across adapter endpoints
- **Enable declarative registration**: Allow protocols to register themselves through configuration rather than hardcoded factory logic

## Impact

- **Affected specs**: `specs/gateway/spec.md` - API wiring and presentation layer requirements
- **Affected code**:
  - `src/Medical_KG_rev/gateway/app.py` - Refactor into composable setup functions and protocol registry
  - `src/Medical_KG_rev/gateway/rest/router.py` - Introduce shared presenter helpers and query objects
  - `src/Medical_KG_rev/gateway/` - Add protocol plugin registry and setup abstractions
- **Affected systems**: Multi-protocol API gateway, application bootstrapping, error handling, protocol integration
