# Pluggy Specification Clarification - Summary

## Changes Made

I've updated the `add-adapter-plugin-framework` OpenSpec change proposal to make **Pluggy** explicit and clear throughout all documentation files.

## Files Updated

### 1. `proposal.md`

- ✅ Added explicit **Pluggy** library reference with version (pluggy>=1.3.0)
- ✅ Added link to Pluggy documentation (<https://pluggy.readthedocs.io/>)
- ✅ Added note that Pluggy is "the same plugin framework used by pytest"
- ✅ Added explicit **Tenacity** library reference (tenacity>=8.2.0)
- ✅ Added explicit **pydantic-settings** library reference (pydantic-settings>=2.0.0)

### 2. `specs/biomedical-adapters/spec.md`

- ✅ Updated "Adapter Registration" requirement to mention **Pluggy plugin framework**
- ✅ Added references to `HookspecMarker` and `HookimplMarker`
- ✅ Updated scenario to specify "Pluggy plugin manager" explicitly
- ✅ Added "Pluggy's hook implementation mechanism" in registration flow

### 3. `specs/ingestion-orchestration/spec.md`

- ✅ Updated "Plugin Manager Integration" requirement to specify **Pluggy-based** plugin manager
- ✅ Added reference to Pluggy's `PluginManager` class
- ✅ Updated scenarios to explicitly mention Pluggy

### 4. `design.md`

- ✅ Expanded "Decision 1" to include full Pluggy details:
  - Version requirement: 1.3.0+
  - Industry adoption (pytest, tox, devpi)
  - Hook specifications and implementation markers
  - Type safety features
  - Battle-tested status
- ✅ Added SQLAlchemy to "Alternatives Considered" with clear explanation of why it's wrong
- ✅ Enhanced code examples with detailed comments about Pluggy usage
- ✅ Added documentation links for all libraries

### 5. `tasks.md`

- ✅ Task 1.1: Explicit "Install and configure **Pluggy** dependency (pluggy>=1.3.0)"
- ✅ Task 1.2: "Define `AdapterHookSpec` with Pluggy hook specifications using `@hookspec`"
- ✅ Task 1.3: "using Pluggy's `PluginManager` class"
- ✅ Task 1.7: "via Pluggy hook calls"
- ✅ Task 1.8: "for Pluggy plugin manager"
- ✅ Section 3: Added explicit **Tenacity** references (tenacity>=8.2.0)
- ✅ Section 4: Added explicit **pydantic-settings** references (pydantic-settings>=2.0.0)
- ✅ Section 5: Added "with Pluggy `@hookimpl` decorators"
- ✅ Section 8: Updated all orchestration tasks to mention Pluggy
- ✅ Section 10: Updated testing tasks to mention Pluggy

### 6. `README.md`

- ✅ Added library versions and documentation links for all three core libraries:
  - Pluggy 1.3.0+ with link
  - Tenacity 8.2.0+ with link
  - pydantic-settings 2.0.0+ with link
- ✅ Added note about Pluggy being used by pytest

## Key Specifications Now Clear

### Pluggy Usage

- **What**: Pluggy is the plugin framework for adapter discovery and lifecycle management
- **Version**: 1.3.0 or higher
- **Why**: Industry-standard used by pytest, type-safe, battle-tested
- **How**: Hook specifications (`@hookspec`), hook implementations (`@hookimpl`), `PluginManager` class
- **Not**: SQLAlchemy (which is an ORM for databases, not a plugin system)

### Tenacity Usage

- **What**: Retry and resilience library for handling transient failures
- **Version**: 8.2.0 or higher
- **Why**: Declarative retry policies, built-in backoff strategies
- **How**: Decorators like `@retry`, configurable wait strategies

### Pydantic-Settings Usage

- **What**: Type-safe configuration management
- **Version**: 2.0.0 or higher
- **Why**: 12-factor app compliance, automatic validation, hot-reload support
- **How**: Environment variables with `MK_ADAPTER_` prefix

## Validation Status

```bash
$ openspec validate add-adapter-plugin-framework --strict
Change 'add-adapter-plugin-framework' is valid
```

✅ All requirements properly formatted
✅ All scenarios use correct syntax
✅ All library specifications now explicit
✅ No validation errors or warnings

## Implementation Clarity

Developers implementing this proposal now have:

1. **Clear library choices**: Pluggy, Tenacity, pydantic-settings with versions
2. **Documentation links**: Direct links to official documentation for each library
3. **Rationale**: Detailed explanation of why each library was chosen
4. **Anti-patterns**: Clear explanation of why SQLAlchemy is NOT appropriate
5. **Code examples**: Commented code showing Pluggy hook specifications and implementations
6. **Industry validation**: References to pytest and other major projects using Pluggy

## Next Steps

The proposal is ready for:

1. Technical review by team
2. Approval from architecture leads
3. Implementation following the explicit specifications
4. No ambiguity about which libraries to use or how to use them
