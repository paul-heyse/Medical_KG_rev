## Change Proposal: add-pipeline-structure-documentation

### Why
- The pipeline surface (gateway coordinators, chunking services, embedding policy/persister, orchestration stage plugins) currently mixes unrelated responsibilities inside the same modules without any narrative or docstrings. This makes it very difficult for downstream teams to understand the execution flow or to review the unresolved merge blocks that were intentionally left duplicated in prior passes.
- Several files now contain duplicate logic blocks because reviewers could not determine which variant to keep. Without a deterministic clean-up plan we risk shipping both versions, introducing dead code paths, and breaking runtime expectations when future contributors touch the area.
- The lack of consistent grouping (chunking vs. embedding vs. shared infrastructure) and the absence of structured comments prevents us from enforcing coding conventions at scale, slows onboarding, and makes it impossible to auto-generate documentation from the code base.

### Goals
1. Produce a deterministic clean-up plan for every conflicted file introduced in the recent pipeline refactors (gateway services/coordinators, chunking error translator, orchestration stage plugins, embedding policy/persister/telemetry, retrieval chunking command/service, and related tests).
2. Introduce comprehensive, purpose-driven documentation comments for **every** class, dataclass, protocol, function, and module that participates in chunking or embedding orchestration so reviewers can immediately understand intent and side-effects.
3. Restructure each module so domain responsibilities are grouped together (e.g., chunking request/response helpers collocated; embedding policy classes adjacent) with explicit section headers and ordering rules that can be codified into static analysis checks.
4. Establish validation tooling (pre-commit friendly) that fails CI if required docstrings or section labels are missing, ensuring the structure remains enforceable after initial adoption.
5. Update OpenSpec tasks to retire legacy/deprecated pipeline entry points that are superseded by the coordinator + command architecture, preventing future drift.

### Non-Goals
- Introducing new runtime behaviors for chunking or embedding execution beyond clarifying existing intent.
- Rewriting the high-level orchestration architecture (Dagster job graph, lifecycle manager semantics). The focus is clarity, documentation, and structural hygiene.

### Stakeholders / Reviewers
- Pipeline platform team (owners of chunking & embedding services)
- Gateway maintainers (REST and SSE orchestration)
- Developer experience team (documentation automation)

### Proposed Implementation Plan

#### Phase 1 – Audit & Source of Truth Alignment
- Enumerate all files touched by the recent chunking/embedding/coordinator work and catalogue duplicated blocks or mutually exclusive implementations that were left side-by-side.
- Decide for each duplication which branch is canonical using runtime dependencies, existing unit tests, and orchestrator wiring as reference. Document the decision matrix inside the proposal appendix.
- Capture current ordering of classes/functions per module to inform the re-grouping rules.

#### Phase 2 – Documentation Framework & Standards
- Adopt a consistent docstring format (Google-style for functions/methods, short imperative summaries for classes) and add module-level docstrings describing intent and collaboration graph.
- Add lightweight decorators or helper utilities (e.g., `structlog` context helpers) only where necessary to surface runtime metadata in the documentation.
- Introduce `pydantic.dataclasses` or `typing.Annotated` metadata where it reduces boilerplate in describing payloads, leveraging libraries already in the stack instead of hand-rolled validators.

#### Phase 3 – Structural Refactor per Domain
- **Gateway Coordinators**: Split chunking vs. embedding coordinators into clearly labeled sections, ensure error translation helpers live in dedicated modules, and collapse lifecycle helper functions near their usage.
- **Gateway Services Layer**: Re-order synchronous handler methods so chunking entry points precede embedding, annotate dependencies with docstrings, and eliminate duplicate implementations retained from conflict resolution.
- **Retrieval Chunking Service**: Co-locate the `ChunkCommand`, parsing helpers, and service execution methods; ensure validation utilities sit directly beside command definitions.
- **Embedding Policy/Persister/Telemetry**: Group policy definitions, rules, and caching strategies together; place persister interfaces and concrete implementations in logical sequences; annotate telemetry hooks with context on emitted metrics.
- **Orchestration Stage Plugins & Contracts**: Re-group stage definitions by pipeline phase and ensure plugin registration helpers carry docstrings that map to runtime hooks.
- **Tests**: Mirror production structure in tests by grouping fixtures and helper factories under matching section headers, ensuring every test case states the behavior under validation.

#### Phase 4 – Tooling & Enforcement
- Extend linting configuration to require module, class, and function docstrings (e.g., enable `pydocstyle`/`ruff` rules already available in the repo) and create custom checks for ordered section headers using a simple AST visitor script.
- Add continuous documentation extraction via `mkdocs` or similar to publish pipeline API docs from docstrings, ensuring comments remain accurate.

#### Phase 5 – Documentation & Change Management
- Update OpenSpec task lists tied to the affected changes to note the structural documentation milestone and mark legacy cleanup items as complete once resolved.
- Provide migration guidance describing how to add new chunking/embedding functionality while adhering to the new layout and documentation standards.

### Risks & Mitigations
- **Risk**: Enforcing docstrings could become noisy. *Mitigation*: Provide templates/snippets and pre-populate docstrings during refactor; configure lint to allow TODO markers with expiry dates.
- **Risk**: Removing duplicate code might accidentally drop untested behavior. *Mitigation*: Use targeted regression tests and cross-reference with integration flows before deletion.
- **Risk**: Section reordering could break import cycles. *Mitigation*: Introduce explicit re-export modules (`__all__`) and keep initialization-free modules to avoid side effects.

### Open Questions
- Should documentation generation feed into existing MkDocs site or remain as internal developer docs?
- Do we want to standardize on Google vs. NumPy docstring format across the repository? Proposal assumes Google-style unless stakeholders prefer otherwise.

### Appendix: Initial File Inventory (non-exhaustive)
- `src/Medical_KG_rev/gateway/chunking_errors.py`
- `src/Medical_KG_rev/gateway/coordinators/{base,chunking,embedding}.py`
- `src/Medical_KG_rev/gateway/coordinators/job_lifecycle.py`
- `src/Medical_KG_rev/gateway/services.py`
- `src/Medical_KG_rev/services/retrieval/chunking.py`
- `src/Medical_KG_rev/services/embedding/{policy,persister,telemetry}.py`
- `src/Medical_KG_rev/orchestration/dagster/{runtime,stages}.py`
- `src/Medical_KG_rev/orchestration/stages/{contracts,plugins}.py`
- Corresponding test modules under `tests/gateway`, `tests/services`, `tests/orchestration`
