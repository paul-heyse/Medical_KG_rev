## ADDED Requirements

### Requirement: Pipeline modules must expose structured documentation and grouping
- All modules that orchestrate chunking or embedding flows (gateway coordinators, gateway services, retrieval chunking service, embedding policy/persister/telemetry, orchestration stage plugins, and their tests) SHALL include a module-level docstring that summarizes responsibilities, upstream/downstream collaborators, and key side effects.
- Every class, dataclass, protocol, and function defined in those modules SHALL declare a docstring describing what the unit does, why it exists, and any critical invariants or error semantics.
- Each module SHALL organize definitions into labeled sections grouped by domain (e.g., chunking request/response, embedding policies, shared utilities) with clearly delineated ordering that keeps unrelated functionality separated.
- The repository linting configuration SHALL enforce the presence of module/class/function docstrings and SHALL fail validation if required section headers are missing or out of order.

#### Scenario: Module documentation surfaces canonical responsibilities
Given the repository contains `src/Medical_KG_rev/gateway/coordinators/chunking.py`
When a developer opens the module
Then they see a module-level docstring explaining the coordinator's role and dependencies
And every class/function in the file includes a docstring describing what and why it exists
And chunking-specific constructs appear in a dedicated "Chunking" section preceding shared utilities or embedding constructs
And linting fails if the section headers or docstrings are removed.

### Requirement: Duplicate pipeline implementations must be reconciled and legacy helpers removed
- Conflicting or duplicated implementations within the affected modules SHALL be audited and reduced to a single canonical code path documented in the proposal appendix.
- Legacy helpers superseded by the coordinator + command architecture SHALL be deleted, and dependent references SHALL be updated or removed.
- The project documentation (including OpenSpec tasks and `LEGACY_DECOMMISSION_CHECKLIST.md`) SHALL record the removed components and the rationale for their deletion.

#### Scenario: Duplicate gateway service logic is resolved
Given `src/Medical_KG_rev/gateway/services.py` currently contains two variants of the chunking handler
When the change is applied
Then only the authoritative implementation remains with an explanatory docstring and correct section placement
And obsolete helper functions referencing the removed implementation are deleted
And the decommission checklist is updated to reflect the cleanup.
