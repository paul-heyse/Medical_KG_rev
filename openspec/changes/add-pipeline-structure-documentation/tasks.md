## 1. Discovery & Audit
- [ ] 1.1 Inventory every pipeline-related file touched in recent refactors and capture duplicates/alternatives that must be reconciled.
- [ ] 1.2 Produce side-by-side diffs for `gateway/services.py`, coordinator modules, and chunking/embedding services that show both competing implementations for stakeholder review.
- [ ] 1.3 Document which implementation is authoritative for each duplication in an appendix table (include reasoning and dependencies).
- [ ] 1.4 Confirm no additional pipeline modules exist outside the inventoried list (search for "chunk"/"embedding" orchestrations across repo).

## 2. Documentation Standards
- [ ] 2.1 Finalize docstring format (Google-style) and share quick-reference guide with examples for modules, classes, methods, and functions.
- [ ] 2.2 Add module-level docstrings summarizing responsibilities and collaboration across all targeted pipeline modules.
- [ ] 2.3 Ensure every class, dataclass, protocol, and function in scope has a docstring that explains **what** it does and **why** it exists.
- [ ] 2.4 Insert section headers (e.g., `# --- Chunking Commands ---`) at the top of each logical grouping and document ordering rules within each file.

## 3. Structural Refactor
- [ ] 3.1 Reorder gateway coordinator modules so chunking-related constructs precede embedding constructs and shared utilities are isolated in dedicated sections.
- [ ] 3.2 Collapse duplicate coordinator logic left from previous merges and delete superseded branches after validation.
- [ ] 3.3 Restructure `gateway/services.py` into grouped sections (chunking endpoints, embedding endpoints, admin/utility) with cohesive helper placement.
- [ ] 3.4 Reorganize retrieval chunking service & command definitions to keep validation helpers adjacent to the command dataclass.
- [ ] 3.5 Group embedding policy, persister, and telemetry components into clearly labeled sections, ensuring interfaces precede implementations.
- [ ] 3.6 Reflow orchestration stage plugin files so stage definitions are ordered by pipeline phase and helper registries live at the end.
- [ ] 3.7 Mirror the production layout in associated tests with descriptive docstrings per fixture/test case.

## 4. Tooling & Enforcement
- [ ] 4.1 Enable or tighten lint rules (`ruff`, `pydocstyle`) to require module/class/function docstrings.
- [ ] 4.2 Implement a lightweight AST-based check (or extend existing tooling) that verifies required section headers appear in each module.
- [ ] 4.3 Wire the new checks into CI/pre-commit and document remediation steps for contributors.
- [ ] 4.4 Integrate docstring coverage reporting into the documentation pipeline (e.g., MkDocs summary page).

## 5. Validation & Documentation
- [ ] 5.1 Run targeted unit tests (gateway, retrieval, embedding, orchestration) after structural changes to ensure functionality remains intact.
- [ ] 5.2 Update OpenSpec change trackers previously touched to reflect the documentation & cleanup milestone.
- [ ] 5.3 Add a developer guide describing how to extend the pipeline under the new structure and docstring expectations.
- [ ] 5.4 Provide before/after examples in documentation to illustrate clarity improvements.

## 6. Legacy Decommissioning
- [ ] 6.1 Delete residual legacy pipeline helpers superseded by the coordinator and command architecture (document each removal).
- [ ] 6.2 Confirm no references to deprecated helpers remain in tests, docs, or configuration.
- [ ] 6.3 Update `LEGACY_DECOMMISSION_CHECKLIST.md` with removed items tied to this change.
