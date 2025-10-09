## ADDED Requirements

### Requirement: Pipeline modules must expose structured documentation and grouping

- All modules that orchestrate chunking or embedding flows (gateway coordinators, gateway services, retrieval chunking service, embedding policy/persister/telemetry, orchestration stage plugins, and their tests) SHALL include a module-level docstring that summarizes responsibilities, upstream/downstream collaborators, key side effects, thread safety characteristics, and performance considerations.
- Every class, dataclass, protocol, and function defined in those modules SHALL declare a docstring describing what the unit does, why it exists, any critical invariants or error semantics, and include appropriate sections (Args, Returns, Raises, Examples).
- Each module SHALL organize definitions into labeled sections grouped by domain (e.g., IMPORTS, REQUEST/RESPONSE MODELS, COORDINATOR IMPLEMENTATION, ERROR TRANSLATION, EXPORTS) with clearly delineated ordering that keeps unrelated functionality separated.
- The repository linting configuration SHALL enforce the presence of module/class/function docstrings and SHALL fail validation if required section headers are missing or out of order.
- Documentation SHALL follow Google-style docstring format with consistent structure across all modules.

#### Scenario: Module documentation surfaces canonical responsibilities

- **GIVEN** the repository contains `src/Medical_KG_rev/gateway/coordinators/chunking.py`
- **WHEN** a developer opens the module
- **THEN** they see a module-level docstring explaining the coordinator's role, dependencies, side effects, and thread safety
- **AND** every class/function in the file includes a docstring with appropriate sections (Args, Returns, Raises)
- **AND** chunking-specific constructs appear in a "# COORDINATOR IMPLEMENTATION" section
- **AND** error translation logic appears in a "# ERROR TRANSLATION" section
- **AND** sections appear in the canonical order defined in section_headers.md
- **AND** linting fails with specific error messages if docstrings or section headers are removed

#### Scenario: Function docstrings include all required sections

- **GIVEN** a function `_extract_text` in `chunking.py` that accepts parameters and raises exceptions
- **WHEN** a developer views the function
- **THEN** the docstring includes an "Args:" section documenting each parameter
- **AND** the docstring includes a "Returns:" section describing the return value
- **AND** the docstring includes a "Raises:" section listing exceptions with conditions
- **AND** the docstring includes an "Example:" section showing typical usage
- **AND** ruff docstring checker passes with 0 errors

#### Scenario: Class docstrings document attributes and lifecycle

- **GIVEN** a class `ChunkingCoordinator` with instance attributes
- **WHEN** a developer views the class
- **THEN** the docstring includes an "Attributes:" section listing all public attributes
- **AND** the docstring includes an "Invariants:" section describing class invariants
- **AND** the docstring includes a "Thread Safety:" note
- **AND** the docstring includes an "Example:" section showing instantiation and usage

#### Scenario: Section headers enforce organizational structure

- **GIVEN** a coordinator module `chunking.py`
- **WHEN** the section header checker runs
- **THEN** it validates that sections appear in order: IMPORTS, REQUEST/RESPONSE MODELS, COORDINATOR IMPLEMENTATION, ERROR TRANSLATION, EXPORTS
- **AND** it validates that imports are grouped: stdlib, third-party, first-party, relative
- **AND** it validates that each section contains appropriate content
- **AND** it fails with clear error messages if sections are missing or out of order

### Requirement: Duplicate pipeline implementations must be reconciled and legacy helpers removed

- Conflicting or duplicated implementations within the affected modules SHALL be audited and reduced to a single canonical code path documented in the proposal audit.md.
- The audit SHALL document each duplicate with: file path, line numbers, description of both implementations, criteria for choosing canonical version, and rationale.
- Duplicate code blocks (imports, exception handling, helper methods) SHALL be identified and removed, keeping only the canonical implementation.
- Legacy helpers superseded by the coordinator + command architecture SHALL be deleted, and dependent references SHALL be updated or removed.
- The project documentation (including OpenSpec tasks and `LEGACY_DECOMMISSION_CHECKLIST.md`) SHALL record the removed components with: name, location, superseded by, last used date, removal date, and rationale for deletion.

#### Scenario: Duplicate imports are identified and resolved

- **GIVEN** `src/Medical_KG_rev/gateway/coordinators/chunking.py` contains duplicate imports on lines 18 and 21
- **WHEN** the audit is performed
- **THEN** the audit.md documents both import statements with exact line numbers
- **AND** the audit specifies which import to keep and which to delete
- **AND** the refactoring removes the duplicate import
- **AND** the imports are organized in groups: stdlib, third-party, first-party, relative

#### Scenario: Duplicate code blocks are identified and canonical version is selected

- **GIVEN** `chunking.py` contains two implementations of exception handling (lines 95-119 and 120-210)
- **WHEN** the audit is performed
- **THEN** audit.md documents both implementations with line ranges
- **AND** audit.md specifies criteria for selection: "First block uses ChunkCommand and ChunkingErrorTranslator (newer architecture)"
- **AND** audit.md specifies which block to delete: "Second block (lines 120-210)"
- **AND** refactoring keeps the canonical implementation and deletes the duplicate
- **AND** all references are updated to use canonical implementation

#### Scenario: Legacy helpers are documented and removed

- **GIVEN** a legacy function `old_chunk_handler` in `gateway/legacy.py` that is no longer referenced
- **WHEN** the legacy decommissioning audit is performed
- **THEN** audit.md lists the legacy function with 0 references
- **AND** the function is deleted from the codebase
- **AND** `LEGACY_DECOMMISSION_CHECKLIST.md` is updated with entry:

  ```markdown
  - [x] Removed `old_chunk_handler` from `src/Medical_KG_rev/gateway/legacy.py`
    - Superseded by: ChunkingCoordinator
    - Last used in: N/A (unreferenced)
    - Removed in: add-pipeline-structure-documentation change (2024-10-08)
    - Reason: Superseded by coordinator + command architecture
  ```

### Requirement: Documentation templates and standards must be provided for consistency

- The change SHALL provide comprehensive docstring templates for: modules, classes, functions, methods, dataclasses, protocols, exception handlers, async functions, decorators, properties, and constants.
- The change SHALL provide section header standards documenting required sections and ordering for each module type: coordinators, services, policies, orchestration, tests.
- Templates SHALL include all required sections with explanations and examples for each section.
- Templates SHALL be stored in a discoverable location and referenced in developer documentation.
- Standards SHALL specify import ordering rules, class method ordering rules, and section content validation rules.

#### Scenario: Module docstring template includes all required sections

- **GIVEN** the template at `templates/module_docstring.py`
- **WHEN** a developer uses the template
- **THEN** the template includes sections for: summary, detailed explanation, key responsibilities, collaborators (upstream/downstream), side effects, thread safety, performance characteristics, and example
- **AND** each section includes explanatory comments describing what to document
- **AND** the template shows correct Google-style formatting

#### Scenario: Section header standards define canonical structure for coordinators

- **GIVEN** the standards document `section_headers.md`
- **WHEN** a developer consults it for coordinator module structure
- **THEN** it specifies exact section headers: IMPORTS, REQUEST/RESPONSE MODELS, COORDINATOR IMPLEMENTATION, ERROR TRANSLATION, EXPORTS
- **AND** it specifies section ordering rules
- **AND** it specifies content rules (e.g., IMPORTS section contains only import statements)
- **AND** it provides examples of correctly structured modules

### Requirement: Automated tooling must enforce documentation standards

- The repository SHALL include a docstring checker that fails if any module, class, or function in scope lacks a docstring.
- The repository SHALL include a section header checker that validates section presence and ordering in all pipeline modules.
- The repository SHALL include a docstring coverage checker that calculates and reports coverage percentage, failing if below 90%.
- Checkers SHALL integrate with pre-commit hooks to run automatically on modified files.
- Checkers SHALL integrate with CI to run on all pull requests affecting pipeline code.
- Checker output SHALL be clear and actionable, specifying exact file, line number, and remediation steps.

#### Scenario: Docstring checker fails on missing docstrings

- **GIVEN** a function in `chunking.py` without a docstring
- **WHEN** ruff docstring checker runs
- **THEN** it fails with error code D103 (missing docstring in public function)
- **AND** error message includes file path, line number, and function name
- **AND** developer documentation explains how to fix D103 errors

#### Scenario: Section header checker validates section ordering

- **GIVEN** a coordinator module with sections out of order (ERROR TRANSLATION before COORDINATOR IMPLEMENTATION)
- **WHEN** section header checker runs
- **THEN** it fails with error message: "Section 'ERROR TRANSLATION' appears before 'COORDINATOR IMPLEMENTATION' in chunking.py:150"
- **AND** error includes expected ordering from section_headers.md

#### Scenario: Docstring coverage checker enforces minimum coverage

- **GIVEN** pipeline modules with 85% docstring coverage
- **WHEN** coverage checker runs with --min-coverage 90
- **THEN** it fails with exit code 1
- **AND** output shows coverage per file: "chunking.py: 85% (17/20 items documented)"
- **AND** output lists items missing docstrings with file:line:name format

#### Scenario: Pre-commit hooks run documentation checks automatically

- **GIVEN** a developer modifies `chunking.py` and attempts to commit
- **WHEN** pre-commit hooks run
- **THEN** ruff docstring check runs on modified file
- **AND** section header check runs on modified file
- **AND** docstring coverage check runs on modified file
- **AND** commit is blocked if any check fails
- **AND** developer sees clear error messages explaining failures

### Requirement: Type hints must be complete and follow modern Python conventions

- All functions and methods in pipeline modules SHALL have return type annotations.
- All function parameters SHALL have type annotations except for self and cls.
- Collection types SHALL use generics from `collections.abc` (Mapping, Sequence) instead of bare dict/list.
- Optional types SHALL use union syntax `Type | None` instead of `Optional[Type]`.
- Procedures that don't return values SHALL be annotated with `-> None`.
- Type hints SHALL be validated by mypy in strict mode.

#### Scenario: Functions have complete type annotations

- **GIVEN** a function `_extract_text(self, job_id, request)` in `chunking.py`
- **WHEN** mypy strict mode runs
- **THEN** it validates `job_id: str` and `request: ChunkingRequest` annotations exist
- **AND** it validates return type annotation `-> str` exists
- **AND** it passes with no type errors

#### Scenario: Collections use generic types from collections.abc

- **GIVEN** a function that accepts a dictionary parameter
- **WHEN** type hints are reviewed
- **THEN** the parameter uses `Mapping[str, Any]` instead of `dict`
- **AND** if mutation is needed, it uses `MutableMapping[str, Any]` instead of `dict`
- **AND** mypy validates the generic type parameters

### Requirement: Inline comments must explain complex logic and design decisions

- Exception handling blocks SHALL include comments explaining: what condition triggers the exception, how it's handled, and what side effects occur (metrics, logs).
- Complex algorithms SHALL include comments explaining the approach and key steps.
- Non-obvious design decisions SHALL include comments explaining the rationale.
- Magic numbers or string constants SHALL include comments explaining their meaning.
- Comments SHALL be clear, concise, and add value beyond what the code itself conveys.

#### Scenario: Exception handlers include explanatory comments

- **GIVEN** an exception handling block catching `ProfileNotFoundError`
- **WHEN** code is reviewed
- **THEN** a comment before the try block explains: "Attempt chunking and translate any failures to coordinator errors"
- **AND** a comment in the except block explains: "Profile specified in request does not exist in config/chunking/profiles/"
- **AND** a comment explains the metric emitted: "record_chunking_failure(profile, 'ProfileNotFoundError')"

### Requirement: Developer guides must provide extension patterns and best practices

- The change SHALL produce a comprehensive pipeline extension guide covering: adding coordinators, services, orchestration stages, policies, and persisters.
- The guide SHALL include step-by-step instructions with code examples for each extension type.
- The guide SHALL document testing patterns for each component type with fixture examples.
- The guide SHALL explain error handling strategies and how to add new error translations.
- The change SHALL produce architecture decision records explaining key design choices: coordinator architecture, section headers, error translation strategy, docstring format choice.
- Guides SHALL be discoverable in the documentation site and referenced in contribution guidelines.

#### Scenario: Extension guide provides complete implementation pattern

- **GIVEN** a developer wants to add a new coordinator for validation operations
- **WHEN** they consult `docs/guides/pipeline_extension_guide.md`
- **THEN** the guide provides step-by-step instructions for:
  1. Defining request/response dataclasses
  2. Implementing coordinator class inheriting from BaseCoordinator
  3. Implementing _execute method with lifecycle integration
  4. Adding error translation logic
  5. Writing docstrings
  6. Adding unit tests
- **AND** the guide includes a complete code example showing all steps
- **AND** the example compiles and runs without errors

### Requirement: Before/after examples must demonstrate improvements

- The change SHALL provide before/after code examples showing improvements in: docstring coverage, code organization, duplicate removal, and import organization.
- Examples SHALL include metrics quantifying improvements: docstring coverage percentage, lines of code reduction, duplicate blocks removed.
- Examples SHALL be stored in a discoverable location within the change proposal directory.
- Examples SHALL include a README explaining the improvements and linking to full refactored files.

#### Scenario: Before/after example shows docstring coverage improvement

- **GIVEN** before/after examples in `examples/` directory
- **WHEN** a developer reviews the examples
- **THEN** `before_chunking_coordinator.py` shows code with no module docstring and methods without docstrings
- **AND** `after_chunking_coordinator.py` shows the same code with comprehensive docstrings
- **AND** `examples/README.md` quantifies: "Docstring coverage: 20% → 100%"
- **AND** the README links to the full refactored file

#### Scenario: Before/after example shows duplicate code removal

- **GIVEN** before/after examples showing exception handling
- **WHEN** a developer reviews the examples
- **THEN** `before_chunking_coordinator.py` shows duplicate exception handling blocks (lines 95-210)
- **AND** `after_chunking_coordinator.py` shows single canonical exception handling block
- **AND** `examples/README.md` quantifies: "Duplicate code blocks removed: 5, Lines reduced: 150 lines (42%)"

### Requirement: Test modules must follow documentation standards

- All test modules SHALL include module-level docstrings explaining what component is under test.
- All pytest fixtures SHALL include docstrings explaining what the fixture provides and typical usage.
- All test functions SHALL include docstrings following format: "Test that [component] [behavior] when [condition]."
- Test function names SHALL follow pattern: `test_<component>_<behavior>_<condition>`.
- Test modules SHALL use section headers: IMPORTS, FIXTURES, UNIT TESTS - [Component], INTEGRATION TESTS, HELPER FUNCTIONS.

#### Scenario: Test module has comprehensive documentation

- **GIVEN** `tests/gateway/test_chunking_coordinator.py`
- **WHEN** a developer opens the file
- **THEN** it includes module docstring: "Tests for ChunkingCoordinator in src/Medical_KG_rev/gateway/coordinators/chunking.py"
- **AND** fixture `chunking_coordinator` has docstring explaining what it provides
- **AND** test functions have descriptive docstrings
- **AND** test names follow pattern: `test_chunking_coordinator_raises_error_when_profile_not_found`

### Requirement: Documentation must be automatically generated and published

- The repository SHALL configure mkdocstrings to extract API documentation from docstrings.
- The documentation site SHALL include API documentation pages for: coordinators, services, embedding, and orchestration modules.
- API documentation pages SHALL use `:::` syntax to automatically include docstrings from source code.
- Documentation generation SHALL fail if docstrings are malformed or incomplete.
- Generated documentation SHALL be reviewed for formatting correctness.

#### Scenario: API documentation is automatically generated from docstrings

- **GIVEN** `mkdocs.yml` configured with mkdocstrings plugin
- **WHEN** `mkdocs build` runs
- **THEN** it generates `api/coordinators.html` with ChunkingCoordinator documentation extracted from source
- **AND** docstring sections (Args, Returns, Raises) are formatted correctly
- **AND** cross-references (`:class:`, `:func:`) are rendered as links
- **AND** examples are formatted as code blocks
