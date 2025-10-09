## ADDED Requirements

### Requirement: Repository-wide documentation standards must be established

- All Python modules in the Medical_KG_rev repository SHALL include comprehensive module-level docstrings that summarize responsibilities, collaborators, side effects, thread safety characteristics, and performance considerations.
- Every class, dataclass, protocol, function, and method SHALL have docstrings describing purpose, behavior, parameters, return values, exceptions, and usage examples.
- All modules SHALL organize code into labeled sections with consistent ordering rules appropriate for each module type (gateway, service, adapter, orchestration, kg, storage, validation, utility, test).
- Documentation SHALL follow Google-style docstring format with consistent structure across all modules.
- The repository SHALL achieve 100% docstring coverage across all 360+ Python files.

#### Scenario: Gateway module has comprehensive documentation

- **GIVEN** the repository contains `src/Medical_KG_rev/gateway/coordinators/chunking.py`
- **WHEN** a developer opens the module
- **THEN** they see a module-level docstring explaining the coordinator's role, dependencies, side effects, and thread safety
- **AND** every class/function in the file includes a docstring with appropriate sections (Args, Returns, Raises, Examples)
- **AND** code is organized into sections: IMPORTS, REQUEST/RESPONSE MODELS, COORDINATOR IMPLEMENTATION, ERROR TRANSLATION, EXPORTS
- **AND** docstring coverage checker reports 100% coverage for this file

#### Scenario: Service module follows service-specific documentation standards

- **GIVEN** the repository contains `src/Medical_KG_rev/services/embedding/policy.py`
- **WHEN** a developer views the module
- **THEN** the module docstring explains the policy system's role in namespace access control
- **AND** all policy classes have docstrings documenting their evaluation logic
- **AND** code is organized into sections: IMPORTS, DATA MODELS, INTERFACES, IMPLEMENTATIONS, FACTORY FUNCTIONS, EXPORTS
- **AND** all methods include Args, Returns, Raises sections with examples

#### Scenario: Adapter module documents external API integration

- **GIVEN** the repository contains `src/Medical_KG_rev/adapters/biomedical.py`
- **WHEN** a developer examines the module
- **THEN** the module docstring explains the adapter's role in biomedical data integration
- **AND** all adapter classes document their external API dependencies and rate limits
- **AND** code is organized into sections: IMPORTS, DATA MODELS, ADAPTER IMPLEMENTATION, ERROR HANDLING, FACTORY FUNCTIONS, EXPORTS
- **AND** all methods document their external API calls and error handling

#### Scenario: Test module documents testing approach

- **GIVEN** the repository contains `tests/gateway/test_chunking_coordinator.py`
- **WHEN** a developer reviews the test module
- **THEN** the module docstring explains what component is under test
- **AND** all test functions have docstrings following format: "Test that [component] [behavior] when [condition]"
- **AND** code is organized into sections: IMPORTS, FIXTURES, UNIT TESTS - [Component], INTEGRATION TESTS, HELPER FUNCTIONS
- **AND** test function names follow pattern: `test_<component>_<behavior>_<condition>`

### Requirement: Repository-wide code organization standards must be enforced

- All modules SHALL use consistent section headers appropriate for their module type with standardized ordering rules.
- Import statements SHALL be organized into groups: stdlib, third-party, first-party, relative, with alphabetical sorting within each group.
- Class methods SHALL be ordered: `__init__` first, public methods alphabetically, private methods alphabetically, static/class methods last.
- Functions SHALL be ordered: public functions first alphabetically, private functions last alphabetically.
- Section headers SHALL be validated by automated tools that fail CI if violations are found.

#### Scenario: Gateway coordinator module has proper section organization

- **GIVEN** a gateway coordinator module `chunking.py`
- **WHEN** the section header checker runs
- **THEN** it validates sections appear in order: IMPORTS, REQUEST/RESPONSE MODELS, COORDINATOR IMPLEMENTATION, ERROR TRANSLATION, EXPORTS
- **AND** it validates imports are grouped: stdlib, third-party, first-party, relative
- **AND** it validates class methods follow ordering rules
- **AND** it passes with 0 violations

#### Scenario: Service module has service-specific section organization

- **GIVEN** a service module `embedding/policy.py`
- **WHEN** the section header checker runs
- **THEN** it validates sections appear in order: IMPORTS, DATA MODELS, INTERFACES, IMPLEMENTATIONS, FACTORY FUNCTIONS, EXPORTS
- **AND** it validates each section contains appropriate content
- **AND** it passes with 0 violations

#### Scenario: Adapter module has adapter-specific section organization

- **GIVEN** an adapter module `biomedical.py`
- **WHEN** the section header checker runs
- **THEN** it validates sections appear in order: IMPORTS, DATA MODELS, ADAPTER IMPLEMENTATION, ERROR HANDLING, FACTORY FUNCTIONS, EXPORTS
- **AND** it validates adapter-specific content in appropriate sections
- **AND** it passes with 0 violations

### Requirement: Duplicate code must be eliminated across the entire repository

- Duplicate implementations within and across modules SHALL be identified using AST analysis and pattern matching.
- Each duplicate SHALL be documented with: file path, line numbers, description of both implementations, criteria for choosing canonical version, and rationale.
- Duplicate code blocks (imports, functions, classes, exception handling) SHALL be removed, keeping only the canonical implementation.
- Legacy helpers superseded by modern patterns SHALL be deleted, and dependent references SHALL be updated or removed.
- The project documentation SHALL record all removed components with: name, location, superseded by, last used date, removal date, and rationale.

#### Scenario: Duplicate functions are identified and resolved

- **GIVEN** the repository contains duplicate function implementations across multiple modules
- **WHEN** the duplicate code detector runs
- **THEN** it identifies all duplicate functions with exact file paths and line numbers
- **AND** it generates a report with criteria for selecting canonical implementations
- **AND** refactoring removes duplicates while preserving functionality
- **AND** all references are updated to use canonical implementations

#### Scenario: Duplicate imports are consolidated

- **GIVEN** multiple modules import the same symbols from different locations
- **WHEN** the import analyzer runs
- **THEN** it identifies duplicate imports with exact locations
- **AND** it selects canonical import sources based on established patterns
- **AND** refactoring consolidates imports to canonical sources
- **AND** all modules use consistent import organization

#### Scenario: Legacy code is identified and removed

- **GIVEN** the repository contains legacy functions superseded by modern patterns
- **WHEN** the legacy code detector runs
- **THEN** it identifies all legacy code with reference counts
- **AND** it documents what each legacy item is superseded by
- **AND** refactoring removes unreferenced legacy code
- **AND** `LEGACY_DECOMMISSION_CHECKLIST.md` is updated with removal details

### Requirement: Type hints must be modernized across the entire repository

- All functions and methods SHALL have complete type annotations including return types.
- All function parameters SHALL have type annotations except for self and cls.
- Collection types SHALL use generics from `collections.abc` (Mapping, Sequence) instead of bare dict/list.
- Optional types SHALL use union syntax `Type | None` instead of `Optional[Type]`.
- Procedures that don't return values SHALL be annotated with `-> None`.
- Type hints SHALL be validated by mypy in strict mode across the entire repository.

#### Scenario: All functions have complete type annotations

- **GIVEN** the repository contains functions with missing type annotations
- **WHEN** mypy strict mode runs on the entire repository
- **THEN** it validates all functions have parameter and return type annotations
- **AND** it validates all collections use generic types from collections.abc
- **AND** it validates all optional types use union syntax
- **AND** it passes with 0 type errors

#### Scenario: Collections use modern generic types

- **GIVEN** the repository contains functions using bare dict/list types
- **WHEN** the type hint checker runs
- **THEN** it identifies all uses of bare dict/list
- **AND** it validates replacement with Mapping/Sequence from collections.abc
- **AND** it validates generic type parameters are specified
- **AND** mypy validates the modernized type hints

#### Scenario: Optional types use union syntax

- **GIVEN** the repository contains functions using Optional[Type] syntax
- **WHEN** the type hint checker runs
- **THEN** it identifies all uses of Optional[Type]
- **AND** it validates replacement with Type | None syntax
- **AND** it validates all union types use modern syntax
- **AND** mypy validates the modernized type hints

### Requirement: Automated enforcement tools must validate repository-wide standards

- The repository SHALL include a docstring checker that fails if any module, class, or function lacks a docstring.
- The repository SHALL include a section header checker that validates section presence and ordering in all modules.
- The repository SHALL include a docstring coverage checker that calculates and reports coverage percentage, failing if below 100%.
- The repository SHALL include a duplicate code detector that identifies and reports duplicate implementations.
- The repository SHALL include a type hint checker that validates modern Python type hint usage.
- All checkers SHALL integrate with pre-commit hooks and CI to run automatically on all changes.

#### Scenario: Docstring checker enforces 100% coverage

- **GIVEN** the repository contains modules with missing docstrings
- **WHEN** the docstring coverage checker runs
- **THEN** it calculates coverage percentage for each file and the entire repository
- **AND** it fails with exit code 1 if coverage is below 100%
- **AND** it reports specific items missing docstrings with file:line:name format
- **AND** pre-commit hooks block commits until coverage reaches 100%

#### Scenario: Section header checker validates all module types

- **GIVEN** the repository contains modules of different types (gateway, service, adapter, etc.)
- **WHEN** the section header checker runs
- **THEN** it validates section presence and ordering for each module type
- **AND** it validates section content is appropriate for each section
- **AND** it fails with clear error messages if violations are found
- **AND** CI blocks merges until all violations are resolved

#### Scenario: Duplicate code detector identifies all duplicates

- **GIVEN** the repository contains duplicate code across modules
- **WHEN** the duplicate code detector runs
- **THEN** it identifies all duplicate functions, classes, and imports
- **AND** it generates detailed reports with file paths and line numbers
- **AND** it provides criteria for selecting canonical implementations
- **AND** CI blocks merges until duplicates are resolved

#### Scenario: Type hint checker enforces modern syntax

- **GIVEN** the repository contains functions with outdated type hint syntax
- **WHEN** the type hint checker runs
- **THEN** it identifies all uses of deprecated Optional[Type] syntax
- **AND** it identifies all uses of bare dict/list types
- **AND** it validates replacement with modern union and generic syntax
- **AND** CI blocks merges until type hints are modernized

### Requirement: Complete API documentation must be generated from docstrings

- The repository SHALL configure mkdocstrings to extract API documentation from all module docstrings.
- The documentation site SHALL include comprehensive API documentation pages for all major subsystems.
- API documentation pages SHALL use `:::` syntax to automatically include docstrings from source code.
- Documentation generation SHALL fail if docstrings are malformed or incomplete.
- Generated documentation SHALL be reviewed for formatting correctness and completeness.

#### Scenario: API documentation is generated for all modules

- **GIVEN** `mkdocs.yml` configured with mkdocstrings plugin
- **WHEN** `mkdocs build` runs
- **THEN** it generates API documentation for all major subsystems
- **AND** docstring sections (Args, Returns, Raises) are formatted correctly
- **AND** cross-references (`:class:`, `:func:`) are rendered as links
- **AND** examples are formatted as code blocks
- **AND** all modules are included in the API documentation

#### Scenario: Documentation generation fails on malformed docstrings

- **GIVEN** the repository contains modules with malformed docstrings
- **WHEN** `mkdocs build` runs
- **THEN** it fails with clear error messages about malformed docstrings
- **AND** error messages specify exact file and line number
- **AND** CI blocks deployment until docstrings are fixed

### Requirement: Developer resources must provide comprehensive extension guidance

- The repository SHALL provide comprehensive extension guides for all major subsystems.
- Extension guides SHALL include step-by-step instructions with code examples for each component type.
- Extension guides SHALL document testing patterns for each component type with fixture examples.
- Extension guides SHALL explain error handling strategies and how to add new error translations.
- The repository SHALL include architecture decision records explaining key design choices.
- All guides SHALL be discoverable in the documentation site and referenced in contribution guidelines.

#### Scenario: Extension guide provides complete implementation patterns

- **GIVEN** a developer wants to add a new adapter for a biomedical data source
- **WHEN** they consult `docs/guides/adapter_development_guide.md`
- **THEN** the guide provides step-by-step instructions for:
  1. Defining adapter interface
  2. Implementing data fetching logic
  3. Implementing data parsing logic
  4. Adding error handling and retry logic
  5. Writing comprehensive docstrings
  6. Adding unit and integration tests
- **AND** the guide includes complete code examples
- **AND** the examples compile and run without errors

#### Scenario: Extension guide documents testing patterns

- **GIVEN** a developer wants to test a new service component
- **WHEN** they consult `docs/guides/service_development_guide.md`
- **THEN** the guide provides testing patterns for:
  - Unit testing with mocked dependencies
  - Integration testing with real dependencies
  - Performance testing with benchmarks
  - Error handling testing with exception scenarios
- **AND** the guide includes fixture examples for each testing pattern
- **AND** the examples demonstrate best practices

### Requirement: Visual documentation must illustrate repository architecture

- The repository SHALL include comprehensive diagrams showing relationships between all major components.
- Diagrams SHALL use Mermaid syntax and be stored in the documentation repository.
- Diagrams SHALL illustrate data flow, component interactions, and architectural patterns.
- Diagrams SHALL be referenced in relevant documentation and kept up-to-date with code changes.

#### Scenario: Repository architecture diagram shows all major components

- **GIVEN** the repository contains multiple major subsystems
- **WHEN** a developer views `docs/diagrams/repository_architecture.mmd`
- **THEN** the diagram shows all major components: Gateway, Services, Adapters, Orchestration, KG, Storage, Validation, Utils
- **AND** the diagram shows relationships between components
- **AND** the diagram uses consistent styling and clear labels
- **AND** the diagram is referenced in the main documentation

#### Scenario: Domain interaction diagrams show data flow

- **GIVEN** the repository contains complex data flows between domains
- **WHEN** a developer views domain interaction diagrams
- **THEN** the diagrams show data flow between adapters, services, and storage
- **AND** the diagrams illustrate error handling and retry patterns
- **AND** the diagrams show performance characteristics and bottlenecks
- **AND** the diagrams are kept up-to-date with code changes

### Requirement: Troubleshooting guides must document common issues and solutions

- The repository SHALL include comprehensive troubleshooting guides for all major subsystems.
- Troubleshooting guides SHALL document common issues, error messages, and step-by-step solutions.
- Troubleshooting guides SHALL include diagnostic commands and tools for identifying issues.
- Troubleshooting guides SHALL be organized by subsystem and issue type for easy navigation.

#### Scenario: Troubleshooting guide helps resolve adapter issues

- **GIVEN** a developer encounters an adapter-related error
- **WHEN** they consult `docs/troubleshooting/repository_issues.md`
- **THEN** the guide provides specific solutions for:
  - API rate limiting issues
  - Authentication failures
  - Data parsing errors
  - Network connectivity problems
- **AND** the guide includes diagnostic commands for identifying root causes
- **AND** the guide provides step-by-step resolution procedures

#### Scenario: Troubleshooting guide helps resolve service issues

- **GIVEN** a developer encounters a service-related error
- **WHEN** they consult the troubleshooting guide
- **THEN** the guide provides specific solutions for:
  - Service startup failures
  - Configuration errors
  - Dependency injection issues
  - Performance problems
- **AND** the guide includes monitoring and logging guidance
- **AND** the guide provides escalation procedures for complex issues

## MODIFIED Requirements

### Requirement: Existing pipeline documentation standards must be extended repository-wide

- The successful pipeline documentation standards SHALL be extended to cover all 360+ Python files in the repository.
- All modules SHALL follow the same docstring templates, section header standards, and validation rules established for pipeline modules.
- The existing validation tools SHALL be enhanced to support all module types across the repository.
- The existing pre-commit hooks and CI workflows SHALL be updated to enforce standards repository-wide.

#### Scenario: Pipeline standards are applied to all modules

- **GIVEN** the repository contains modules beyond the original pipeline scope
- **WHEN** the documentation standards are applied
- **THEN** all modules follow the same docstring templates and section header standards
- **AND** all modules pass the same validation checks
- **AND** all modules are included in the same CI workflows
- **AND** the entire repository achieves 100% docstring coverage

## REMOVED Requirements

### Requirement: Legacy documentation approaches must be eliminated

- **Reason**: Legacy documentation approaches are inconsistent and don't provide adequate guidance for developers.
- **Migration**: All legacy documentation approaches are replaced with the comprehensive Google-style docstring format and standardized section headers.

### Requirement: Inconsistent code organization must be eliminated

- **Reason**: Inconsistent code organization makes the repository difficult to navigate and maintain.
- **Migration**: All modules are reorganized to follow consistent section headers and ordering rules appropriate for their module type.
