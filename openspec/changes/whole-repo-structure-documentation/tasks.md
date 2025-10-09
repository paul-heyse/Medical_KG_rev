# Whole Repository Structure Documentation Tasks

This document provides highly detailed, actionable tasks for AI agents to implement comprehensive documentation and structural refactoring across the entire Medical_KG_rev repository. Each task specifies exact files, expected outcomes, and validation criteria.

## 0. Pre-Implementation Setup

- [ ] 0.1 **Create comprehensive audit workbook** at `openspec/changes/whole-repo-structure-documentation/audit.md` with sections: Complete File Inventory, Documentation Gap Analysis, Duplicate Code Detection, Type Hint Assessment, Structural Analysis, Legacy Code Identification
- [ ] 0.2 **Extend documentation templates directory** at `openspec/changes/whole-repo-structure-documentation/templates/` with additional templates for: adapter modules, validation modules, kg modules, storage modules, utility modules, test modules
- [ ] 0.3 **Create comprehensive section header standards** at `openspec/changes/whole-repo-structure-documentation/section_headers.md` listing all required section headers and ordering rules for all module types across the repository
- [ ] 0.4 **Create domain-specific documentation guides** at `openspec/changes/whole-repo-structure-documentation/guides/` with specific guidance for each major subsystem

## 1. Comprehensive Repository Audit

### 1.1 Complete File Inventory

- [ ] 1.1.1 **Create comprehensive file inventory** in `audit.md` under "Complete File Inventory" section including exact file paths, line counts, and primary responsibilities for all Python files:

  **Gateway Modules (15+ files):**
  - `src/Medical_KG_rev/gateway/coordinators/` (4 files)
  - `src/Medical_KG_rev/gateway/services.py`
  - `src/Medical_KG_rev/gateway/chunking_errors.py`
  - `src/Medical_KG_rev/gateway/presentation/errors.py`
  - `src/Medical_KG_rev/gateway/rest/` (if exists)
  - `src/Medical_KG_rev/gateway/graphql/` (if exists)
  - `src/Medical_KG_rev/gateway/grpc/` (if exists)

  **Service Modules (50+ files):**
  - `src/Medical_KG_rev/services/embedding/` (all submodules)
  - `src/Medical_KG_rev/services/chunking/` (all submodules)
  - `src/Medical_KG_rev/services/retrieval/` (all submodules)
  - `src/Medical_KG_rev/services/reranking/` (all submodules)
  - `src/Medical_KG_rev/services/evaluation/` (all submodules)
  - `src/Medical_KG_rev/services/extraction/` (all submodules)
  - `src/Medical_KG_rev/services/gpu/` (all submodules)
  - `src/Medical_KG_rev/services/mineru/` (all submodules)
  - `src/Medical_KG_rev/services/ingestion/` (all submodules)
  - `src/Medical_KG_rev/services/parsing/` (all submodules)
  - `src/Medical_KG_rev/services/grpc/` (all submodules)
  - `src/Medical_KG_rev/services/health.py`

  **Adapter Modules (30+ files):**
  - `src/Medical_KG_rev/adapters/base.py`
  - `src/Medical_KG_rev/adapters/yaml_parser.py`
  - `src/Medical_KG_rev/adapters/biomedical.py`
  - `src/Medical_KG_rev/adapters/core/` (all submodules)
  - `src/Medical_KG_rev/adapters/openalex/` (all submodules)
  - `src/Medical_KG_rev/adapters/pmc/` (all submodules)
  - `src/Medical_KG_rev/adapters/unpaywall/` (all submodules)
  - `src/Medical_KG_rev/adapters/terminology/` (all submodules)
  - `src/Medical_KG_rev/adapters/openfda/` (all submodules)
  - `src/Medical_KG_rev/adapters/clinicaltrials/` (all submodules)
  - `src/Medical_KG_rev/adapters/crossref/` (all submodules)
  - `src/Medical_KG_rev/adapters/plugins/` (all submodules)
  - `src/Medical_KG_rev/adapters/mixins/` (all submodules)

  **Orchestration Modules (20+ files):**
  - `src/Medical_KG_rev/orchestration/dagster/` (all submodules)
  - `src/Medical_KG_rev/orchestration/stages/` (all submodules)
  - `src/Medical_KG_rev/orchestration/ledger.py`
  - `src/Medical_KG_rev/orchestration/openlineage.py`
  - `src/Medical_KG_rev/orchestration/events.py`
  - `src/Medical_KG_rev/orchestration/kafka.py`
  - `src/Medical_KG_rev/orchestration/state/` (all submodules)
  - `src/Medical_KG_rev/orchestration/haystack/` (all submodules)

  **Knowledge Graph Modules (5+ files):**
  - `src/Medical_KG_rev/kg/schema.py`
  - `src/Medical_KG_rev/kg/neo4j_client.py`
  - `src/Medical_KG_rev/kg/cypher_templates.py`
  - `src/Medical_KG_rev/kg/shacl.py`

  **Storage Modules (10+ files):**
  - `src/Medical_KG_rev/storage/` (all submodules)
  - `src/Medical_KG_rev/services/vector_store/` (all submodules)

  **Validation Modules (5+ files):**
  - `src/Medical_KG_rev/validation/fhir.py`
  - `src/Medical_KG_rev/validation/ucum.py`

  **Utility Modules (20+ files):**
  - `src/Medical_KG_rev/utils/` (all submodules)

  **Test Modules (100+ files):**
  - `tests/` (all subdirectories and files)

- [ ] 1.1.2 **Count lines of code** for each file using `find src/ tests/ -name "*.py" -exec wc -l {} + | sort -nr` and record in inventory table with columns: File Path | Lines | Primary Responsibility | Upstream Dependencies | Downstream Dependents | Documentation Status

### 1.2 Documentation Gap Analysis

- [ ] 1.2.1 **Run comprehensive docstring coverage analysis** across entire repository:

  ```bash
  python scripts/check_docstring_coverage.py --min-coverage 0 src/Medical_KG_rev/ tests/
  ```

- [ ] 1.2.2 **Document missing docstrings by category** and record in `audit.md`:
  - Missing module docstrings: Count and list all files without module-level docstrings
  - Missing class docstrings: Count and list all classes without docstrings
  - Missing function docstrings: Count and list all functions without docstrings
  - Missing dataclass field documentation: Count and list all dataclasses without field comments
  - Incomplete docstrings: Count functions missing Args/Returns/Raises sections

- [ ] 1.2.3 **Run pydocstyle analysis** on entire repository:

  ```bash
  pydocstyle src/Medical_KG_rev/ tests/ > audit_pydocstyle.txt 2>&1
  ```

- [ ] 1.2.4 **Categorize documentation gaps by domain** in `audit.md`:
  - Gateway modules: List specific missing docstrings
  - Service modules: List specific missing docstrings
  - Adapter modules: List specific missing docstrings
  - Orchestration modules: List specific missing docstrings
  - Knowledge graph modules: List specific missing docstrings
  - Storage modules: List specific missing docstrings
  - Validation modules: List specific missing docstrings
  - Utility modules: List specific missing docstrings
  - Test modules: List specific missing docstrings

### 1.3 Duplicate Code Detection

- [ ] 1.3.1 **Use AST analysis to identify duplicate functions** across repository:

  ```bash
  python scripts/find_duplicate_functions.py src/Medical_KG_rev/ > audit_duplicates.txt
  ```

- [ ] 1.3.2 **Search for duplicate imports** across all modules:

  ```bash
  rg -t py "from.*import" src/Medical_KG_rev/ | sort | uniq -d
  ```

- [ ] 1.3.3 **Identify duplicate class definitions** and similar patterns:

  ```bash
  rg -t py "^class " src/Medical_KG_rev/ | sort | uniq -d
  ```

- [ ] 1.3.4 **Document duplicate code patterns** in `audit.md` with:
  - File paths and line numbers for duplicates
  - Description of duplicated functionality
  - Criteria for selecting canonical implementation
  - Rationale for deletion decisions

### 1.4 Type Hint Assessment

- [ ] 1.4.1 **Run mypy in strict mode** on entire repository:

  ```bash
  mypy --strict src/Medical_KG_rev/ > audit_mypy.txt 2>&1
  ```

- [ ] 1.4.2 **Identify type hint gaps** in `audit.md`:
  - Functions missing return type annotations
  - Parameters with Any type or no annotation
  - Use of bare `dict`/`list` instead of `Mapping`/`Sequence` with generics
  - Missing `-> None` on procedures
  - Optional types not using `Type | None` syntax

- [ ] 1.4.3 **Categorize type issues by severity**:
  - Critical: Missing return types on public functions
  - High: Missing parameter types on public functions
  - Medium: Use of deprecated Optional syntax
  - Low: Missing types on private functions

### 1.5 Structural Analysis

- [ ] 1.5.1 **Identify files without section headers** across entire repository:

  ```bash
  python scripts/check_section_headers.py src/Medical_KG_rev/ tests/
  ```

- [ ] 1.5.2 **Analyze import organization** across all modules:
  - Files with ungrouped imports
  - Files with incorrect import ordering
  - Missing import grouping (stdlib, third-party, first-party, relative)
  - Missing alphabetical sorting within groups

- [ ] 1.5.3 **Analyze method ordering** across all modules:
  - Files with private methods before public methods
  - Files with scattered helper functions
  - Missing method grouping by visibility
  - Missing alphabetical ordering within groups

- [ ] 1.5.4 **Document structural issues** in `audit.md` with specific file paths and line numbers

### 1.6 Legacy Code Identification

- [ ] 1.6.1 **Search for deprecated markers** across entire repository:

  ```bash
  rg -t py "@deprecated|warnings.warn.*DeprecationWarning" src/Medical_KG_rev/ tests/
  ```

- [ ] 1.6.2 **Search for legacy comments** across entire repository:

  ```bash
  rg -t py "legacy|old|deprecated|superseded|todo.*remove" src/Medical_KG_rev/ tests/
  ```

- [ ] 1.6.3 **Document legacy code** in `audit.md` with table:
  - File | Function/Class | Deprecated Why | Replaced By | References Count

- [ ] 1.6.4 **Find references to legacy code** for each identified item:

  ```bash
  rg -t py "legacy_function_name" src/ tests/
  ```

## 2. Standards Extension & Tooling Enhancement

### 2.1 Extend Documentation Templates

- [ ] 2.1.1 **Create adapter module template** at `templates/adapter_module_docstring.py`:

  ```python
  """[One-line summary of adapter purpose].

  This adapter provides [detailed explanation of what the adapter does, its role
  in data integration, and key design decisions].

  Key Responsibilities:
      - [Responsibility 1: Data fetching from external source]
      - [Responsibility 2: Data parsing and transformation]
      - [Responsibility 3: Error handling and retry logic]
      - [Responsibility 4: Rate limiting and backoff]

  Collaborators:
      - Upstream: [List modules/services that call this adapter]
      - Downstream: [List modules/services this adapter depends on]

  Side Effects:
      - [External API calls, rate limiting, caching]
      - [Metric emission, logging]

  Thread Safety:
      - [Thread-safe: All public methods can be called concurrently]
      - [Not thread-safe: Must be called from single thread]

  Performance Characteristics:
      - [Rate limits, timeouts, retry behavior]
      - [Memory usage patterns, caching behavior]

  Example:
      >>> from Medical_KG_rev.adapters import OpenAlexAdapter
      >>> adapter = OpenAlexAdapter(api_key="...")
      >>> result = adapter.fetch("10.1371/journal.pone.0123456")
  """
  ```

- [ ] 2.1.2 **Create validation module template** at `templates/validation_module_docstring.py`
- [ ] 2.1.3 **Create kg module template** at `templates/kg_module_docstring.py`
- [ ] 2.1.4 **Create storage module template** at `templates/storage_module_docstring.py`
- [ ] 2.1.5 **Create utility module template** at `templates/utility_module_docstring.py`
- [ ] 2.1.6 **Create test module template** at `templates/test_module_docstring.py`

### 2.2 Enhance Section Header Standards

- [ ] 2.2.1 **Define adapter module structure** in `section_headers.md`:

  ```python
  # ============================================================================
  # IMPORTS
  # ============================================================================

  # ============================================================================
  # DATA MODELS
  # ============================================================================

  # ============================================================================
  # ADAPTER IMPLEMENTATION
  # ============================================================================

  # ============================================================================
  # ERROR HANDLING
  # ============================================================================

  # ============================================================================
  # FACTORY FUNCTIONS
  # ============================================================================

  # ============================================================================
  # EXPORTS
  # ============================================================================
  ```

- [ ] 2.2.2 **Define validation module structure** in `section_headers.md`
- [ ] 2.2.3 **Define kg module structure** in `section_headers.md`
- [ ] 2.2.4 **Define storage module structure** in `section_headers.md`
- [ ] 2.2.5 **Define utility module structure** in `section_headers.md`
- [ ] 2.2.6 **Define test module structure** in `section_headers.md`

### 2.3 Upgrade Validation Tools

- [ ] 2.3.1 **Enhance section header checker** to support all module types:
  - Add support for adapter, validation, kg, storage, utility module types
  - Update validation rules for each module type
  - Add domain-specific section requirements

- [ ] 2.3.2 **Enhance docstring coverage checker** for repository-wide analysis:
  - Add support for all module types
  - Improve reporting with domain breakdown
  - Add trend analysis capabilities

- [ ] 2.3.3 **Create duplicate code detector** at `scripts/find_duplicate_code.py`:
  - Use AST analysis to find duplicate functions
  - Use pattern matching to find duplicate imports
  - Generate detailed reports with line numbers

- [ ] 2.3.4 **Create type hint checker** at `scripts/check_type_hints.py`:
  - Validate modern Python type hint usage
  - Check for deprecated Optional syntax
  - Validate generic type parameters

### 2.4 Configure Enforcement

- [ ] 2.4.1 **Update pre-commit hooks** in `.pre-commit-config.yaml`:

  ```yaml
  - repo: local
    hooks:
      - id: ruff-docstring-check
        name: Check docstrings with ruff
        entry: ruff check --select D
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/

      - id: section-header-check
        name: Check section headers
        entry: python scripts/check_section_headers.py
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/

      - id: docstring-coverage
        name: Check docstring coverage
        entry: python scripts/check_docstring_coverage.py --min-coverage 90
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/

      - id: duplicate-code-check
        name: Check for duplicate code
        entry: python scripts/find_duplicate_code.py
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/

      - id: type-hint-check
        name: Check type hints
        entry: python scripts/check_type_hints.py
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/
  ```

- [ ] 2.4.2 **Update CI workflow** in `.github/workflows/documentation.yml`:

  ```yaml
  name: Repository Documentation Quality

  on:
    pull_request:
      paths:
        - 'src/Medical_KG_rev/**'
        - 'tests/**'
    push:
      branches: [main]

  jobs:
    check-docs:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4

        - name: Set up Python
          uses: actions/setup-python@v5
          with:
            python-version: '3.11'

        - name: Install dependencies
          run: |
            pip install ruff mypy

        - name: Check docstrings
          run: |
            ruff check --select D src/Medical_KG_rev/ tests/

        - name: Check section headers
          run: |
            python scripts/check_section_headers.py

        - name: Check docstring coverage
          run: |
            python scripts/check_docstring_coverage.py --min-coverage 90

        - name: Check for duplicate code
          run: |
            python scripts/find_duplicate_code.py

        - name: Check type hints
          run: |
            python scripts/check_type_hints.py

        - name: Run mypy
          run: |
            mypy --strict src/Medical_KG_rev/
  ```

### 2.5 Create Migration Tools

- [ ] 2.5.1 **Create automated docstring generator** at `scripts/generate_docstrings.py`:
  - Analyze function signatures to generate Args sections
  - Analyze return statements to generate Returns sections
  - Analyze raise statements to generate Raises sections
  - Use templates to generate consistent docstrings

- [ ] 2.5.2 **Create section header inserter** at `scripts/insert_section_headers.py`:
  - Analyze module structure to determine appropriate sections
  - Insert section headers in correct locations
  - Reorganize code into appropriate sections

- [ ] 2.5.3 **Create import organizer** at `scripts/organize_imports.py`:
  - Group imports by category (stdlib, third-party, first-party, relative)
  - Sort imports alphabetically within groups
  - Remove duplicate imports

## 3. Domain-by-Domain Refactoring

### 3.1 Gateway Modules Refactoring

- [ ] 3.1.1 **Apply documentation standards to all gateway coordinators**:
  - `src/Medical_KG_rev/gateway/coordinators/base.py` (already completed)
  - `src/Medical_KG_rev/gateway/coordinators/chunking.py` (already completed)
  - `src/Medical_KG_rev/gateway/coordinators/embedding.py` (already completed)
  - `src/Medical_KG_rev/gateway/coordinators/job_lifecycle.py` (already completed)

- [ ] 3.1.2 **Apply documentation standards to gateway services**:
  - `src/Medical_KG_rev/gateway/services.py` (already completed)
  - `src/Medical_KG_rev/gateway/chunking_errors.py` (already completed)
  - `src/Medical_KG_rev/gateway/presentation/errors.py` (already completed)

- [ ] 3.1.3 **Apply documentation standards to remaining gateway modules**:
  - Any additional gateway modules identified in audit
  - Ensure 100% docstring coverage
  - Apply consistent section headers
  - Organize imports and methods

### 3.2 Service Modules Refactoring

- [ ] 3.2.1 **Apply documentation standards to embedding services**:
  - `src/Medical_KG_rev/services/embedding/persister.py` (already completed)
  - `src/Medical_KG_rev/services/embedding/telemetry.py` (already completed)
  - `src/Medical_KG_rev/services/embedding/registry.py` (already completed)
  - `src/Medical_KG_rev/services/embedding/policy.py` (already completed)
  - `src/Medical_KG_rev/services/embedding/service.py` (already completed)
  - `src/Medical_KG_rev/services/embedding/events.py`
  - `src/Medical_KG_rev/services/embedding/cache.py`
  - `src/Medical_KG_rev/services/embedding/namespace/` (all submodules)

- [ ] 3.2.2 **Apply documentation standards to chunking services**:
  - `src/Medical_KG_rev/services/chunking/runtime.py` (already completed)
  - `src/Medical_KG_rev/services/chunking/` (all remaining submodules)

- [ ] 3.2.3 **Apply documentation standards to retrieval services**:
  - `src/Medical_KG_rev/services/retrieval/retrieval_service.py` (already completed)
  - `src/Medical_KG_rev/services/retrieval/` (all remaining submodules)

- [ ] 3.2.4 **Apply documentation standards to reranking services**:
  - `src/Medical_KG_rev/services/reranking/` (all submodules)

- [ ] 3.2.5 **Apply documentation standards to evaluation services**:
  - `src/Medical_KG_rev/services/evaluation/test_sets.py` (already completed)
  - `src/Medical_KG_rev/services/evaluation/metrics.py` (already completed)
  - `src/Medical_KG_rev/services/evaluation/ci.py` (already completed)
  - `src/Medical_KG_rev/services/evaluation/` (all remaining submodules)

- [ ] 3.2.6 **Apply documentation standards to extraction services**:
  - `src/Medical_KG_rev/services/extraction/` (all submodules)

- [ ] 3.2.7 **Apply documentation standards to gpu services**:
  - `src/Medical_KG_rev/services/gpu/` (all submodules)

- [ ] 3.2.8 **Apply documentation standards to mineru services**:
  - `src/Medical_KG_rev/services/mineru/service.py` (already completed)
  - `src/Medical_KG_rev/services/mineru/types.py` (already completed)
  - `src/Medical_KG_rev/services/mineru/cli_wrapper.py` (already completed)
  - `src/Medical_KG_rev/services/mineru/vllm_client.py` (already completed)
  - `src/Medical_KG_rev/services/mineru/circuit_breaker.py` (already completed)
  - `src/Medical_KG_rev/services/mineru/artifacts.py` (already completed)
  - `src/Medical_KG_rev/services/mineru/metrics.py` (already completed)
  - `src/Medical_KG_rev/services/mineru/output_parser.py` (already completed)
  - `src/Medical_KG_rev/services/mineru/pipeline.py` (already completed)
  - `src/Medical_KG_rev/services/mineru/postprocessor.py` (already completed)
  - `src/Medical_KG_rev/services/mineru/__init__.py` (already completed)

- [ ] 3.2.9 **Apply documentation standards to remaining service modules**:
  - `src/Medical_KG_rev/services/ingestion/` (all submodules)
  - `src/Medical_KG_rev/services/parsing/` (all submodules)
  - `src/Medical_KG_rev/services/grpc/` (all submodules)
  - `src/Medical_KG_rev/services/health.py` (already completed)

### 3.3 Adapter Modules Refactoring

- [ ] 3.3.1 **Apply documentation standards to core adapters**:
  - `src/Medical_KG_rev/adapters/base.py` (already completed)
  - `src/Medical_KG_rev/adapters/yaml_parser.py` (already completed)
  - `src/Medical_KG_rev/adapters/biomedical.py` (already completed)

- [ ] 3.3.2 **Apply documentation standards to domain-specific adapters**:
  - `src/Medical_KG_rev/adapters/core/` (all submodules)
  - `src/Medical_KG_rev/adapters/openalex/` (all submodules)
  - `src/Medical_KG_rev/adapters/pmc/` (all submodules)
  - `src/Medical_KG_rev/adapters/unpaywall/` (all submodules)
  - `src/Medical_KG_rev/adapters/terminology/` (all submodules)
  - `src/Medical_KG_rev/adapters/openfda/` (all submodules)
  - `src/Medical_KG_rev/adapters/clinicaltrials/` (all submodules)
  - `src/Medical_KG_rev/adapters/crossref/` (all submodules)

- [ ] 3.3.3 **Apply documentation standards to adapter infrastructure**:
  - `src/Medical_KG_rev/adapters/plugins/` (all submodules)
  - `src/Medical_KG_rev/adapters/mixins/` (all submodules)

### 3.4 Orchestration Modules Refactoring

- [ ] 3.4.1 **Apply documentation standards to dagster modules**:
  - `src/Medical_KG_rev/orchestration/dagster/` (all submodules)

- [ ] 3.4.2 **Apply documentation standards to stage modules**:
  - `src/Medical_KG_rev/orchestration/stages/contracts.py`
  - `src/Medical_KG_rev/orchestration/stages/plugins.py`
  - `src/Medical_KG_rev/orchestration/stages/plugin_manager.py`
  - `src/Medical_KG_rev/orchestration/stages/plugins/builtin.py` (already completed)
  - `src/Medical_KG_rev/orchestration/stages/pdf_download.py` (already completed)
  - `src/Medical_KG_rev/orchestration/stages/pdf_gate.py` (already completed)

- [ ] 3.4.3 **Apply documentation standards to orchestration infrastructure**:
  - `src/Medical_KG_rev/orchestration/ledger.py`
  - `src/Medical_KG_rev/orchestration/openlineage.py`
  - `src/Medical_KG_rev/orchestration/events.py`
  - `src/Medical_KG_rev/orchestration/kafka.py`
  - `src/Medical_KG_rev/orchestration/state/` (all submodules)
  - `src/Medical_KG_rev/orchestration/haystack/` (all submodules)

### 3.5 Knowledge Graph Modules Refactoring

- [ ] 3.5.1 **Apply documentation standards to kg modules**:
  - `src/Medical_KG_rev/kg/schema.py` (already completed)
  - `src/Medical_KG_rev/kg/neo4j_client.py` (already completed)
  - `src/Medical_KG_rev/kg/cypher_templates.py` (already completed)
  - `src/Medical_KG_rev/kg/shacl.py` (already completed)

### 3.6 Storage Modules Refactoring

- [ ] 3.6.1 **Apply documentation standards to storage modules**:
  - `src/Medical_KG_rev/storage/` (all submodules)

- [ ] 3.6.2 **Apply documentation standards to vector store modules**:
  - `src/Medical_KG_rev/services/vector_store/monitoring.py`
  - `src/Medical_KG_rev/services/vector_store/registry.py`
  - `src/Medical_KG_rev/services/vector_store/service.py`
  - `src/Medical_KG_rev/services/vector_store/factory.py`
  - `src/Medical_KG_rev/services/vector_store/gpu.py`
  - `src/Medical_KG_rev/services/vector_store/compression.py`
  - `src/Medical_KG_rev/services/vector_store/evaluation.py`
  - `src/Medical_KG_rev/services/vector_store/types.py` (already completed)
  - `src/Medical_KG_rev/services/vector_store/models.py`
  - `src/Medical_KG_rev/services/vector_store/errors.py` (already completed)
  - `src/Medical_KG_rev/services/vector_store/stores/` (all submodules)

### 3.7 Validation Modules Refactoring

- [ ] 3.7.1 **Apply documentation standards to validation modules**:
  - `src/Medical_KG_rev/validation/fhir.py`
  - `src/Medical_KG_rev/validation/ucum.py` (already completed)

### 3.8 Utility Modules Refactoring

- [ ] 3.8.1 **Apply documentation standards to utility modules**:
  - `src/Medical_KG_rev/utils/errors.py` (already completed)
  - `src/Medical_KG_rev/utils/` (all remaining submodules)

### 3.9 Test Modules Refactoring

- [ ] 3.9.1 **Apply documentation standards to test modules**:
  - `tests/adapters/` (all submodules)
  - `tests/auth/` (all submodules)
  - `tests/chunking/` (all submodules)
  - `tests/config/` (all submodules)
  - `tests/contract/` (all submodules)
  - `tests/embeddings/` (all submodules)
  - `tests/eval/` (all submodules)
  - `tests/gateway/` (all submodules)
  - `tests/integration/` (all submodules)
  - `tests/kg/` (all submodules)
  - `tests/models/` (all submodules)
  - `tests/observability/` (all submodules)
  - `tests/orchestration/` (all submodules)
  - `tests/performance/` (all submodules)
  - `tests/quality/` (all submodules)
  - `tests/scripts/` (all submodules)
  - `tests/services/` (all submodules)
  - `tests/storage/` (all submodules)
  - `tests/utils/` (all submodules)
  - `tests/validation/` (all submodules)
  - `tests/test_basic.py`

## 4. Advanced Documentation & Integration

### 4.1 API Documentation Generation

- [ ] 4.1.1 **Configure mkdocstrings for complete repository coverage** in `mkdocs.yml`:

  ```yaml
  plugins:
    - search
    - mkdocstrings:
        handlers:
          python:
            options:
              show_source: true
              show_root_heading: true
              heading_level: 2
              docstring_style: google
              merge_init_into_class: true
  ```

- [ ] 4.1.2 **Create comprehensive API documentation pages**:
  - `docs/api/gateway.md` - Gateway coordinators and services
  - `docs/api/services.md` - All service modules
  - `docs/api/adapters.md` - All adapter modules
  - `docs/api/orchestration.md` - Orchestration modules
  - `docs/api/kg.md` - Knowledge graph modules
  - `docs/api/storage.md` - Storage modules
  - `docs/api/validation.md` - Validation modules
  - `docs/api/utils.md` - Utility modules

- [ ] 4.1.3 **Update navigation** in `mkdocs.yml`:

  ```yaml
  nav:
    - Home: index.md
    - API Documentation:
      - Gateway: api/gateway.md
      - Services: api/services.md
      - Adapters: api/adapters.md
      - Orchestration: api/orchestration.md
      - Knowledge Graph: api/kg.md
      - Storage: api/storage.md
      - Validation: api/validation.md
      - Utilities: api/utils.md
  ```

### 4.2 Architecture Decision Records

- [ ] 4.2.1 **Create ADR for repository-wide documentation standards**:
  - `docs/adr/0005-repository-documentation-standards.md`

- [ ] 4.2.2 **Create ADR for domain-specific section headers**:
  - `docs/adr/0006-domain-specific-section-headers.md`

- [ ] 4.2.3 **Create ADR for automated documentation enforcement**:
  - `docs/adr/0007-automated-documentation-enforcement.md`

- [ ] 4.2.4 **Create ADR for type hint modernization**:
  - `docs/adr/0008-type-hint-modernization.md`

### 4.3 Developer Extension Guides

- [ ] 4.3.1 **Create comprehensive extension guide** at `docs/guides/repository_extension_guide.md`:
  - Adding new adapters
  - Adding new services
  - Adding new orchestration stages
  - Adding new validation rules
  - Adding new storage backends
  - Adding new utility functions
  - Testing patterns for each component type

- [ ] 4.3.2 **Create domain-specific guides**:
  - `docs/guides/adapter_development_guide.md`
  - `docs/guides/service_development_guide.md`
  - `docs/guides/orchestration_development_guide.md`
  - `docs/guides/validation_development_guide.md`
  - `docs/guides/storage_development_guide.md`

### 4.4 Visual Documentation

- [ ] 4.4.1 **Create repository architecture diagram** at `docs/diagrams/repository_architecture.mmd`:

  ```mermaid
  graph TD
      Gateway[Gateway Layer]
      Services[Service Layer]
      Adapters[Adapter Layer]
      Orchestration[Orchestration Layer]
      KG[Knowledge Graph Layer]
      Storage[Storage Layer]
      Validation[Validation Layer]
      Utils[Utility Layer]

      Gateway --> Services
      Services --> Adapters
      Services --> Storage
      Services --> Validation
      Orchestration --> Services
      KG --> Storage
      Utils --> Services
      Utils --> Adapters
  ```

- [ ] 4.4.2 **Create domain interaction diagrams**:
  - `docs/diagrams/adapter_data_flow.mmd`
  - `docs/diagrams/service_interactions.mmd`
  - `docs/diagrams/orchestration_pipeline.mmd`
  - `docs/diagrams/storage_architecture.mmd`

### 4.5 Troubleshooting Guides

- [ ] 4.5.1 **Create comprehensive troubleshooting guide** at `docs/troubleshooting/repository_issues.md`:
  - Gateway issues
  - Service issues
  - Adapter issues
  - Orchestration issues
  - Knowledge graph issues
  - Storage issues
  - Validation issues
  - Documentation lint failures

## 5. Validation & Quality Assurance

### 5.1 Comprehensive Testing

- [ ] 5.1.1 **Run full test suite** to ensure no regressions:

  ```bash
  pytest tests/ -v --cov=src/Medical_KG_rev --cov-report=html --cov-report=term
  ```

- [ ] 5.1.2 **Verify test coverage maintained** or improved

- [ ] 5.1.3 **Run integration tests** to validate system functionality

- [ ] 5.1.4 **Run contract tests** to ensure API contracts unchanged

### 5.2 Documentation Validation

- [ ] 5.2.1 **Run all documentation validation tools**:

  ```bash
  python scripts/check_docstring_coverage.py --min-coverage 100 src/Medical_KG_rev/
  python scripts/check_section_headers.py src/Medical_KG_rev/
  python scripts/find_duplicate_code.py src/Medical_KG_rev/
  python scripts/check_type_hints.py src/Medical_KG_rev/
  ```

- [ ] 5.2.2 **Verify 100% docstring coverage** across entire repository

- [ ] 5.2.3 **Verify 0 section header violations**

- [ ] 5.2.4 **Verify 0 duplicate code blocks**

- [ ] 5.2.5 **Verify 0 type hint violations**

### 5.3 Performance Validation

- [ ] 5.3.1 **Run performance benchmarks** to ensure no degradation

- [ ] 5.3.2 **Profile memory usage** to ensure documentation doesn't impact memory

- [ ] 5.3.3 **Measure import time** to ensure no significant slowdown

### 5.4 Integration Testing

- [ ] 5.4.1 **Test all major workflows** end-to-end

- [ ] 5.4.2 **Validate cross-module interactions** work correctly

- [ ] 5.4.3 **Test error handling** across all modules

### 5.5 Final Quality Checks

- [ ] 5.5.1 **Run pre-commit on all files**:

  ```bash
  pre-commit run --all-files
  ```

- [ ] 5.5.2 **Run ruff on entire repository**:

  ```bash
  ruff check src/ tests/
  ```

- [ ] 5.5.3 **Run mypy on entire repository**:

  ```bash
  mypy --strict src/Medical_KG_rev/
  ```

- [ ] 5.5.4 **Build documentation**:

  ```bash
  mkdocs build --strict
  ```

- [ ] 5.5.5 **Verify documentation renders correctly** in browser

## 6. Finalization & Sign-Off

### 6.1 Create Summary Report

- [ ] 6.1.1 **Create comprehensive summary** at `openspec/changes/whole-repo-structure-documentation/SUMMARY.md`:
  - Total files refactored: [count]
  - Docstring coverage: Before [percent]% → After 100%
  - Lines of code: Before [count] → After [count]
  - Duplicate code blocks removed: [count]
  - Type hints modernized: [count]
  - Section headers added: [count]
  - Validation tools created: [count]
  - Documentation pages created: [count]

### 6.2 Request Peer Review

- [ ] 6.2.1 **Create PR with all changes**

- [ ] 6.2.2 **Include comprehensive PR description** with:
  - Link to change proposal
  - Link to summary report
  - List of key improvements
  - Validation results

- [ ] 6.2.3 **Request review from all stakeholder teams**

### 6.3 Address Review Feedback

- [ ] 6.3.1 **Incorporate reviewer feedback**

- [ ] 6.3.2 **Update documentation** based on feedback

- [ ] 6.3.3 **Re-run validation checks** after changes

### 6.4 Merge and Archive

- [ ] 6.4.1 **Merge PR** after approval

- [ ] 6.4.2 **Run OpenSpec archive**:

  ```bash
  openspec archive whole-repo-structure-documentation
  ```

- [ ] 6.4.3 **Update project documentation** to reflect new standards

- [ ] 6.4.4 **Announce completion** to team

---

## Notes for AI Agents

**Critical Success Factors:**

1. **Completeness**: Every Python file in the repository must have comprehensive documentation
2. **Consistency**: Follow established templates and standards exactly
3. **Validation**: Use automated tools to ensure compliance
4. **Testing**: Maintain all existing functionality during refactoring
5. **Documentation**: Document all decisions and changes

**When In Doubt:**

- Refer to templates in `templates/` directory
- Check section header standards in `section_headers.md`
- Look at completed pipeline modules for examples
- Consult developer guides for extension patterns

**Order of Operations:**

1. Complete comprehensive audit (tasks 1.1-1.6)
2. Extend standards and tooling (tasks 2.1-2.5)
3. Refactor domain-by-domain (tasks 3.1-3.9)
4. Create advanced documentation (tasks 4.1-4.5)
5. Validate and sign-off (tasks 5.1-6.4)

**Success Criteria:**

- 100% docstring coverage across entire repository
- 0 errors from all validation tools
- All tests passing
- Complete API documentation generated
- Developer guides and examples created
- Performance maintained or improved
