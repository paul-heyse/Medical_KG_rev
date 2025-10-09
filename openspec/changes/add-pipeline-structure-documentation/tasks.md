# Pipeline Structure Documentation Tasks

This document provides highly detailed, actionable tasks for AI agents to implement comprehensive documentation and structural refactoring of the pipeline codebase. Each task specifies exact files, line numbers (where known), expected outcomes, and validation criteria.

## 0. Pre-Implementation Setup

- [x] 0.1 **Create audit workbook** at `openspec/changes/add-pipeline-structure-documentation/audit.md` with sections: Duplicate Code Analysis, Missing Documentation, Import Issues, Structural Problems, Type Hint Gaps, Test Coverage
- [x] 0.2 **Create docstring templates directory** at `openspec/changes/add-pipeline-structure-documentation/templates/` with example templates for: module, class, function, method, dataclass, protocol, exception handler, async function, decorator, property
- [x] 0.3 **Create section header standards document** at `openspec/changes/add-pipeline-structure-documentation/section_headers.md` listing all required section headers and ordering rules per module type (coordinator, service, policy, orchestration, test)

## 1. Discovery & Audit

### 1.1 File Inventory

- [x] 1.1.1 **Create comprehensive file inventory** in `audit.md` under "File Inventory" section including exact file paths, line counts, and primary responsibilities for:
  - `src/Medical_KG_rev/gateway/coordinators/base.py`
  - `src/Medical_KG_rev/gateway/coordinators/chunking.py`
  - `src/Medical_KG_rev/gateway/coordinators/embedding.py`
  - `src/Medical_KG_rev/gateway/coordinators/job_lifecycle.py`
  - `src/Medical_KG_rev/gateway/services.py`
  - `src/Medical_KG_rev/gateway/chunking_errors.py`
  - `src/Medical_KG_rev/services/retrieval/chunking.py`
  - `src/Medical_KG_rev/services/retrieval/chunking_command.py` (check if exists)
  - `src/Medical_KG_rev/services/embedding/policy.py`
  - `src/Medical_KG_rev/services/embedding/persister.py`
  - `src/Medical_KG_rev/services/embedding/telemetry.py`
  - `src/Medical_KG_rev/services/embedding/registry.py`
  - `src/Medical_KG_rev/services/embedding/namespace/access.py`
  - `src/Medical_KG_rev/services/embedding/namespace/registry.py`
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py`
  - `src/Medical_KG_rev/orchestration/dagster/stages.py`
  - `src/Medical_KG_rev/orchestration/stages/contracts.py`
  - `src/Medical_KG_rev/orchestration/stages/plugins.py`
  - `src/Medical_KG_rev/orchestration/stages/plugin_manager.py`
  - `src/Medical_KG_rev/orchestration/stages/plugins/builtin.py`
  - All test files under `tests/gateway/`, `tests/services/`, `tests/orchestration/` matching these modules

- [x] 1.1.2 **Count lines of code** for each file using `wc -l` and record in inventory table with columns: File Path | Lines | Primary Responsibility | Upstream Dependencies | Downstream Dependents

### 1.2 Duplicate Code Analysis

- [x] 1.2.1 **Document duplicate imports in `chunking.py`:** In `audit.md` under "Duplicate Imports", record:
  - Line 18 vs 21: `from Medical_KG_rev.gateway.models import DocumentChunk` (duplicated)
  - Line 19 vs 22: `from Medical_KG_rev.observability.metrics import record_chunking_failure` (duplicated)
  - Line 20 vs 23: Import of ChunkingService (once without ChunkCommand, once with)
  - **Resolution:** Keep lines 18-23 group that imports ChunkCommand, delete lines 18-20

- [x] 1.2.2 **Document duplicate imports in `services.py`:** In `audit.md`, record:
  - Lines 56, 61, 67: Multiple imports from different locations of ProblemDetail, UCUMValidator, FHIRValidator
  - Use `rg "import ProblemDetail|import UCUMValidator|import FHIRValidator" src/Medical_KG_rev/gateway/services.py -n` to find exact lines
  - **Resolution:** Keep imports from canonical modules (`..utils.errors`, `..validation`, `..validation.fhir`), delete duplicates

- [x] 1.2.3 **Document duplicate code blocks in `chunking.py`:** In `audit.md` under "Duplicate Code Blocks", create table:

  | Location | Description | Canonical Implementation | Delete |
  |----------|-------------|-------------------------|--------|
  | Lines 77-84 vs 85-91 | ChunkCommand creation vs ChunkingOptions creation | ChunkCommand (used by error translator) | ChunkingOptions block |
  | Lines 95-119 vs 120-210 | Exception handling with _translate_error vs manual ProblemDetail | First block (uses error translator) | Second block |
  | Lines 239-249 vs 275-287 | _extract_text method | First (cleaner error handling) | Second |
  | Lines 251-274 vs 320-346 | _translate_error vs _error method | _translate_error (integrates with translator) | _error method |
  | Lines 289-293 | _metadata_without_text helper | N/A (only used by deleted ChunkingOptions code) | Delete |

- [x] 1.2.4 **Search for other duplicate code:** Run `rg -t py "def " src/Medical_KG_rev/gateway/coordinators/chunking.py` to list all function definitions and manually inspect for duplicates not yet identified

### 1.3 Missing Documentation Audit

- [x] 1.3.1 **Run pydocstyle on all files:** Execute `pydocstyle src/Medical_KG_rev/gateway/ src/Medical_KG_rev/services/ src/Medical_KG_rev/orchestration/ > audit_pydocstyle.txt 2>&1` and summarize results in `audit.md`

- [x] 1.3.2 **Count missing docstrings by category** and record in `audit.md`:
  - Missing module docstrings: `rg -t py --files-without-match '^"""' src/Medical_KG_rev/gateway/ src/Medical_KG_rev/services/ src/Medical_KG_rev/orchestration/ | wc -l`
  - Missing class docstrings: Use AST parser to count classes without docstrings
  - Missing function docstrings: Use AST parser to count functions without docstrings
  - Missing dataclass field documentation: Check for dataclasses without inline field comments

- [x] 1.3.3 **Identify incomplete docstrings:** For files that have docstrings, check for missing sections:
  - Functions missing Args section when they have parameters
  - Functions missing Returns section when they return non-None values
  - Functions missing Raises section when they raise exceptions
  - Classes missing Attributes section when they have instance variables

### 1.4 Type Hint Analysis

- [x] 1.4.1 **Run mypy in strict mode:** Execute `mypy --strict src/Medical_KG_rev/gateway/ src/Medical_KG_rev/services/ src/Medical_KG_rev/orchestration/ > audit_mypy.txt 2>&1` and summarize type issues in `audit.md`

- [x] 1.4.2 **Identify type hint gaps** in `audit.md`:
  - Functions missing return type annotations
  - Parameters with Any type or no annotation
  - Use of bare `dict`/`list` instead of `Mapping`/`Sequence` with generics
  - Missing `-> None` on procedures
  - Optional types not using `Type | None` syntax (using `Optional[Type]` instead)

### 1.5 Structural Issues

- [x] 1.5.1 **Identify files without section headers:** List all files in scope that lack structural section headers

- [x] 1.5.2 **Identify files with inconsistent ordering:** Check for files where:
  - Imports are not grouped (stdlib, third-party, first-party, relative)
  - Private methods appear before public methods
  - Helper functions are scattered throughout instead of grouped at end
  - Exception handlers appear before the code they protect

### 1.6 Search for Additional Pipeline Modules

- [x] 1.6.1 **Search for unlisted pipeline orchestration code:** Run `rg -t py "chunk|embed|coordinator|pipeline" --files-with-matches src/ | sort | uniq` and cross-reference against inventory to find any unlisted modules; add to `audit.md` if found

### 1.7 Module Dependency Graph

- [x] 1.7.1 **Create dependency graph** in `audit.md` using Mermaid syntax showing which modules import which, highlighting circular dependencies or unnecessary coupling

## 2. Documentation Standards & Templates

### 2.1 Module Docstring Template

- [x] 2.1.1 **Create `templates/module_docstring.py`** with comprehensive template:

  ```python
  """[One-line summary of module purpose].

  This module provides [detailed explanation of what the module does, its role
  in the larger system, and key design decisions].

  Key Responsibilities:
      - [Responsibility 1: Be specific about what the module handles]
      - [Responsibility 2: Include data transformations, external calls, etc.]
      - [Responsibility 3: Mention any caching, rate limiting, etc.]

  Collaborators:
      - Upstream: [List modules/services that call into this one]
      - Downstream: [List modules/services this one depends on]

  Side Effects:
      - [Database writes, external API calls, file I/O, metric emission]
      - [Global state modifications, cache updates]
      - [None if pure/functional]

  Thread Safety:
      - [Thread-safe: All public functions can be called from multiple threads]
      - [Not thread-safe: Must be called from single thread]
      - [Conditionally safe: Describe conditions]

  Performance Characteristics:
      - [Time complexity for main operations]
      - [Memory usage patterns]
      - [Rate limits or throttling behavior]

  Example:
      >>> from Medical_KG_rev.gateway.coordinators import ChunkingCoordinator
      >>> coordinator = ChunkingCoordinator(...)
      >>> result = coordinator.execute(request)
  """
  ```

### 2.2 Class Docstring Template

- [x] 2.2.1 **Create `templates/class_docstring.py`** with template:

  ```python
  """[One-line summary of class purpose].

  [Detailed explanation of what the class does, why it exists, and how it fits
  into the larger architecture. Explain the key abstractions it provides.]

  This class implements the [pattern name] pattern for [purpose]. It coordinates
  between [upstream components] and [downstream components] to [achieve goal].

  Attributes:
      attribute_name: [Type already in code, describe purpose and valid ranges]
      _private_attr: [Describe internal state and invariants]

  Invariants:
      - [List any class invariants that must hold throughout object lifetime]
      - [Example: self._cache is never None after __init__]
      - [Example: self._count is always >= 0]

  Thread Safety:
      - [Thread-safe if all methods are thread-safe]
      - [Not thread-safe: describe which methods are unsafe]
      - [Conditionally safe: describe locking strategy]

  Lifecycle:
      - [Describe object lifecycle: creation, usage, cleanup]
      - [Mention if cleanup is automatic or requires explicit close()]

  Example:
      >>> coordinator = ChunkingCoordinator(
      ...     lifecycle=JobLifecycleManager(),
      ...     chunker=ChunkingService(),
      ...     config=CoordinatorConfig(name="chunking")
      ... )
      >>> result = coordinator.execute(ChunkingRequest(...))
      >>> print(f"Processed {len(result.chunks)} chunks")
  """
  ```

### 2.3 Function/Method Docstring Template

- [x] 2.3.1 **Create `templates/function_docstring.py`** with template:

  ```python
  """[One-line imperative summary: 'Extract text' not 'Extracts text'].

  [Detailed explanation including:
   - What the function does step-by-step
   - Why it exists (what problem it solves)
   - Algorithm or approach used
   - Edge cases handled
   - Performance characteristics if relevant]

  Args:
      param_name: [Describe purpose, valid values, constraints]
          [Additional lines for complex parameters]
          [Example: Must be non-empty string matching pattern '^NCT\\d{8}$']
      optional_param: [Describe purpose and what None means]. Defaults to None.
      **kwargs: [Describe what keyword arguments are accepted if using **kwargs]

  Returns:
      [Describe return value structure and meaning]
      [For complex return types, describe structure]
      [Example: Tuple of (chunks: list[Chunk], metadata: dict[str, Any])]
      [Specify what None return means if applicable]

  Raises:
      ExceptionType: [When and why this exception is raised]
          [Include conditions that trigger it]
          [Include what the exception message will contain]
      AnotherException: [When this occurs]

  Note:
      [Any important implementation notes]
      [Performance considerations: "O(n) time, O(1) space"]
      [Thread safety: "Not thread-safe due to shared cache"]
      [Side effects: "Emits 'chunking.started' metric"]

  Warning:
      [Any gotchas or surprising behavior users should know about]
      [Example: "May return empty list if document has no parseable text"]

  Example:
      >>> text = coordinator._extract_text(job_id, request)
      >>> assert len(text) > 0
      >>> # Raises InvalidDocumentError if text is empty
  """
  ```

### 2.4 Additional Templates

- [x] 2.4.1 **Create `templates/dataclass_docstring.py`** showing dataclass with field documentation
- [x] 2.4.2 **Create `templates/protocol_docstring.py`** showing Protocol/ABC with interface contract documentation
- [x] 2.4.3 **Create `templates/exception_handler_docstring.py`** showing exception handling with inline comments
- [x] 2.4.4 **Create `templates/async_docstring.py`** showing async function documentation
- [x] 2.4.5 **Create `templates/decorator_docstring.py`** showing decorator documentation
- [x] 2.4.6 **Create `templates/property_docstring.py`** showing @property documentation
- [x] 2.4.7 **Create `templates/constant_docstring.py`** showing module-level constant documentation
- [x] 2.4.8 **Create `templates/test_docstring.py`** showing test function documentation format

### 2.5 Cross-Reference Standards

- [x] 2.5.1 **Create `templates/cross_reference_guide.md`** documenting how to use Sphinx-style cross-references:
  - `:class:`ClassName`` for classes
  - `:func:`function_name`` for functions
  - `:meth:`ClassName.method_name`` for methods
  - `:mod:`module_name`` for modules
  - `:exc:`ExceptionName`` for exceptions
  - `:data:`CONSTANT_NAME`` for constants
  - `:attr:`ClassName.attribute_name`` for attributes

## 3. Section Header Standards

- [x] 3.1 **Create `section_headers.md`** with canonical section headers and ordering rules

- [x] 3.2 **Define Gateway Coordinator module structure** in `section_headers.md`:

  ```python
  # ============================================================================
  # IMPORTS
  # ============================================================================
  # (stdlib imports)
  # (blank line)
  # (third-party imports)
  # (blank line)
  # (first-party imports from Medical_KG_rev)
  # (blank line)
  # (relative imports)

  # ============================================================================
  # REQUEST/RESPONSE MODELS
  # ============================================================================
  # (Dataclasses for request and result types used by coordinator)

  # ============================================================================
  # COORDINATOR IMPLEMENTATION
  # ============================================================================
  # (Main coordinator class with __init__ and public execute method)

  # ============================================================================
  # PRIVATE HELPERS
  # ============================================================================
  # (Private methods for text extraction, metadata handling, etc.)

  # ============================================================================
  # ERROR TRANSLATION
  # ============================================================================
  # (Methods for translating exceptions to coordinator errors)

  # ============================================================================
  # EXPORTS
  # ============================================================================
  # (__all__ list)
  ```

- [x] 3.3 **Define Service Layer module structure** in `section_headers.md`:

  ```python
  # ============================================================================
  # IMPORTS
  # ============================================================================

  # ============================================================================
  # TYPE DEFINITIONS & CONSTANTS
  # ============================================================================

  # ============================================================================
  # SERVICE CLASS DEFINITION
  # ============================================================================

  # ============================================================================
  # CHUNKING ENDPOINTS
  # ============================================================================

  # ============================================================================
  # EMBEDDING ENDPOINTS
  # ============================================================================

  # ============================================================================
  # RETRIEVAL ENDPOINTS
  # ============================================================================

  # ============================================================================
  # ADAPTER MANAGEMENT ENDPOINTS
  # ============================================================================

  # ============================================================================
  # VALIDATION ENDPOINTS
  # ============================================================================

  # ============================================================================
  # EXTRACTION ENDPOINTS
  # ============================================================================

  # ============================================================================
  # ADMIN & UTILITY ENDPOINTS
  # ============================================================================

  # ============================================================================
  # PRIVATE HELPERS
  # ============================================================================
  ```

- [x] 3.4 **Define Policy/Strategy module structure** in `section_headers.md`:

  ```python
  # ============================================================================
  # IMPORTS
  # ============================================================================

  # ============================================================================
  # DATA MODELS
  # ============================================================================

  # ============================================================================
  # INTERFACES (Protocols/ABCs)
  # ============================================================================

  # ============================================================================
  # IMPLEMENTATIONS
  # ============================================================================

  # ============================================================================
  # FACTORY FUNCTIONS
  # ============================================================================

  # ============================================================================
  # EXPORTS
  # ============================================================================
  ```

- [x] 3.5 **Define Orchestration module structure** in `section_headers.md`:

  ```python
  # ============================================================================
  # IMPORTS
  # ============================================================================

  # ============================================================================
  # STAGE CONTEXT DATA MODELS
  # ============================================================================

  # ============================================================================
  # STAGE IMPLEMENTATIONS
  # ============================================================================
  # (Grouped by pipeline phase: metadata, PDF, chunk, embed, index)

  # ============================================================================
  # PLUGIN REGISTRATION
  # ============================================================================

  # ============================================================================
  # EXPORTS
  # ============================================================================
  ```

- [x] 3.6 **Define Test module structure** in `section_headers.md`:

  ```python
  # ============================================================================
  # IMPORTS
  # ============================================================================

  # ============================================================================
  # FIXTURES
  # ============================================================================

  # ============================================================================
  # UNIT TESTS - [ComponentName]
  # ============================================================================

  # ============================================================================
  # INTEGRATION TESTS
  # ============================================================================

  # ============================================================================
  # HELPER FUNCTIONS
  # ============================================================================
  ```

- [x] 3.7 **Document ordering rules within sections** in `section_headers.md`:
  - **Imports:** stdlib, third-party, first-party, relative (each group alphabetically sorted)
  - **Classes:** Base classes before subclasses, interfaces before implementations
  - **Class methods:** `__init__` first, public methods (alphabetically), private methods (alphabetically), static/class methods last
  - **Functions:** Public functions before private functions, alphabetical within each group

## 4. Structural Refactor - Chunking Coordinator

### 4.1 Fix Duplicate Imports

- [x] 4.1.1 **In `src/Medical_KG_rev/gateway/coordinators/chunking.py`:**
  - Remove duplicate import of `DocumentChunk` on line 21 (keep line 18)
  - Remove duplicate import of `record_chunking_failure` on line 22 (keep line 19)
  - Keep import group on lines 20-23 that includes ChunkCommand, remove earlier import of ChunkingService without ChunkCommand
  - Ensure imports follow ordering: stdlib → Medical_KG_rev.chunking → Medical_KG_rev.gateway → Medical_KG_rev.observability → Medical_KG_rev.services → relative imports

### 4.2 Insert Section Headers

- [x] 4.2.1 **In `chunking.py`:** Insert section headers following template from `section_headers.md`:
  - Insert `# IMPORTS` header after module docstring (after line 1)
  - Insert `# REQUEST/RESPONSE MODELS` header before ChunkingRequest dataclass (~line 36)
  - Insert `# COORDINATOR IMPLEMENTATION` header before ChunkingCoordinator class (~line 51)
  - Insert `# ERROR TRANSLATION` header before _translate_error method
  - Insert `# EXPORTS` header before `__all__` (~line 349)

### 4.3 Resolve Duplicate Code - Command Creation

- [x] 4.3.1 **Identify canonical implementation:** Lines 77-84 create ChunkCommand (newer interface), lines 85-91 create ChunkingOptions (older)
- [x] 4.3.2 **Delete ChunkingOptions code block** (lines 85-91 with variables: metadata, options)
- [x] 4.3.3 **Delete unused helper** `_metadata_without_text` (lines 289-293) if only used by deleted ChunkingOptions code
- [x] 4.3.4 **Verify ChunkCommand is used** throughout exception handling and error translation

### 4.4 Resolve Duplicate Code - Exception Handling

- [x] 4.4.1 **Keep first exception handling block** (lines 95-119) that:
  - Uses `self._chunker.chunk(command)` with ChunkCommand
  - Catches exceptions and calls `self._translate_error(job_id, command, exc)`
  - Is cleaner and integrates with ChunkingErrorTranslator

- [x] 4.4.2 **Delete second exception handling block** (lines 120-210) that:
  - Uses `self._chunker.chunk(request.tenant_id, request.document_id, text, options)` with old interface
  - Manually creates ProblemDetail instances
  - Has redundant error handling

- [x] 4.4.3 **Verify all exception types are covered:** Check that ChunkingErrorTranslator handles all exceptions that can be raised by ChunkingService

### 4.5 Resolve Duplicate Code - Text Extraction

- [x] 4.5.1 **Keep first `_extract_text` implementation** (lines 239-249) that:
  - Checks request.text first
  - Falls back to request.options["text"]
  - Raises InvalidDocumentError cleanly

- [x] 4.5.2 **Delete second `_extract_text` implementation** (lines 275-287) that:
  - Has similar logic but creates ProblemDetail manually
  - References lifecycle.mark_failed inline

### 4.6 Resolve Duplicate Code - Error Recording

- [x] 4.6.1 **Keep `_record_failure` method** (lines 320-335) that:
  - Accepts ChunkCommand and ChunkingErrorReport
  - Extracts profile from command.options
  - Calls record_chunking_failure metric
  - Calls lifecycle.mark_failed

- [x] 4.6.2 **Delete `_error` method** (lines 295-318) that:
  - Manually constructs ProblemDetail
  - Has similar error recording logic
  - Is superseded by _translate_error → _record_failure flow

- [x] 4.6.3 **Verify all call sites** use `_translate_error` which internally calls `_record_failure`

### 4.7 Add Comprehensive Docstrings

- [x] 4.7.1 **Add module docstring to `chunking.py`** using template from `templates/module_docstring.py`:
  - Explain chunking coordinator's role: gateway layer component that coordinates synchronous chunking jobs
  - Key responsibilities: job lifecycle management, request validation, error translation, metrics emission
  - Collaborators: Upstream (gateway services), Downstream (ChunkingService, JobLifecycleManager, ChunkingErrorTranslator)
  - Side effects: Creates job entries, emits metrics, logs errors
  - Thread safety: Not thread-safe (not designed for concurrent use)

- [x] 4.7.2 **Add docstring to ChunkingRequest dataclass:**
  - Describe each field with valid ranges
  - `document_id`: Unique identifier for document being chunked
  - `text`: Optional document text (can also be in options["text"])
  - `strategy`: Chunking strategy name (e.g., "section", "semantic"), defaults to "section"
  - `chunk_size`: Maximum tokens per chunk, defaults to profile setting
  - `overlap`: Token overlap between chunks, defaults to profile setting
  - `options`: Additional metadata and configuration

- [x] 4.7.3 **Add docstring to ChunkingResult dataclass:**
  - Describe fields: `chunks` (sequence of DocumentChunk objects with content and metadata), `job_id`, `duration_s`, `metadata`

- [x] 4.7.4 **Add docstring to ChunkingCoordinator class:**
  - Describe role: Coordinates synchronous chunking operations by managing job lifecycle, delegating to ChunkingService, and translating errors
  - Attributes: `_lifecycle` (JobLifecycleManager), `_chunker` (ChunkingService), `_errors` (ChunkingErrorTranslator)
  - Example usage showing instantiation and execute call

- [x] 4.7.5 **Add docstring to `__init__` method:**
  - Args: lifecycle (JobLifecycleManager for tracking jobs), chunker (ChunkingService for actual chunking), config (CoordinatorConfig with name and settings), errors (Optional error translator, auto-created if not provided)

- [x] 4.7.6 **Add docstring to `_execute` method:**
  - Describe full flow: create job → extract text → create command → call chunker → handle exceptions → assemble chunks → mark completed → return result
  - Args: request (ChunkingRequest with document and chunking parameters)
  - Returns: ChunkingResult with chunks and metadata
  - Raises: CoordinatorError (for all handled errors after translation)
  - Note: Emits metrics for failures, updates job lifecycle

- [x] 4.7.7 **Add docstring to `_extract_text` method:**
  - Describe: Extracts document text from request, checking request.text first then request.options["text"]
  - Args: job_id (for error reporting), request (ChunkingRequest)
  - Returns: str (non-empty document text)
  - Raises: InvalidDocumentError (if no valid text found)

- [x] 4.7.8 **Add docstring to `_translate_error` method:**
  - Describe: Translates chunking exceptions to coordinator errors using ChunkingErrorTranslator
  - Args: job_id, command (ChunkCommand for context), exc (Exception to translate)
  - Returns: CoordinatorError with problem detail and context
  - Note: Calls _record_failure internally to update metrics and lifecycle

- [x] 4.7.9 **Add docstring to `_record_failure` method:**
  - Describe: Records chunking failure by emitting metrics and updating job lifecycle
  - Args: job_id, command (for extracting profile), report (ChunkingErrorReport with problem details)
  - Returns: None
  - Note: Side effects include metric emission and lifecycle update

### 4.8 Add Type Hints

- [x] 4.8.1 **Verify all functions have return type annotations** in `chunking.py`
- [x] 4.8.2 **Replace bare `dict` with `Mapping` or `MutableMapping`** from `collections.abc`
- [x] 4.8.3 **Add generic parameters** to collections (e.g., `list[DocumentChunk]`, `dict[str, Any]`)
- [x] 4.8.4 **Ensure `-> None` on all procedures** (functions that don't return values)

### 4.9 Add Inline Comments

- [x] 4.9.1 **Add comment before exception handling block** explaining exception translation strategy:

  ```python
  # Attempt chunking and translate any failures to coordinator errors.
  # ChunkingErrorTranslator maps chunking exceptions to HTTP problem details
  # with appropriate status codes, retry hints, and user-facing messages.
  ```

- [x] 4.9.2 **Add comment in chunk assembly loop** explaining metadata merging:

  ```python
  # Merge chunk metadata with standard fields (granularity, chunker).
  # Preserve chunk-specific metadata while ensuring required fields present.
  ```

- [x] 4.9.3 **Add comment in `_extract_text`** explaining dual source logic:

  ```python
  # Text can be provided in request.text or request.options["text"].
  # Check request.text first for backwards compatibility, then fall back to options.
  ```

## 5. Structural Refactor - Embedding Coordinator

- [x] 5.1 **Read and audit `src/Medical_KG_rev/gateway/coordinators/embedding.py`:** Look for duplicate imports, duplicate code blocks, missing docstrings; document findings in `audit.md` under "Embedding Coordinator Audit"

- [x] 5.2 **Insert section headers** following same structure as chunking.py: IMPORTS, REQUEST/RESPONSE MODELS, COORDINATOR IMPLEMENTATION, ERROR TRANSLATION, EXPORTS

- [x] 5.3 **Resolve any duplicate code** following same process as chunking.py: identify canonical implementation, delete duplicate, update references

- [x] 5.4 **Add comprehensive docstrings** using templates: module, dataclasses, class, methods (following same structure as chunking.py documentation)

- [x] 5.5 **Add type hints** ensuring all functions have return types, collections have generic parameters

- [x] 5.6 **Add inline comments** to complex logic: exception handling, result assembly, error translation

## 6. Structural Refactor - Base Coordinator

- [x] 6.1 **Read and audit `src/Medical_KG_rev/gateway/coordinators/base.py`** for missing docstrings

- [x] 6.2 **Insert section headers:** IMPORTS, DATA MODELS, BASE COORDINATOR INTERFACE, METRICS, EXPORTS

- [x] 6.3 **Add comprehensive docstrings:**
  - Module: Explain base coordinator abstractions
  - CoordinatorConfig: Document configuration dataclass
  - CoordinatorRequest/CoordinatorResult: Document base request/result interfaces
  - BaseCoordinator: Document abstract base class, generic parameters, template method pattern
  - CoordinatorMetrics: Document metrics interface

## 7. Structural Refactor - Job Lifecycle Manager

- [x] 7.1 **Read and audit `src/Medical_KG_rev/gateway/coordinators/job_lifecycle.py`**

- [x] 7.2 **Insert section headers:** IMPORTS, JOB STATE DATA MODEL, LIFECYCLE MANAGER, EXPORTS

- [x] 7.3 **Add comprehensive docstrings:**
  - Module: Explain job lifecycle tracking
  - JobLifecycleManager: Document job state management, storage backend, state transitions
  - Methods: Document create_job, mark_completed, mark_failed, update_metadata with state transition semantics

## 8. Structural Refactor - Gateway Services Layer

### 8.1 Audit and Document Duplicates

- [ ] 8.1.1 **Run duplicate import check:** Use `rg "from.*import (ProblemDetail|UCUMValidator|FHIRValidator)" src/Medical_KG_rev/gateway/services.py -n` to find all imports; document in `audit.md`

- [ ] 8.1.2 **Identify canonical import sources:**
  - ProblemDetail: `from ..utils.errors import ProblemDetail`
  - UCUMValidator: `from ..validation import UCUMValidator`
  - FHIRValidator: `from ..validation.fhir import FHIRValidator`

### 8.2 Fix Duplicate Imports

- [ ] 8.2.1 **Remove duplicate imports** keeping only canonical sources identified above

- [ ] 8.2.2 **Organize imports** in groups: stdlib, third-party, first-party Medical_KG_rev (alphabetically sorted), relative

### 8.3 Insert Section Headers

- [ ] 8.3.1 **Add section headers to `services.py`:**
  - `# IMPORTS` at top
  - `# TYPE DEFINITIONS & CONSTANTS` before any module-level constants or type aliases
  - `# SERVICE CLASS DEFINITION` before GatewayService class
  - `# INITIALIZATION & SETUP` for **init** and setup methods
  - `# CHUNKING ENDPOINTS` before chunking-related methods
  - `# EMBEDDING ENDPOINTS` before embedding-related methods
  - `# RETRIEVAL ENDPOINTS` before retrieval-related methods
  - `# ADAPTER MANAGEMENT ENDPOINTS` before adapter CRUD methods
  - `# VALIDATION ENDPOINTS` before UCUM/FHIR validation methods
  - `# EXTRACTION ENDPOINTS` before extraction methods
  - `# ADMIN & UTILITY ENDPOINTS` before health/status methods
  - `# PRIVATE HELPERS` before private methods

### 8.4 Reorder Methods

- [ ] 8.4.1 **Within each section, order methods alphabetically** (except **init** which stays first in class)

- [ ] 8.4.2 **Move all private helper methods** to PRIVATE HELPERS section at end of class

- [ ] 8.4.3 **Ensure coordinator integration methods** are in correct sections:
  - Methods calling ChunkingCoordinator → CHUNKING ENDPOINTS
  - Methods calling EmbeddingCoordinator → EMBEDDING ENDPOINTS

### 8.5 Add Docstrings

- [ ] 8.5.1 **Add module docstring** explaining:
  - Role: Protocol-agnostic service layer between protocol handlers (REST/GraphQL/gRPC) and domain logic
  - Responsibilities: Request coordination, error translation, metrics emission, audit logging
  - Pattern: Façade pattern over coordinators and domain services

- [ ] 8.5.2 **Add class docstring** to GatewayService explaining:
  - Purpose: Single entry point for all gateway operations
  - Attributes: List all injected dependencies (coordinators, validators, etc.)
  - Thread safety: Document concurrency characteristics

- [ ] 8.5.3 **Add method docstrings** for all public methods with Args, Returns, Raises, Example

- [ ] 8.5.4 **Add docstrings for private helpers** explaining why they exist and what they do

### 8.6 Add Inline Comments

- [ ] 8.6.1 **Add comments explaining multi-step operations** in complex methods

- [ ] 8.6.2 **Add comments explaining error handling strategies** where exceptions are caught

- [ ] 8.6.3 **Add comments explaining metric emission points** with metric names

## 9. Structural Refactor - Retrieval Chunking Service

### 9.1 Audit

- [ ] 9.1.1 **Read `src/Medical_KG_rev/services/retrieval/chunking.py`** and check if ChunkCommand is defined here or in separate file

- [ ] 9.1.2 **Check for validation helpers** and their location relative to ChunkCommand

- [ ] 9.1.3 **Check for missing docstrings** and document in `audit.md`

### 9.2 Insert Section Headers

- [ ] 9.2.1 **Add section headers:**
  - `# IMPORTS`
  - `# COMMAND MODELS` (if ChunkCommand defined here)
  - `# VALIDATION HELPERS` (if present)
  - `# SERVICE IMPLEMENTATION`
  - `# EXPORTS`

### 9.3 Co-locate Related Code

- [ ] 9.3.1 **If ChunkCommand is in separate file** (`chunking_command.py`), evaluate whether to:
  - Move it to `chunking.py` for better cohesion, OR
  - Keep separate but add cross-references in docstrings

- [ ] 9.3.2 **Ensure validation helpers** are immediately adjacent to ChunkCommand dataclass for clarity

### 9.4 Add Docstrings

- [ ] 9.4.1 **Add module docstring** explaining:
  - Role: Service layer adapter to chunking library
  - Responsibilities: Profile loading, chunker configuration, chunk execution, error handling

- [ ] 9.4.2 **Add ChunkCommand dataclass docstring** with field documentation including valid value ranges

- [ ] 9.4.3 **Add ChunkingService class docstring** explaining:
  - Purpose: Facade over chunking library
  - Methods: chunk() for synchronous chunking, available_strategies() for listing strategies
  - Configuration: How profiles are loaded and applied

- [ ] 9.4.4 **Add method docstrings** with Args, Returns, Raises

### 9.5 Add Type Hints and Comments

- [ ] 9.5.1 **Verify type hints** on all functions
- [ ] 9.5.2 **Add inline comments** explaining chunking algorithm selection and profile application

## 10. Structural Refactor - Embedding Services

### 10.1 Audit Embedding Modules

- [ ] 10.1.1 **Read all embedding service modules:**
  - `src/Medical_KG_rev/services/embedding/policy.py`
  - `src/Medical_KG_rev/services/embedding/persister.py`
  - `src/Medical_KG_rev/services/embedding/telemetry.py`
  - `src/Medical_KG_rev/services/embedding/registry.py`
  - `src/Medical_KG_rev/services/embedding/namespace/access.py`
  - `src/Medical_KG_rev/services/embedding/namespace/registry.py`

- [ ] 10.1.2 **Check for duplicates, missing docstrings, inconsistent structure** and document in `audit.md` under "Embedding Services Audit"

### 10.2 Refactor Policy Module

- [ ] 10.2.1 **Insert section headers in `policy.py`:**
  - `# IMPORTS`
  - `# DATA MODELS` (NamespaceAccessDecision, NamespacePolicySettings, _CacheEntry)
  - `# INTERFACE` (NamespaceAccessPolicy ABC)
  - `# IMPLEMENTATIONS` (concrete policy classes)
  - `# FACTORY FUNCTIONS` (build_policy_chain)
  - `# EXPORTS`

- [ ] 10.2.2 **Add comprehensive docstrings:**
  - Module: Explain namespace access policy system
  - NamespaceAccessDecision: Document decision dataclass fields
  - NamespaceAccessPolicy: Document abstract interface, cache strategy, evaluation flow
  - Concrete implementations: Document policy logic and when to use each

### 10.3 Refactor Persister Module

- [ ] 10.3.1 **Insert section headers in `persister.py`:**
  - `# IMPORTS`
  - `# DATA MODELS` (PersistenceContext, PersisterRuntimeSettings)
  - `# INTERFACE` (EmbeddingPersister protocol/ABC)
  - `# IMPLEMENTATIONS` (concrete persisters)
  - `# FACTORY FUNCTIONS` (build_persister)
  - `# EXPORTS`

- [ ] 10.3.2 **Add comprehensive docstrings:**
  - Module: Explain embedding persistence abstraction
  - PersistenceContext: Document context fields
  - EmbeddingPersister: Document interface contract
  - Implementations: Document storage backend specifics

### 10.4 Refactor Telemetry Module

- [ ] 10.4.1 **Insert section headers in `telemetry.py`:**
  - `# IMPORTS`
  - `# INTERFACE` (EmbeddingTelemetry protocol/ABC)
  - `# IMPLEMENTATIONS` (StandardEmbeddingTelemetry)
  - `# EXPORTS`

- [ ] 10.4.2 **Add comprehensive docstrings:**
  - Module: Explain telemetry abstraction
  - EmbeddingTelemetry: Document interface and metrics emitted
  - StandardEmbeddingTelemetry: Document Prometheus integration

### 10.5 Refactor Registry Module

- [ ] 10.5.1 **Insert section headers in `registry.py`:**
  - `# IMPORTS`
  - `# REGISTRY IMPLEMENTATION`
  - `# EXPORTS`

- [ ] 10.5.2 **Add comprehensive docstrings:**
  - Module: Explain embedding model registry
  - EmbeddingModelRegistry: Document model registration, lookup, and namespace mapping

### 10.6 Document Embedding Pipeline Flow

- [ ] 10.6.1 **Create `docs/guides/embedding_pipeline_flow.md`** with:
  - Mermaid diagram showing flow: Request → Policy Check → Namespace Resolution → Model Selection → Embedding Generation → Persistence → Telemetry
  - Detailed explanation of each stage
  - Error handling at each stage
  - Example request/response

- [ ] 10.6.2 **Reference pipeline flow document** in each embedding module docstring

### 10.7 Add Type Hints and Comments

- [ ] 10.7.1 **Verify type hints** in all embedding modules
- [ ] 10.7.2 **Add inline comments** explaining policy evaluation logic, caching strategies, persistence operations

## 11. Structural Refactor - Orchestration Modules

### 11.1 Audit Orchestration Modules

- [ ] 11.1.1 **Read orchestration modules:**
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py`
  - `src/Medical_KG_rev/orchestration/dagster/stages.py`
  - `src/Medical_KG_rev/orchestration/stages/contracts.py`
  - `src/Medical_KG_rev/orchestration/stages/plugins.py`
  - `src/Medical_KG_rev/orchestration/stages/plugin_manager.py`
  - `src/Medical_KG_rev/orchestration/stages/plugins/builtin.py`

- [ ] 11.1.2 **Document duplicates, missing docstrings, structure issues** in `audit.md` under "Orchestration Audit"

### 11.2 Refactor Contracts Module

- [ ] 11.2.1 **Insert section headers in `contracts.py`:**
  - `# IMPORTS`
  - `# STAGE CONTEXT DATA MODELS`
  - `# STAGE RESULT DATA MODELS`
  - `# STAGE INTERFACE` (Protocol/ABC)
  - `# EXPORTS`

- [ ] 11.2.2 **Add docstrings:**
  - Module: Explain stage contracts and data flow
  - StageContext: Document fields passed between stages
  - StageResult: Document result structure
  - Stage interface: Document protocol for implementing stages

### 11.3 Refactor Plugins Module

- [ ] 11.3.1 **Insert section headers in `plugins.py`:**
  - `# IMPORTS`
  - `# PLUGIN REGISTRY`
  - `# PLUGIN DECORATOR`
  - `# EXPORTS`

- [ ] 11.3.2 **Add docstrings:**
  - Module: Explain plugin registration system
  - Plugin decorator: Document how to register stages
  - Registry: Document stage discovery

### 11.4 Refactor Builtin Plugins

- [ ] 11.4.1 **Insert section headers in `plugins/builtin.py`:**
  - `# IMPORTS`
  - `# METADATA EXTRACTION STAGES`
  - `# PDF PROCESSING STAGES`
  - `# CHUNKING STAGES`
  - `# EMBEDDING STAGES`
  - `# INDEXING STAGES`
  - `# VALIDATION STAGES`
  - `# PLUGIN REGISTRATION`

- [ ] 11.4.2 **Order stage implementations by pipeline phase:** metadata → PDF → chunk → embed → index

- [ ] 11.4.3 **Add docstrings to each stage function:**
  - Describe what stage does
  - Document inputs (StageContext fields used)
  - Document outputs (StageResult fields produced)
  - Document side effects (external API calls, file I/O)
  - Add example

### 11.5 Refactor Plugin Manager

- [ ] 11.5.1 **Insert section headers in `plugin_manager.py`:**
  - `# IMPORTS`
  - `# PLUGIN MANAGER IMPLEMENTATION`
  - `# PLUGIN DISCOVERY`
  - `# EXPORTS`

- [ ] 11.5.2 **Add docstrings:**
  - Module: Explain plugin management
  - PluginManager: Document how plugins are loaded and retrieved

### 11.6 Refactor Dagster Modules

- [ ] 11.6.1 **Insert section headers in `dagster/stages.py`:**
  - `# IMPORTS`
  - `# DAGSTER OP WRAPPERS`
  - `# PIPELINE CONSTRUCTION`
  - `# EXPORTS`

- [ ] 11.6.2 **Insert section headers in `dagster/runtime.py`:**
  - `# IMPORTS`
  - `# DAGSTER RUNTIME CONFIGURATION`
  - `# JOB SUBMISSION`
  - `# EXPORTS`

- [ ] 11.6.3 **Add comprehensive docstrings** explaining Dagster integration, op wrapping, job submission

## 12. Structural Refactor - Error Translation

### 12.1 Audit

- [ ] 12.1.1 **Read `src/Medical_KG_rev/gateway/chunking_errors.py`** and check for missing docstrings, incomplete error mappings

### 12.2 Insert Section Headers

- [ ] 12.2.1 **Add section headers:**
  - `# IMPORTS`
  - `# ERROR REPORT DATA MODEL`
  - `# ERROR TRANSLATOR IMPLEMENTATION`
  - `# ERROR MAPPING HELPERS`
  - `# EXPORTS`

### 12.3 Add Docstrings

- [ ] 12.3.1 **Add module docstring** explaining error translation strategy: maps chunking library exceptions to HTTP problem details

- [ ] 12.3.2 **Add ChunkingErrorReport docstring** documenting all fields: problem (ProblemDetail), severity, metric, job_id

- [ ] 12.3.3 **Add ChunkingErrorTranslator docstring** explaining:
  - Purpose: Centralized exception-to-HTTP error mapping
  - Strategy: Maps each chunking exception type to status code, problem type, retry strategy
  - Extensibility: How to add new exception mappings

- [ ] 12.3.4 **Add docstrings to translate method** with Args, Returns (ChunkingErrorReport or None)

### 12.4 Create Error Translation Decision Table

- [ ] 12.4.1 **Create `docs/error_translation_matrix.md`** with table:

  | Exception Type | HTTP Status | Problem Type | Retry Strategy | Metric Name |
  |----------------|-------------|--------------|----------------|-------------|
  | ProfileNotFoundError | 400 | profile-not-found | No retry | ProfileNotFoundError |
  | TokenizerMismatchError | 500 | tokenizer-mismatch | No retry | TokenizerMismatchError |
  | ChunkingFailedError | 500 | chunking-failed | No retry | ChunkingFailedError |
  | InvalidDocumentError | 400 | invalid-document | No retry | InvalidDocumentError |
  | ChunkerConfigurationError | 422 | invalid-configuration | No retry | ChunkerConfigurationError |
  | ChunkingUnavailableError | 503 | service-unavailable | Retry with backoff | ChunkingUnavailableError |
  | MineruOutOfMemoryError | 503 | gpu-oom | Retry after cooldown | MineruOutOfMemoryError |
  | MineruGpuUnavailableError | 503 | gpu-unavailable | Retry after cooldown | MineruGpuUnavailableError |
  | MemoryError | 503 | resource-exhausted | Retry after 60s | MemoryError |
  | TimeoutError | 503 | timeout | Retry after 30s | TimeoutError |

- [ ] 12.4.2 **Reference decision table** in ChunkingErrorTranslator docstring

## 13. Test Module Refactoring

### 13.1 Audit Test Modules

- [ ] 13.1.1 **Identify all test modules** for pipeline code:
  - `tests/gateway/test_chunking_coordinator.py` (if exists)
  - `tests/gateway/test_embedding_coordinator.py` (if exists)
  - `tests/gateway/test_services.py` (if exists)
  - `tests/gateway/test_job_lifecycle.py` (if exists)
  - `tests/services/test_chunking.py` (if exists)
  - `tests/services/test_embedding_policy.py` (if exists)
  - `tests/services/test_embedding_persister.py` (if exists)
  - `tests/orchestration/test_stage_plugins.py` (if exists)
  - `tests/orchestration/test_plugin_manager.py` (if exists)

- [ ] 13.1.2 **Document missing docstrings, unclear fixtures, poorly named tests** in `audit.md` under "Test Documentation Issues"

### 13.2 Insert Section Headers in Test Modules

- [ ] 13.2.1 **For each test module, add section headers:**
  - `# IMPORTS`
  - `# FIXTURES`
  - `# UNIT TESTS - [ComponentName]`
  - `# INTEGRATION TESTS` (if applicable)
  - `# HELPER FUNCTIONS`

### 13.3 Add Test Documentation

- [ ] 13.3.1 **Add module docstring** to each test module explaining what component is under test

- [ ] 13.3.2 **Add fixture docstrings** explaining what fixture provides and typical usage pattern

- [ ] 13.3.3 **Add test function docstrings** following format: "Test that [component] [behavior] when [condition]."
  - Examples:
    - "Test that ChunkingCoordinator raises CoordinatorError when ChunkingService raises ProfileNotFoundError."
    - "Test that EmbeddingPolicy denies access when tenant is not in allowed list."
    - "Test that ChunkingService returns chunks when given valid document text."

### 13.4 Standardize Test Naming

- [ ] 13.4.1 **Ensure all test functions follow naming pattern:** `test_<component>_<behavior>_<condition>`
  - Examples:
    - `test_chunking_coordinator_raises_error_when_profile_not_found`
    - `test_embedding_policy_denies_access_when_tenant_not_allowed`
    - `test_chunking_service_returns_chunks_when_valid_text_provided`

- [ ] 13.4.2 **Rename tests** that don't follow this pattern

### 13.5 Add Inline Comments to Test Setup

- [ ] 13.5.1 **Add comments explaining mock configurations** when mocks are created with specific behaviors

- [ ] 13.5.2 **Add comments explaining assertion rationale** for non-obvious checks

## 14. Tooling & Enforcement

### 14.1 Configure Ruff for Docstrings

- [x] 14.1.1 **In `pyproject.toml`, add docstring rules:**

  ```toml
  [tool.ruff.lint]
  select = [
      "D",      # pydocstyle
      "D100",   # Missing docstring in public module
      "D101",   # Missing docstring in public class
      "D102",   # Missing docstring in public method
      "D103",   # Missing docstring in public function
      "D104",   # Missing docstring in public package
      "D105",   # Missing docstring in magic method
      "D107",   # Missing docstring in __init__
  ]

  [tool.ruff.lint.pydocstyle]
  convention = "google"

  [tool.ruff.lint.per-file-ignores]
  "tests/**/*.py" = ["D"]  # Exempt tests from docstring requirements
  ```

### 14.2 Create Section Header Checker

- [x] 14.2.1 **Create `scripts/check_section_headers.py`** that:
  - Uses `ast` module to parse Python files
  - Extracts comments matching pattern `# ={10,} SECTION_NAME ={10,}`
  - Validates sections appear in order defined in `section_headers.md`
  - Validates each section contains appropriate AST node types
  - Outputs violations with file:line:message format
  - Exits with non-zero code if violations found

- [x] 14.2.2 **Make script executable:** `chmod +x scripts/check_section_headers.py`

- [x] 14.2.3 **Test script** on refactored files to ensure it correctly identifies section headers

### 14.3 Create Docstring Coverage Checker

- [x] 14.3.1 **Create `scripts/check_docstring_coverage.py`** that:
  - Uses `ast` module to find all modules, classes, functions in specified directories
  - Checks for presence of docstring (first statement is ast.Expr with string value)
  - Calculates coverage: (items with docstrings / total items) × 100
  - Outputs coverage report per file with percentages
  - Fails with exit code 1 if coverage < 90% for files in scope

- [x] 14.3.2 **Make script executable:** `chmod +x scripts/check_docstring_coverage.py`

- [x] 14.3.3 **Test script** on refactored files to verify coverage calculation

### 14.4 Add Pre-Commit Hooks

- [x] 14.4.1 **In `.pre-commit-config.yaml`, add hooks:**

  ```yaml
  - repo: local
    hooks:
      - id: ruff-docstring-check
        name: Check docstrings with ruff
        entry: ruff check --select D
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/(gateway|services|orchestration)/

      - id: section-header-check
        name: Check section headers
        entry: python scripts/check_section_headers.py
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/(gateway|services|orchestration)/

      - id: docstring-coverage
        name: Check docstring coverage
        entry: python scripts/check_docstring_coverage.py --min-coverage 90
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/(gateway|services|orchestration)/
  ```

- [ ] 14.4.2 **Test pre-commit hooks locally:** Run `pre-commit run --all-files` and verify hooks execute correctly

### 14.5 Add CI Workflow

- [x] 14.5.1 **Create or update `.github/workflows/documentation-quality.yml`:**

  ```yaml
  name: Documentation Quality

  on:
    pull_request:
      paths:
        - 'src/Medical_KG_rev/gateway/**'
        - 'src/Medical_KG_rev/services/**'
        - 'src/Medical_KG_rev/orchestration/**'
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
            pip install ruff

        - name: Check docstrings
          run: |
            ruff check --select D src/Medical_KG_rev/gateway src/Medical_KG_rev/services src/Medical_KG_rev/orchestration

        - name: Check section headers
          run: |
            python scripts/check_section_headers.py

        - name: Check docstring coverage
          run: |
            python scripts/check_docstring_coverage.py --min-coverage 90
  ```

### 14.6 Configure Documentation Generation

- [ ] 14.6.1 **In `mkdocs.yml`, configure mkdocstrings plugin:**

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

- [ ] 14.6.2 **Add API documentation pages to nav:**

  ```yaml
  nav:
    - Home: index.md
    - Pipeline API:
      - Coordinators: api/coordinators.md
      - Services: api/services.md
      - Embedding: api/embedding.md
      - Orchestration: api/orchestration.md
  ```

- [ ] 14.6.3 **Create API documentation pages:**
  - `docs/api/coordinators.md`: Use `:::Medical_KG_rev.gateway.coordinators` syntax to include docstrings
  - `docs/api/services.md`: Include GatewayService docstrings
  - `docs/api/embedding.md`: Include embedding service docstrings
  - `docs/api/orchestration.md`: Include orchestration docstrings

### 14.7 Create Remediation Guide

- [ ] 14.7.1 **Create `docs/contributing/documentation_standards.md`** with sections:
  - **Overview:** Why documentation matters, standards we follow
  - **Google-Style Docstrings:** Full explanation with examples for functions, classes, modules
  - **Section Headers:** Required sections and ordering per module type
  - **Running Checks Locally:** How to run ruff, section checker, coverage checker
  - **Interpreting Errors:** Common error messages and how to fix them
  - **Templates:** Link to templates in `openspec/changes/add-pipeline-structure-documentation/templates/`
  - **Examples:** Link to before/after examples

## 15. Validation & Testing

### 15.1 Run Unit Tests

- [ ] 15.1.1 **Run gateway tests:** `pytest tests/gateway/ -v` and verify all tests pass
- [ ] 15.1.2 **Run service tests:** `pytest tests/services/ -v` and verify all tests pass
- [ ] 15.1.3 **Run orchestration tests:** `pytest tests/orchestration/ -v` and verify all tests pass
- [ ] 15.1.4 **If tests fail:** Document failure in `audit.md` under "Test Failures" with:
  - Test name
  - Failure reason
  - Root cause: (a) broken functionality from refactor, or (b) test needs updating
  - Fix required

### 15.2 Run Integration Tests

- [ ] 15.2.1 **Run integration tests:** `pytest tests/integration/ -v` and verify no regressions
- [ ] 15.2.2 **Test multi-stage pipeline flows** that exercise coordinator → service → orchestration integration

### 15.3 Run Contract Tests

- [ ] 15.3.1 **Run contract tests:** `pytest tests/contract/ -v` to ensure API contracts haven't changed
- [ ] 15.3.2 **If contracts changed unintentionally:** Identify breaking changes and revert or document

### 15.4 Manual Smoke Testing

- [ ] 15.4.1 **Start local stack:** `docker-compose up -d`
- [ ] 15.4.2 **Start API gateway:** `python -m Medical_KG_rev.gateway.main`
- [ ] 15.4.3 **Send chunking request:**

  ```bash
  curl -X POST http://localhost:8000/v1/chunk \
    -H "Content-Type: application/json" \
    -d '{"tenant_id": "test", "document_id": "doc1", "text": "Sample document text for chunking."}'
  ```

  Verify response contains chunks

- [ ] 15.4.4 **Send embedding request:**

  ```bash
  curl -X POST http://localhost:8000/v1/embed \
    -H "Content-Type: application/json" \
    -d '{"tenant_id": "test", "texts": ["Sample text to embed"]}'
  ```

  Verify response contains embeddings

- [ ] 15.4.5 **Check logs** for any new errors or warnings introduced by refactoring

### 15.5 Run Documentation Quality Checks

- [ ] 15.5.1 **Run ruff docstring check:**

  ```bash
  ruff check --select D src/Medical_KG_rev/gateway src/Medical_KG_rev/services src/Medical_KG_rev/orchestration
  ```

  Verify 0 errors

- [ ] 15.5.2 **Run section header check:**

  ```bash
  python scripts/check_section_headers.py
  ```

  Verify all files pass

- [ ] 15.5.3 **Run docstring coverage check:**

  ```bash
  python scripts/check_docstring_coverage.py --min-coverage 90
  ```

  Verify coverage ≥ 90%

### 15.6 Generate and Review API Documentation

- [ ] 15.6.1 **Build documentation:** `mkdocs build --strict` and verify build succeeds with no warnings

- [ ] 15.6.2 **Open documentation in browser:** Open `site/index.html` and navigate to API documentation pages

- [ ] 15.6.3 **Verify docstrings render correctly** with proper formatting (headings, code blocks, lists)

- [ ] 15.6.4 **Verify cross-references work** (clicking :class:, :func: links navigates correctly)

### 15.7 Create Before/After Examples

- [ ] 15.7.1 **Create examples directory:** `openspec/changes/add-pipeline-structure-documentation/examples/`

- [ ] 15.7.2 **Create `examples/before_chunking_coordinator.py`:** Copy snippet from current `chunking.py` showing:
  - Missing module docstring
  - Duplicate imports (lines 18-23)
  - Duplicate code blocks (lines 77-91, 95-210)
  - Methods without docstrings

- [ ] 15.7.3 **Create `examples/after_chunking_coordinator.py`:** Show refactored version with:
  - Comprehensive module docstring
  - Clean imports (no duplicates)
  - Single implementations (duplicates removed)
  - Section headers
  - Comprehensive docstrings on all classes/methods

- [ ] 15.7.4 **Create `examples/before_services.py`:** Show snippet with duplicate imports, no section headers

- [ ] 15.7.5 **Create `examples/after_services.py`:** Show refactored snippet with clean imports, section headers

- [ ] 15.7.6 **Create `examples/README.md`:** Explain improvements demonstrated by before/after examples with specific metrics:
  - Docstring coverage: Before 20% → After 100%
  - Duplicate code blocks removed: [count]
  - Lines reduced: [count]

## 16. Developer Guide & Documentation

### 16.1 Create Pipeline Extension Guide

- [ ] 16.1.1 **Create `docs/guides/pipeline_extension_guide.md`** with sections:

  **Overview:**
  - High-level pipeline architecture: Gateway → Coordinators → Services → Orchestration
  - Data flow: Request → Validation → Execution → Response
  - Error handling strategy: Exception translation, problem details, metrics

  **Adding a New Coordinator:**
  - Step 1: Define request/response dataclasses inheriting from CoordinatorRequest/CoordinatorResult
  - Step 2: Implement coordinator class inheriting from BaseCoordinator[RequestType, ResultType]
  - Step 3: Implement _execute method with job lifecycle integration
  - Step 4: Add error translation logic
  - Step 5: Write comprehensive docstrings following templates
  - Step 6: Add unit tests
  - Code example showing complete coordinator implementation

  **Adding a New Orchestration Stage:**
  - Step 1: Implement stage function accepting StageContext, returning StageResult
  - Step 2: Register stage using @stage_plugin decorator
  - Step 3: Add stage to pipeline configuration YAML
  - Step 4: Write docstring documenting inputs, outputs, side effects
  - Step 5: Add unit tests mocking StageContext
  - Code example

  **Adding a New Embedding Policy:**
  - Step 1: Create class inheriting from NamespaceAccessPolicy
  - Step 2: Implement _evaluate method with policy logic
  - Step 3: Register policy in policy chain via build_policy_chain
  - Step 4: Write docstring documenting policy rules
  - Step 5: Add unit tests for allow/deny scenarios
  - Code example

  **Adding a New Persister:**
  - Step 1: Implement EmbeddingPersister protocol
  - Step 2: Implement persist method with storage backend integration
  - Step 3: Register persister in build_persister factory
  - Step 4: Write docstring documenting storage semantics, consistency guarantees
  - Step 5: Add unit and integration tests
  - Code example

  **Error Handling:**
  - How to add new exception types to error translator
  - How to map exceptions to HTTP problem details
  - How to define retry strategies
  - Code example

  **Testing:**
  - Fixture patterns for coordinators (mock lifecycle, services)
  - Fixture patterns for services (mock external dependencies)
  - Fixture patterns for stages (mock StageContext)
  - Assertion patterns for validating results
  - Code examples

  **Documentation:**
  - How to follow docstring standards (reference templates)
  - How to add section headers
  - How to run documentation checks locally
  - How to generate API documentation

### 16.2 Create Architecture Decision Records

- [ ] 16.2.1 **Create `docs/adr/0001-coordinator-architecture.md`:**
  - Context: Need for separation between protocol handlers and domain logic
  - Decision: Introduce coordinator layer
  - Consequences: Better testability, protocol independence, centralized error handling

- [ ] 16.2.2 **Create `docs/adr/0002-section-headers.md`:**
  - Context: Need for consistent code organization
  - Decision: Mandate section headers in all pipeline modules
  - Consequences: Improved readability, enforceability via linting

- [ ] 16.2.3 **Create `docs/adr/0003-error-translation-strategy.md`:**
  - Context: Need for consistent error responses across protocols
  - Decision: Implement error translator pattern
  - Consequences: Centralized error mapping, easier to update error messages

- [ ] 16.2.4 **Create `docs/adr/0004-google-style-docstrings.md`:**
  - Context: Need for consistent docstring format
  - Decision: Adopt Google-style docstrings
  - Consequences: Better tooling support (mkdocstrings), clear structure

### 16.3 Update OpenSpec Change Trackers

- [ ] 16.3.1 **Mark coordinator documentation complete** in `openspec/changes/add-foundation-infrastructure/tasks.md`

- [ ] 16.3.2 **Mark service layer documentation complete** in `openspec/changes/add-multi-protocol-gateway/tasks.md`

- [ ] 16.3.3 **Mark stage plugin documentation complete** in `openspec/changes/add-ingestion-orchestration/tasks.md`

### 16.4 Create Pipeline Flow Diagrams

- [ ] 16.4.1 **Create `docs/diagrams/chunking_flow.mmd` (Mermaid):**

  ```mermaid
  sequenceDiagram
      participant Client
      participant GatewayService
      participant ChunkingCoordinator
      participant JobLifecycle
      participant ChunkingService
      participant ChunkingLibrary

      Client->>GatewayService: POST /v1/chunk
      GatewayService->>ChunkingCoordinator: execute(request)
      ChunkingCoordinator->>JobLifecycle: create_job(tenant, "chunk")
      JobLifecycle-->>ChunkingCoordinator: job_id
      ChunkingCoordinator->>ChunkingService: chunk(command)
      ChunkingService->>ChunkingLibrary: chunk(text, strategy)
      ChunkingLibrary-->>ChunkingService: chunks
      ChunkingService-->>ChunkingCoordinator: chunks
      ChunkingCoordinator->>JobLifecycle: mark_completed(job_id)
      ChunkingCoordinator-->>GatewayService: ChunkingResult
      GatewayService-->>Client: 200 OK with chunks
  ```

- [ ] 16.4.2 **Create `docs/diagrams/embedding_flow.mmd`** showing embedding pipeline with policy, namespace resolution, model selection, persistence

- [ ] 16.4.3 **Create `docs/diagrams/orchestration_flow.mmd`** showing multi-stage pipeline execution

### 16.5 Document Module Dependencies

- [ ] 16.5.1 **Create `docs/diagrams/module_dependencies.mmd`:**

  ```mermaid
  graph TD
      GatewayService[Gateway Service Layer]
      ChunkCoord[Chunking Coordinator]
      EmbedCoord[Embedding Coordinator]
      ChunkService[Chunking Service]
      EmbedPolicy[Embedding Policy]
      EmbedPersister[Embedding Persister]
      Orchestration[Orchestration Stages]

      GatewayService --> ChunkCoord
      GatewayService --> EmbedCoord
      ChunkCoord --> ChunkService
      EmbedCoord --> EmbedPolicy
      EmbedCoord --> EmbedPersister
      GatewayService --> Orchestration

      style GatewayService fill:#e1f5ff
      style ChunkCoord fill:#ffe1e1
      style EmbedCoord fill:#ffe1e1
      style ChunkService fill:#e1ffe1
      style EmbedPolicy fill:#e1ffe1
      style EmbedPersister fill:#e1ffe1
      style Orchestration fill:#ffe1ff
  ```

- [ ] 16.5.2 **Add layer annotations** showing clear boundaries: Gateway Layer, Coordinator Layer, Service Layer, Orchestration Layer

### 16.6 Create Troubleshooting Guide

- [ ] 16.6.1 **Create `docs/troubleshooting/pipeline_issues.md`** with sections:

  **Chunking Coordinator Errors:**
  - ProfileNotFoundError: Check profile exists in config/chunking/profiles/
  - TokenizerMismatchError: Verify embedding model tokenizer matches chunking tokenizer
  - ChunkingUnavailableError: Check MinerU service health, GPU availability
  - MineruOutOfMemoryError: Reduce batch size or document size

  **Embedding Errors:**
  - Namespace access denied: Check tenant_id in JWT, verify namespace allows tenant
  - Model not found: Verify model registered in embedding model registry
  - Persistence failed: Check vector store connectivity, disk space

  **Orchestration Failures:**
  - Stage timeout: Increase timeout in resilience policy config
  - Retry exhausted: Check dead letter queue, review stage logs
  - Dependency missing: Verify all required services are running

  **Documentation Lint Failures:**
  - D100 missing module docstring: Add module docstring at top of file
  - D101 missing class docstring: Add docstring immediately after class definition
  - D103 missing function docstring: Add docstring immediately after function definition
  - Section header missing: Add required section headers per section_headers.md

## 17. Legacy Decommissioning

### 17.1 Identify Legacy Code

- [ ] 17.1.1 **Search for deprecated markers:**

  ```bash
  rg -t py "@deprecated|warnings.warn.*DeprecationWarning" src/Medical_KG_rev/gateway src/Medical_KG_rev/services src/Medical_KG_rev/orchestration
  ```

- [ ] 17.1.2 **Search for legacy comments:**

  ```bash
  rg -t py "legacy|old|deprecated|superseded|todo.*remove" src/Medical_KG_rev/gateway src/Medical_KG_rev/services src/Medical_KG_rev/orchestration
  ```

- [ ] 17.1.3 **Document findings** in `audit.md` under "Legacy Code to Remove" with table:
  - File | Function/Class | Deprecated Why | Replaced By | References Count

### 17.2 Find References

- [ ] 17.2.1 **For each legacy helper identified, run:**

  ```bash
  rg -t py "legacy_function_name" src/ tests/
  ```

- [ ] 17.2.2 **Document all references** in audit table

- [ ] 17.2.3 **Categorize references:**
  - Can be deleted (unused)
  - Must be updated to use new coordinator architecture
  - Must remain temporarily (backwards compatibility)

### 17.3 Delete Unused Legacy Helpers

- [ ] 17.3.1 **For helpers with 0 references:** Delete immediately

- [ ] 17.3.2 **For each deletion, add entry to `LEGACY_DECOMMISSION_CHECKLIST.md`:**

  ```markdown
  - [x] Removed `old_chunk_handler` from `src/Medical_KG_rev/gateway/legacy.py`
    - Superseded by: ChunkingCoordinator
    - Last used in: N/A (unreferenced)
    - Removed in: add-pipeline-structure-documentation change (2024-10-08)
    - Reason: Superseded by coordinator + command architecture
  ```

### 17.4 Update References

- [ ] 17.4.1 **For helpers still referenced:** Refactor call sites to use ChunkingCoordinator/EmbeddingCoordinator

- [ ] 17.4.2 **Update tests** to use new coordinator interfaces

- [ ] 17.4.3 **Document each migration** in `LEGACY_DECOMMISSION_CHECKLIST.md`

### 17.5 Mark Deprecated Code

- [ ] 17.5.1 **For legacy code that must remain temporarily:** Add deprecation warning:

  ```python
  import warnings

  def old_function():
      """Legacy function, use ChunkingCoordinator instead.

      .. deprecated:: 1.5.0
          Use :class:`ChunkingCoordinator` instead. Will be removed in 2.0.0.
      """
      warnings.warn(
          "old_function is deprecated and will be removed in v2.0. "
          "Use ChunkingCoordinator instead.",
          DeprecationWarning,
          stacklevel=2
      )
      # ... implementation
  ```

### 17.6 Remove Legacy Configuration

- [ ] 17.6.1 **Check `config/` directory** for legacy pipeline configurations

- [ ] 17.6.2 **Remove unused configuration files** and document in `LEGACY_DECOMMISSION_CHECKLIST.md`

### 17.7 Clean Up Legacy Tests

- [ ] 17.7.1 **Identify test files** testing only legacy code

- [ ] 17.7.2 **Delete tests** for removed legacy code

- [ ] 17.7.3 **Update tests** to use new coordinator architecture instead of legacy helpers

### 17.8 Update Documentation

- [ ] 17.8.1 **Search documentation for legacy references:**

  ```bash
  rg "legacy|old|deprecated" docs/
  ```

- [ ] 17.8.2 **Remove references to deleted code** from documentation

- [ ] 17.8.3 **Update examples** to use new coordinator architecture

## 18. Final Validation & Sign-Off

### 18.1 Run Full Test Suite

- [ ] 18.1.1 **Run tests with coverage:**

  ```bash
  pytest tests/ -v --cov=src/Medical_KG_rev --cov-report=html --cov-report=term
  ```

- [ ] 18.1.2 **Verify coverage hasn't decreased** compared to baseline

- [ ] 18.1.3 **Verify all tests pass** with no failures or errors

### 18.2 Run All Quality Checks

- [ ] 18.2.1 **Run pre-commit on all files:**

  ```bash
  pre-commit run --all-files
  ```

- [ ] 18.2.2 **Run ruff:**

  ```bash
  ruff check src/
  ```

- [ ] 18.2.3 **Run mypy:**

  ```bash
  mypy src/
  ```

- [ ] 18.2.4 **Verify 0 errors** from all tools

### 18.3 Build Documentation

- [ ] 18.3.1 **Build docs in strict mode:**

  ```bash
  mkdocs build --strict
  ```

- [ ] 18.3.2 **Verify build succeeds** with no warnings or errors

- [ ] 18.3.3 **Review generated documentation** in browser for formatting issues

### 18.4 Validate OpenSpec Change

- [ ] 18.4.1 **Run OpenSpec validation:**

  ```bash
  openspec validate add-pipeline-structure-documentation --strict
  ```

- [ ] 18.4.2 **Fix any validation errors** reported

### 18.5 Create Summary Report

- [ ] 18.5.1 **Create `openspec/changes/add-pipeline-structure-documentation/SUMMARY.md`** documenting:

  **Metrics:**
  - Total files refactored: [count]
  - Lines of code: Before [count] → After [count] (reduction of [count] lines, [percent]%)
  - Duplicate code blocks removed: [count]
  - Duplicate imports fixed: [count]
  - Docstrings added: [count]
  - Section headers added: [count]
  - Legacy helpers removed: [count]
  - Docstring coverage: Before [percent]% → After [percent]%
  - Test coverage: Before [percent]% → After [percent]%

  **Quality Improvements:**
  - All modules now have comprehensive docstrings
  - All classes and functions documented with Args/Returns/Raises
  - Code organized with consistent section headers
  - Duplicate code eliminated
  - Import organization standardized
  - Error translation centralized

  **Artifacts Created:**
  - Docstring templates: [link]
  - Section header standards: [link]
  - Developer extension guide: [link]
  - Architecture decision records: [4 ADRs created]
  - Before/after examples: [link]
  - Pipeline flow diagrams: [3 diagrams created]
  - Troubleshooting guide: [link]
  - Error translation matrix: [link]

  **Tooling Added:**
  - Ruff docstring enforcement
  - Section header checker
  - Docstring coverage checker
  - Pre-commit hooks for documentation quality
  - CI workflow for documentation validation
  - MkDocs API documentation generation

### 18.6 Request Peer Review

- [ ] 18.6.1 **Create PR with all changes**

- [ ] 18.6.2 **In PR description, include:**
  - Link to this change proposal
  - Link to summary report
  - Link to before/after examples
  - Link to new developer guides
  - List of key improvements with metrics

- [ ] 18.6.3 **Request review from:**
  - Pipeline platform team (owners of chunking & embedding services)
  - Gateway maintainers (REST and SSE orchestration)
  - Developer experience team (documentation automation)

### 18.7 Address Review Feedback

- [ ] 18.7.1 **Incorporate reviewer feedback** on documentation, structure, naming

- [ ] 18.7.2 **Update documentation** based on feedback

- [ ] 18.7.3 **Re-run validation checks** after changes

### 18.8 Merge and Archive

- [ ] 18.8.1 **After approval, merge PR** to main branch

- [ ] 18.8.2 **Run OpenSpec archive:**

  ```bash
  openspec archive add-pipeline-structure-documentation
  ```

- [ ] 18.8.3 **Update roadmap** or project tracking to reflect completion of documentation milestone

- [ ] 18.8.4 **Announce completion** to team with links to new documentation and guides

---

## Notes for AI Agents

**Critical Success Factors:**

1. **Exactness:** When fixing duplicates, ensure the canonical implementation is correctly identified before deleting
2. **Completeness:** Every function/class/module must have a docstring, no exceptions
3. **Consistency:** Follow templates exactly, don't deviate from established patterns
4. **Validation:** Run checks after each major section to catch issues early
5. **Documentation:** Document all decisions in audit.md as you work

**When In Doubt:**

- Refer to templates in `templates/` directory
- Check section header standards in `section_headers.md`
- Look at before/after examples for guidance
- Consult developer guide for extension patterns

**Order of Operations:**

1. Complete audit phase first (tasks 0-1) to understand full scope
2. Create templates and standards (tasks 2-3) before refactoring
3. Refactor one module type at a time (coordinators → services → policies → orchestration)
4. Add tooling enforcement (task 14) after refactoring to prevent regression
5. Validate continuously (task 15) to catch issues early
6. Create documentation (task 16) last when all code changes are complete
