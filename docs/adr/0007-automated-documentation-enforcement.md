# ADR-0007: Automated Documentation Enforcement

## Status

**Accepted** - 2024-01-15

## Context

As part of the repository-wide documentation standards (ADR-0005), we need to establish automated enforcement mechanisms to ensure compliance with documentation standards across the entire Medical_KG_rev repository. Manual enforcement is unreliable and doesn't scale to 360+ files.

The current state shows inconsistent application of documentation standards, with some modules having comprehensive documentation while others have minimal or no documentation. Without automated enforcement, the repository will regress to inconsistent documentation standards over time.

## Decision

We will implement comprehensive automated enforcement of documentation standards using:

1. **Pre-commit Hooks**: Run validation tools on modified files before commit
2. **CI/CD Pipelines**: Run validation tools on all files for pull requests and main branch
3. **Coverage Requirements**: Require 100% docstring coverage for all modules
4. **Validation Tools**: Custom tools for section headers, duplicate code, and type hints
5. **Integration with Existing Tools**: Leverage ruff, mypy, and other existing tools
6. **Blocking Criteria**: Failed validation blocks commits and merges

## Implementation Details

### Pre-commit Hooks Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      # Existing hooks
      - id: ruff
        name: Ruff linting
        entry: ruff check
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/

      - id: ruff-docstring-check
        name: Check docstrings with ruff
        entry: ruff check --select D
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/

      # New documentation enforcement hooks
      - id: section-header-check
        name: Check section headers
        entry: python scripts/check_section_headers.py
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/

      - id: docstring-coverage
        name: Check docstring coverage
        entry: python scripts/check_docstring_coverage.py --min-coverage 100
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

      - id: method-ordering-check
        name: Check method ordering
        entry: python scripts/check_method_ordering.py
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/

      - id: organize-imports
        name: Organize imports
        entry: python scripts/organize_imports.py
        language: system
        types: [python]
        files: ^src/Medical_KG_rev/
```

### CI/CD Pipeline Configuration

```yaml
# .github/workflows/documentation.yml
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
          pip install ruff mypy pydocstyle
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Check docstrings
        run: |
          ruff check --select D src/Medical_KG_rev/ tests/

      - name: Check section headers
        run: |
          python scripts/check_section_headers.py

      - name: Check docstring coverage
        run: |
          python scripts/check_docstring_coverage.py --min-coverage 100

      - name: Check for duplicate code
        run: |
          python scripts/find_duplicate_code.py

      - name: Check type hints
        run: |
          python scripts/check_type_hints.py

      - name: Check method ordering
        run: |
          python scripts/check_method_ordering.py

      - name: Run mypy
        run: |
          mypy --strict src/Medical_KG_rev/

      - name: Run pydocstyle
        run: |
          pydocstyle src/Medical_KG_rev/ tests/

      - name: Build documentation
        run: |
          mkdocs build --strict
```

### Validation Tools

#### Docstring Coverage Checker

```python
#!/usr/bin/env python3
"""Check docstring coverage across repository.

Reports:
- Overall coverage percentage
- Per-file coverage breakdown
- Missing docstrings with file:line:name format
- Coverage by module type (gateway, service, adapter, etc.)

Exit Codes:
- 0: 100% coverage achieved
- 1: Coverage below 100%
- 2: Validation errors encountered
"""
```

#### Section Header Checker

```python
#!/usr/bin/env python3
"""Validate section headers across repository.

Validates:
- Section presence for module type
- Section ordering per module type
- Section content appropriateness
- Comment formatting consistency

Exit Codes:
- 0: All sections valid
- 1: Section violations found
- 2: Validation errors encountered
"""
```

#### Duplicate Code Detector

```python
#!/usr/bin/env python3
"""Detect duplicate code across repository.

Detects:
- Duplicate functions (AST comparison)
- Duplicate classes (structure comparison)
- Duplicate imports (import analysis)
- Similar code blocks (token-based similarity)

Reports:
- Exact file paths and line numbers
- Similarity percentage
- Recommended canonical version
- Refactoring suggestions
"""
```

#### Type Hint Checker

```python
#!/usr/bin/env python3
"""Validate modern type hint usage.

Checks:
- Modern union syntax (Type | None vs Optional[Type])
- Collection generics (Mapping vs dict)
- Complete parameter annotations
- Complete return type annotations
- Generic type parameters

Fixes:
- Auto-convert Optional to union syntax
- Auto-convert dict/list to Mapping/Sequence
- Generate missing type annotations (suggestions)
"""
```

#### Method Ordering Checker

```python
#!/usr/bin/env python3
"""Check method ordering within classes.

Validates:
- Public methods before private methods
- Special methods (__init__, __repr__, etc.) first
- Alphabetical ordering within groups
- Consistent method organization

Reports:
- Method ordering violations
- Suggested reordering
- Auto-fix capability
"""
```

#### Import Organizer

```python
#!/usr/bin/env python3
"""Organize imports according to standards.

Organizes:
- Group imports by category (stdlib, third-party, first-party, relative)
- Sort imports alphabetically within groups
- Remove duplicate imports
- Format import statements

Features:
- Auto-fix capability
- Configurable grouping rules
- Integration with existing tools
"""
```

### Coverage Requirements

#### Docstring Coverage

- **Minimum Coverage**: 100% for all modules
- **Scope**: All Python files in `src/Medical_KG_rev/` and `tests/`
- **Validation**: Automated checking on every commit
- **Reporting**: Detailed coverage reports with missing items

#### Section Header Coverage

- **Minimum Coverage**: 100% for all modules
- **Scope**: All Python files in `src/Medical_KG_rev/` and `tests/`
- **Validation**: Automated checking on every commit
- **Reporting**: Detailed reports with missing sections and ordering issues

#### Type Hint Coverage

- **Minimum Coverage**: 100% for all public functions and classes
- **Scope**: All Python files in `src/Medical_KG_rev/`
- **Validation**: Automated checking on every commit
- **Reporting**: Detailed reports with missing annotations

### Blocking Criteria

#### Pre-commit Hooks

- **Failed Validation**: Blocks commit with detailed error messages
- **Fix Suggestions**: Provides specific fix suggestions for each violation
- **Auto-fix Options**: Some violations can be auto-fixed (imports, formatting)
- **Manual Review**: Complex violations require manual intervention

#### CI/CD Pipeline

- **Failed Validation**: Blocks merge with detailed error reports
- **Status Checks**: Required status checks for all pull requests
- **Artifact Upload**: Upload validation reports as build artifacts
- **Notification**: Notify developers of validation failures

## Consequences

### Positive

- **Consistent Quality**: All code meets the same documentation standards
- **Prevented Regression**: Automated enforcement prevents quality degradation
- **Developer Guidance**: Clear feedback helps developers improve their code
- **Reduced Review Overhead**: Automated checks catch issues before code review
- **Scalable Enforcement**: Works effectively with large codebases

### Negative

- **Initial Setup Overhead**: Time required to configure tools and pipelines
- **Learning Curve**: Developers need to learn the validation tools
- **False Positives**: Some valid code may be flagged incorrectly
- **Tooling Complexity**: Additional tools and configurations to maintain
- **Performance Impact**: Validation tools may slow down development workflow

### Risks and Mitigations

- **Risk**: Validation tools may be too strict and block valid code
  - **Mitigation**: Regular review and adjustment of validation rules
- **Risk**: False positives may frustrate developers
  - **Mitigation**: Provide clear error messages and fix suggestions
- **Risk**: Validation tools may slow down development workflow
  - **Mitigation**: Optimize tools for performance, use caching where possible
- **Risk**: Tools may not catch all violations
  - **Mitigation**: Regular review and improvement of validation logic

## Alternatives Considered

### Alternative 1: Manual Code Review Only

- **Description**: Rely solely on human code review for documentation quality
- **Rejected**: Not scalable, inconsistent, and prone to human error
- **Reason**: Manual review cannot ensure consistent application across 360+ files

### Alternative 2: Periodic Audits

- **Description**: Run validation tools periodically rather than on every commit
- **Rejected**: Allows violations to accumulate, harder to fix
- **Reason**: Periodic audits don't prevent regression and create technical debt

### Alternative 3: Optional Validation

- **Description**: Make validation tools optional with no blocking criteria
- **Rejected**: Optional tools are not used consistently
- **Reason**: Optional enforcement leads to inconsistent application

## Success Metrics

- **Validation Compliance**: 100% of commits pass all validation checks
- **Coverage Maintenance**: Docstring coverage remains at 100%
- **Developer Satisfaction**: Positive feedback on validation tools
- **Review Efficiency**: Reduced time spent on documentation review
- **Quality Consistency**: Consistent documentation quality across all modules
- **Regression Prevention**: Zero instances of documentation quality regression

## Implementation Timeline

- **Week 1**: Configure pre-commit hooks and validation tools
- **Week 2**: Set up CI/CD pipeline with validation checks
- **Week 3**: Test validation tools on existing codebase
- **Week 4**: Fix validation issues in existing code
- **Week 5**: Enable blocking criteria for new commits
- **Week 6**: Monitor and adjust validation rules
- **Week 7**: Optimize tool performance
- **Week 8**: Train developers on validation tools
- **Week 9**: Collect feedback and make improvements
- **Week 10**: Finalize validation configuration
- **Week 11**: Document validation process and tools

## References

- [Whole Repository Structure Documentation Proposal](../openspec/changes/whole-repo-structure-documentation/proposal.md)
- [Whole Repository Structure Documentation Design](../openspec/changes/whole-repo-structure-documentation/design.md)
- [Pre-commit Configuration](../.pre-commit-config.yaml)
- [CI/CD Pipeline Configuration](../.github/workflows/documentation.yml)

## Related ADRs

- ADR-0005: Repository-Wide Documentation Standards
- ADR-0006: Domain-Specific Section Headers
- ADR-0008: Type Hint Modernization
