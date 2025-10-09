## Change Proposal: whole-repo-structure-documentation

### Why

- The Medical_KG_rev repository contains 360+ Python files across multiple domains (gateway, services, orchestration, adapters, validation, kg, storage, etc.) with inconsistent documentation standards and code organization.
- Following the successful `add-pipeline-structure-documentation` change that achieved 100% docstring coverage and consistent structure for pipeline modules, we need to extend these rigorous standards to the entire codebase.
- Current state shows only 57.8% overall docstring coverage (532/920 items documented) with 101 files still needing comprehensive documentation, indicating significant technical debt.
- Inconsistent code organization across modules makes onboarding difficult, code reviews inefficient, and automated documentation generation incomplete.
- The lack of standardized section headers, import organization, and method ordering creates maintenance overhead and reduces code quality.
- Without comprehensive documentation standards, the repository cannot effectively serve as a reference implementation for biomedical knowledge integration systems.

### Goals

1. **Achieve 100% docstring coverage** across all 360+ Python files in the repository, extending the successful pipeline documentation standards to every module.
2. **Standardize code organization** with consistent section headers, import ordering, and method organization across all modules.
3. **Eliminate duplicate code** and legacy patterns throughout the repository, not just in pipeline modules.
4. **Establish automated enforcement** of documentation standards with pre-commit hooks, CI validation, and coverage tracking.
5. **Create comprehensive developer resources** including templates, guides, and examples for maintaining documentation standards.
6. **Generate complete API documentation** from docstrings using mkdocstrings across all modules.
7. **Modernize type hints** throughout the repository to use modern Python conventions (union syntax, generics from collections.abc).
8. **Document architectural decisions** and extension patterns for all major subsystems.

### Non-Goals

- Changing runtime behavior or API contracts of existing modules
- Rewriting core business logic or algorithms
- Modifying external dependencies or build system
- Changing test coverage requirements or testing frameworks
- Altering deployment or infrastructure configurations

### Stakeholders / Reviewers

- **Core Platform Team** (owners of gateway, services, orchestration)
- **Data Integration Team** (owners of adapters, validation, kg modules)
- **Infrastructure Team** (owners of storage, monitoring, deployment)
- **Developer Experience Team** (documentation automation, tooling)
- **Quality Assurance Team** (testing standards, validation)

### Proposed Implementation Plan

#### Phase 1 – Comprehensive Repository Audit (Week 1-2)

- **File Inventory**: Catalog all 360+ Python files with exact paths, line counts, responsibilities, and dependencies
- **Documentation Gap Analysis**: Run docstring coverage analysis across entire repository, identifying all missing docstrings
- **Duplicate Code Detection**: Use AST analysis and pattern matching to identify duplicate implementations across modules
- **Type Hint Assessment**: Evaluate type annotation coverage and identify modernization opportunities
- **Structural Analysis**: Assess section header usage, import organization, and method ordering across all modules
- **Legacy Code Identification**: Find deprecated patterns, unused helpers, and superseded implementations

#### Phase 2 – Standards Extension & Tooling Enhancement (Week 2-3)

- **Extend Documentation Templates**: Adapt pipeline templates for all module types (adapters, validation, kg, storage, etc.)
- **Enhance Section Header Standards**: Define canonical structures for all module types beyond pipeline modules
- **Upgrade Validation Tools**: Extend section header checker, docstring coverage checker, and create new validation tools
- **Configure Enforcement**: Update pre-commit hooks, CI workflows, and linting configuration for repository-wide coverage
- **Create Migration Tools**: Develop automated tools for applying standards to large numbers of files

#### Phase 3 – Domain-by-Domain Refactoring (Week 3-8)

- **Gateway Modules**: Apply standards to all gateway components (coordinators, services, presentation, errors)
- **Service Modules**: Document all service implementations (embedding, chunking, retrieval, reranking, evaluation, extraction, gpu, mineru)
- **Adapter Modules**: Standardize all adapter implementations (biomedical, terminology, plugins, mixins)
- **Orchestration Modules**: Document orchestration system (dagster, stages, contracts, plugins, state management)
- **Knowledge Graph Modules**: Apply standards to kg components (schema, neo4j_client, cypher_templates, shacl)
- **Storage Modules**: Document storage abstractions (clients, vector store implementations)
- **Validation Modules**: Standardize validation components (fhir, ucum)
- **Utility Modules**: Document utility functions and helpers

#### Phase 4 – Advanced Documentation & Integration (Week 8-10)

- **API Documentation Generation**: Configure mkdocstrings for complete repository coverage
- **Architecture Decision Records**: Document key architectural decisions across all subsystems
- **Developer Extension Guides**: Create comprehensive guides for extending each major subsystem
- **Visual Documentation**: Create diagrams showing relationships between all major components
- **Troubleshooting Guides**: Document common issues and solutions across all modules
- **Performance Documentation**: Document performance characteristics and optimization opportunities

#### Phase 5 – Validation & Quality Assurance (Week 10-11)

- **Comprehensive Testing**: Ensure all refactored modules maintain existing functionality
- **Documentation Validation**: Verify all generated documentation is accurate and complete
- **Performance Validation**: Ensure documentation additions don't impact runtime performance
- **Integration Testing**: Validate that all modules work together correctly after refactoring
- **Final Quality Checks**: Run all validation tools and achieve 100% compliance

### Risks & Mitigations

- **Risk**: Large-scale refactoring could introduce bugs. *Mitigation*: Comprehensive testing, incremental rollout, and automated validation at each step.
- **Risk**: Documentation overhead could slow development. *Mitigation*: Provide templates, automation tools, and clear guidelines to minimize overhead.
- **Risk**: Inconsistent application of standards. *Mitigation*: Automated enforcement, peer review requirements, and comprehensive examples.
- **Risk**: Breaking changes in external interfaces. *Mitigation*: Focus on internal documentation only, maintain all public APIs unchanged.

### Open Questions

- Should we prioritize certain domains (e.g., core services) over others for initial implementation?
- Do we want to establish different documentation standards for different module types (e.g., utilities vs. core services)?
- Should we create domain-specific section header standards beyond the current pipeline standards?

### Success Metrics

- **Docstring Coverage**: Achieve 100% coverage (currently 57.8%)
- **Files Documented**: All 360+ Python files have comprehensive documentation
- **Validation Compliance**: 0 errors from all documentation validation tools
- **API Documentation**: Complete API documentation generated for all modules
- **Developer Experience**: Reduced onboarding time, improved code review efficiency
- **Maintenance Overhead**: Reduced time spent understanding undocumented code

### Estimated Timeline

- **Total Duration**: 11 weeks
- **Phase 1**: 2 weeks (audit and analysis)
- **Phase 2**: 1 week (standards and tooling)
- **Phase 3**: 5 weeks (domain-by-domain refactoring)
- **Phase 4**: 2 weeks (advanced documentation)
- **Phase 5**: 1 week (validation and QA)

### Dependencies

- Completion of `add-pipeline-structure-documentation` change (provides templates and standards)
- Access to all repository modules and their current state
- Development environment with all validation tools configured
- Team availability for review and validation of refactored modules
