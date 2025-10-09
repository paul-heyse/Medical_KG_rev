# ADR-0006: Domain-Specific Section Headers

## Status

**Accepted** - 2024-01-15

## Context

As part of the repository-wide documentation standards (ADR-0005), we need to define consistent section header standards for all module types across the Medical_KG_rev repository. Different domains have different organizational needs, but consistency across domains is important for maintainability.

The current state shows inconsistent section header usage across modules, with some modules having no section headers at all, and others using ad-hoc organizational patterns. This inconsistency makes the codebase harder to navigate and maintain.

## Decision

We will implement domain-specific section header standards that maintain consistency across the repository while accommodating the unique needs of each domain. Each domain will have:

1. **Consistent ordering rules** across all domains
2. **Domain-specific sections** that reflect the domain's responsibilities
3. **Standardized section header format** using `# ============================================================================== SECTION NAME ==============================================================================`
4. **Validation tools** to enforce section presence and ordering
5. **Migration tools** to apply standards to existing modules

## Domain-Specific Section Standards

### Gateway Modules

```python
# ==============================================================================
# IMPORTS
# ==============================================================================

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

# ==============================================================================
# COORDINATOR IMPLEMENTATION
# ==============================================================================

# ==============================================================================
# ERROR TRANSLATION
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

# ==============================================================================
# EXPORTS
# ==============================================================================
```

### Service Modules

```python
# ==============================================================================
# IMPORTS
# ==============================================================================

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# DATA MODELS
# ==============================================================================

# ==============================================================================
# INTERFACES
# ==============================================================================

# ==============================================================================
# IMPLEMENTATIONS
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# ==============================================================================
# EXPORTS
# ==============================================================================
```

### Adapter Modules

```python
# ==============================================================================
# IMPORTS
# ==============================================================================

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# DATA MODELS
# ==============================================================================

# ==============================================================================
# ADAPTER IMPLEMENTATION
# ==============================================================================

# ==============================================================================
# ERROR HANDLING
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# ==============================================================================
# EXPORTS
# ==============================================================================
```

### Orchestration Modules

```python
# ==============================================================================
# IMPORTS
# ==============================================================================

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# STAGE CONTEXT DATA MODELS
# ==============================================================================

# ==============================================================================
# STAGE IMPLEMENTATIONS
# ==============================================================================

# ==============================================================================
# PLUGIN REGISTRATION
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# ==============================================================================
# EXPORTS
# ==============================================================================
```

### Knowledge Graph Modules

```python
# ==============================================================================
# IMPORTS
# ==============================================================================

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# SCHEMA DATA MODELS
# ==============================================================================

# ==============================================================================
# CLIENT IMPLEMENTATION
# ==============================================================================

# ==============================================================================
# TEMPLATES
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# ==============================================================================
# EXPORTS
# ==============================================================================
```

### Storage Modules

```python
# ==============================================================================
# IMPORTS
# ==============================================================================

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# DATA MODELS
# ==============================================================================

# ==============================================================================
# INTERFACES
# ==============================================================================

# ==============================================================================
# IMPLEMENTATIONS
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# ==============================================================================
# EXPORTS
# ==============================================================================
```

### Validation Modules

```python
# ==============================================================================
# IMPORTS
# ==============================================================================

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# DATA MODELS
# ==============================================================================

# ==============================================================================
# VALIDATOR IMPLEMENTATION
# ==============================================================================

# ==============================================================================
# ERROR HANDLING
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# ==============================================================================
# EXPORTS
# ==============================================================================
```

### Utility Modules

```python
# ==============================================================================
# IMPORTS
# ==============================================================================

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# DATA MODELS
# ==============================================================================

# ==============================================================================
# EXCEPTIONS
# ==============================================================================

# ==============================================================================
# HELPER CLASSES
# ==============================================================================

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# ==============================================================================
# EXPORTS
# ==============================================================================
```

### Test Modules

```python
# ==============================================================================
# IMPORTS
# ==============================================================================

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# FIXTURES
# ==============================================================================

# ==============================================================================
# UNIT TESTS - [ComponentName]
# ==============================================================================

# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# ==============================================================================
# EXPORTS
# ==============================================================================
```

## Implementation Details

### Section Header Format

- **Format**: `# ============================================================================== SECTION NAME ==============================================================================`
- **Length**: Exactly 77 characters per line
- **Spacing**: Single blank line before and after section headers
- **Case**: UPPERCASE with underscores for spaces

### Validation Rules

1. **Section Presence**: All required sections must be present
2. **Section Ordering**: Sections must appear in the defined order
3. **Section Content**: Code must be placed in appropriate sections
4. **Import Organization**: Imports must be grouped and sorted within IMPORTS section
5. **Method Ordering**: Methods must be ordered within their sections

### Migration Strategy

1. **Automated Detection**: Use AST analysis to identify module types
2. **Section Insertion**: Automatically insert missing section headers
3. **Code Reorganization**: Move code to appropriate sections
4. **Validation**: Verify section presence and ordering
5. **Manual Review**: Human review of automated changes

## Consequences

### Positive

- **Consistent Organization**: All modules follow the same organizational pattern
- **Improved Navigability**: Developers can quickly find relevant code sections
- **Better Maintainability**: Consistent structure makes code easier to maintain
- **Automated Validation**: Tools can verify compliance with standards
- **Domain-Specific Optimization**: Each domain has sections tailored to its needs

### Negative

- **Initial Migration Overhead**: Time required to reorganize existing modules
- **Learning Curve**: Developers need to learn the new section standards
- **Tooling Complexity**: Additional validation tools and CI/CD pipeline complexity
- **Maintenance Overhead**: Ongoing effort to maintain section standards

### Risks and Mitigations

- **Risk**: Developers may resist the new organizational structure
  - **Mitigation**: Provide clear examples and migration tools
- **Risk**: Automated migration tools may make incorrect decisions
  - **Mitigation**: Human review of automated changes, manual fallback options
- **Risk**: Section standards may be too restrictive for some use cases
  - **Mitigation**: Allow domain-specific variations within the framework

## Alternatives Considered

### Alternative 1: Single Section Standard for All Modules

- **Description**: Use the same section headers for all module types
- **Rejected**: Too restrictive for different domain needs
- **Reason**: Different domains have different organizational requirements

### Alternative 2: No Section Headers

- **Description**: Remove all section headers and use natural code organization
- **Rejected**: Would lead to inconsistent organization
- **Reason**: Current state shows this approach leads to poor organization

### Alternative 3: Optional Section Headers

- **Description**: Make section headers optional with no enforcement
- **Rejected**: Would not achieve consistency
- **Reason**: Optional standards are not enforced and lead to inconsistency

## Success Metrics

- **Section Compliance**: 100% of modules have correct section headers
- **Ordering Compliance**: 100% of modules have sections in correct order
- **Content Compliance**: 100% of code is placed in appropriate sections
- **Developer Satisfaction**: Positive feedback on improved code organization
- **Navigation Efficiency**: Reduced time to find relevant code sections
- **Maintenance Efficiency**: Reduced time to understand and modify code

## Implementation Timeline

- **Week 1**: Define section standards for all domains
- **Week 2**: Create validation tools for section compliance
- **Week 3**: Implement migration tools for existing modules
- **Week 4**: Apply standards to Gateway modules
- **Week 5**: Apply standards to Service modules
- **Week 6**: Apply standards to Adapter modules
- **Week 7**: Apply standards to Orchestration modules
- **Week 8**: Apply standards to remaining modules
- **Week 9**: Validate compliance and fix issues
- **Week 10**: Update documentation and examples
- **Week 11**: Final validation and sign-off

## References

- [Whole Repository Structure Documentation Proposal](../openspec/changes/whole-repo-structure-documentation/proposal.md)
- [Whole Repository Structure Documentation Design](../openspec/changes/whole-repo-structure-documentation/design.md)
- [Section Headers Standards](../openspec/changes/whole-repo-structure-documentation/section_headers.md)

## Related ADRs

- ADR-0005: Repository-Wide Documentation Standards
- ADR-0007: Automated Documentation Enforcement
- ADR-0008: Type Hint Modernization
