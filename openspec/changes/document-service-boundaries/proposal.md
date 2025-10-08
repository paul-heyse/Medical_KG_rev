## Why

The current gateway codebase lacks clear documentation of service boundaries and interactions between coordinators, orchestrators, registries, and routers. As the system has grown through the refactoring efforts, the relationships between these components have become more complex, making it difficult for new contributors to understand which abstractions own which responsibilities and how they interact. Without clear service boundary documentation, future development and maintenance will be hindered by confusion about component responsibilities and interaction patterns.

## What Changes

- **Create module-level READMEs**: Add lightweight documentation files in `src/Medical_KG_rev/gateway/README.md` and subdirectories explaining service boundaries and interactions
- **Add interaction diagrams**: Create visual diagrams showing how coordinators, orchestrators, registries, and routers interact
- **Document responsibility boundaries**: Clearly define which components own which responsibilities and how they should interact
- **Create component overviews**: Provide high-level descriptions of each major component and its role in the system
- **Add navigation aids**: Create cross-references and navigation guides to help developers understand the codebase structure

## Impact

- **Affected specs**: `specs/gateway/spec.md` - Service boundary documentation and navigation requirements
- **Affected code**:
  - `src/Medical_KG_rev/gateway/` - Add README.md files documenting service boundaries
  - `src/Medical_KG_rev/gateway/coordinators/` - Add coordinator-specific documentation
  - `src/Medical_KG_rev/gateway/orchestrators/` - Add orchestration-specific documentation
- **Affected systems**: Developer experience, code navigation, onboarding, maintenance efficiency
