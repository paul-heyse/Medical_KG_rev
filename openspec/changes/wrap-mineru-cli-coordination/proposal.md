## Why

The `MineruProcessor` class currently handles settings resolution, CLI orchestration, error handling, simulated fallbacks, vLLM health checks, and metadata building in a single monolithic class. This creates a large constructor with many dependencies and makes it difficult to substitute components in tests or alternative deployments. The coordination logic is tightly coupled with execution details, making the class hard to reason about and extend.

## What Changes

- **Extract service interfaces**: Define separate interfaces for `MineruWorker` (CLI lifecycle), `OCRBackend` (actual execution), and `FallbackStrategy` (switching to simulation)
- **Shrink constructor dependencies**: Break down the monolithic class into smaller, focused components
- **Improve testability**: Make it easier to substitute components in tests and mock specific behaviors
- **Clarify execution paths**: Make GPU-vs-simulated runs explicit through strategy classes
- **Align with existing patterns**: Follow the pipeline/pipeline-metrics split pattern but push more coordination logic into composable services

## Impact

- **Affected specs**: `specs/gpu-microservices/spec.md` - MinerU service architecture and interface requirements
- **Affected code**:
  - `src/Medical_KG_rev/services/mineru/service.py` - Refactor into smaller, focused components
  - `src/Medical_KG_rev/services/mineru/` - Add new interfaces and strategy classes
  - `tests/services/mineru/` - Update tests to use new architecture
- **Affected systems**: MinerU GPU service, PDF processing pipeline, service composition
