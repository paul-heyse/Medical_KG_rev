## Why

The current orchestration runtime uses `_apply_stage_output` and `_infer_output_count` to manipulate a shared `dict[str, Any]` for passing stage results forward. This approach makes it difficult to trace which stages populate which keys, forces consumers to perform defensive casting, and creates boilerplate conditionals throughout the runtime logic. The lack of type safety also makes it easy to introduce bugs when accessing stage outputs.

**Critical Barrier**: The PDF pipeline requires specific state tracking for download and gate stages, but the current untyped dictionary approach makes it impossible to reliably track PDF-specific state transitions and ensure proper pipeline progression.

## What Changes

- **Introduce `PipelineState` dataclass**: Create a strongly-typed dataclass with explicit fields for payloads, documents, embeddings, entities, claims, and other stage outputs
- **Replace dict manipulation**: Update `_apply_stage_output` and `_infer_output_count` to work with the typed state object
- **Add helper methods**: Provide convenience methods for optional stages, type-safe accessors, and validation
- **Update stage contracts**: Modify stage interfaces to accept and return typed state objects
- **Maintain backward compatibility**: Ensure existing pipeline configurations continue to work during migration

## Impact

- **Affected specs**: `specs/orchestration/spec.md` - Typed state management requirements
- **Affected code**:
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py` - Replace dict-based state with PipelineState
  - `src/Medical_KG_rev/orchestration/stages/contracts.py` - Update stage interfaces for typed state
  - `src/Medical_KG_rev/orchestration/dagster/stages.py` - Update stage implementations to use typed state
- **Affected systems**: Dagster orchestration runtime, stage execution, pipeline state management
