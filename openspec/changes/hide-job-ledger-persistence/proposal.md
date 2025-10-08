## Why

The `JobLedger` is currently an in-memory map that also acts as the mutation API for job status, PDF gate signals, and retry counters. This creates tight coupling between the orchestration runtime and persistence concerns, making it difficult to implement durable storage, enforce invariants centrally, or test ledger behavior in isolation. The current design mixes stateful transitions with business logic orchestration.

## What Changes

- **Define repository interface**: Create a `LedgerRepository` interface with concrete in-memory and persistent implementations
- **Clarify responsibility boundaries**: Separate stateful transitions from business logic orchestration
- **Centralize invariant enforcement**: Move PDF-related invariants and validation logic into the repository layer
- **Enable durable storage**: Prepare for persistent storage implementations while maintaining in-memory fallback
- **Improve testability**: Make it easier to test ledger behavior with mock implementations and isolated state management

## Impact

- **Affected specs**: `specs/orchestration/spec.md` - Job ledger architecture and persistence requirements
- **Affected code**:
  - `src/Medical_KG_rev/orchestration/ledger.py` - Refactor into repository pattern with interface
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py` - Update to use repository interface
  - `src/Medical_KG_rev/orchestration/dagster/stages.py` - Update stage integration to use repository
- **Affected systems**: Job orchestration, state management, persistence layer
