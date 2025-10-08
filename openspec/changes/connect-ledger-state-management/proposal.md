# Connect Ledger State Management

## Summary
- Introduce a state manager that centralises how orchestration runs interact with the job ledger.
- Ensure Dagster jobs consistently update ledger metadata, retries, and stage transitions.
- Provide automated coverage validating ledger synchronisation behaviour.

## Motivation
Current Dagster runtime code manipulates the ledger directly in multiple places, making it difficult to reason about retry counters, stage metadata, and pipeline bookkeeping. A dedicated state manager ties together job attempts, stage lifecycle updates, and run metadata so that ledger state stays consistent with orchestration behaviour.

## Scope
- New orchestration helper that wraps the existing `JobLedger` implementation.
- Runtime integration so all stage lifecycle events flow through the helper.
- Tests for the helper and adjustments to existing Dagster tests to align with the new behaviour.

## Out of Scope
- Persisted ledger storage engines (remains in-memory).
- New pipeline topologies or adapters.

## Success Criteria
- All new helper methods exercised by unit tests.
- Dagster runtime uses the helper instead of duplicating ledger logic.
- Ledger metadata captures stage attempts, counts, durations, and completion timestamps for each stage.
