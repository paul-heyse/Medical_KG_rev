# Tasks — Connect Ledger State Management

- [x] Create a `LedgerStateManager` helper that encapsulates job attempt tracking, run preparation, stage lifecycle updates, and failure handling against the existing in-memory ledger.
- [x] Integrate the state manager into the Dagster runtime so run submission, stage execution, retries, and completion emit consistent ledger updates.
- [x] Add focused unit coverage for the new manager and refresh orchestration tests to assert the enriched metadata.
