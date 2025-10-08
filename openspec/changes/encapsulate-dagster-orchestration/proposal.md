## Why

The `_submit_dagster_job` method in `GatewayService` currently mixes pipeline resolution, adapter request construction, ledger idempotency, Dagster submission, and problem reporting in a 150+ line helper function. This creates tight coupling between the gateway layer and Dagster orchestration details, making it difficult to understand how job metadata propagates through the system and complicating testing of ingestion workflows.

The current approach requires the gateway to have intimate knowledge of Dagster pipeline topologies, domain resolution logic, and telemetry concerns, violating separation of concerns and making the codebase brittle when orchestration requirements change.

## What Changes

- **Extract `DagsterIngestionClient`**: Create a dedicated orchestration adapter that exposes a clean `submit(dataset, request, item) -> DagsterSubmissionResult` interface
- **Encapsulate pipeline logic**: Hide pipeline resolution, domain mapping, and telemetry concerns behind well-defined subroutines
- **Improve result typing**: Return typed results (success/duplicate/failure) that clearly indicate job outcomes and metadata
- **Simplify gateway orchestration**: Allow `IngestionCoordinator` to focus on result transformation and API response formatting rather than orchestration details
- **Enable testing isolation**: Make it possible to test ingestion workflows without Dagster dependencies through interface mocking

## Impact

- **Affected specs**: `specs/orchestration/spec.md` - Dagster orchestration interface and job metadata flow requirements
- **Affected code**:
  - `src/Medical_KG_rev/gateway/services.py` - Extract Dagster orchestration logic into dedicated client
  - `src/Medical_KG_rev/orchestration/dagster/` - Add new orchestration client module
  - `src/Medical_KG_rev/gateway/rest/router.py` - Update ingestion endpoints to use new interface
- **Affected systems**: Dagster orchestration, job metadata management, ingestion pipeline, error handling
