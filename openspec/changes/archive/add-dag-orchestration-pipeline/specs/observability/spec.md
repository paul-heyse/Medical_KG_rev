# Observability Spec Delta

## ADDED Requirements

### Requirement: CloudEvents Stage Lifecycle Emission

The orchestration system MUST emit CloudEvents 1.0 (<https://cloudevents.io/>) for all stage executions. Each Dagster op wrapper SHALL produce:

- **stage.started**: Emitted when op begins, with attributes:
  - `type`: "org.medicalkg.orchestration.stage.started"
  - `source`: f"orchestration/{stage_name}"
  - `subject`: Document ID
  - `datacontenttype`: "application/json"
  - `data`: `{"job_id", "tenant_id", "correlation_id", "stage", "timestamp"}`

- **stage.completed**: Emitted on success, with additional data:
  - `type`: "org.medicalkg.orchestration.stage.completed"
  - `data`: `{"duration_ms", "output_count", "gpu_used", "retry_count"}`

- **stage.failed**: Emitted on failure, with error details:
  - `type`: "org.medicalkg.orchestration.stage.failed"
  - `data`: `{"error_message", "error_type", "retry_count", "policy_name"}`

- **stage.retrying**: Emitted before retry, with backoff info:
  - `type`: "org.medicalkg.orchestration.stage.retrying"
  - `data`: `{"attempt_number", "backoff_ms", "error_message"}`

All events MUST be published to Kafka topic `orchestration.events.v1` in binary content mode.

#### Scenario: Chunk stage emits start and completion events

- **GIVEN** a document entering the chunk stage in the auto pipeline
- **WHEN** the `chunk_op` begins execution
- **THEN** a CloudEvent `stage.started` is published with:
  - `subject`: "clinicaltrials:NCT04267848"
  - `source`: "orchestration/chunk"
  - `data.stage`: "chunk"
  - `data.timestamp`: ISO 8601 timestamp
- **WHEN** chunking completes after 1200ms producing 42 chunks
- **THEN** a CloudEvent `stage.completed` is published with:
  - `data.duration_ms`: 1200
  - `data.output_count`: 42
  - `data.gpu_used`: false

#### Scenario: Embed stage retries and emits retry events

- **GIVEN** an embed stage with resilience policy "gpu-bound" (max_attempts=1, but circuit_breaker allows retry after reset)
- **WHEN** the first attempt fails with timeout
- **THEN** a CloudEvent `stage.retrying` is published with:
  - `data.attempt_number`: 1
  - `data.backoff_ms`: 2000
  - `data.error_message`: "Request timeout after 60s"
- **WHEN** the second attempt succeeds
- **THEN** a CloudEvent `stage.completed` is published with `data.retry_count`: 1

---

### Requirement: OpenLineage Job Run Tracking (Optional)

The orchestration system MAY emit OpenLineage (<https://openlineage.io/>) events for job runs and dataset lineage. If enabled via feature flag `MK_ENABLE_OPENLINEAGE=true`, the system SHALL:

- Create OpenLineage `RunEvent` for each Dagster job execution with:
  - `run.runId`: Dagster run ID
  - `run.facets.parent`: Parent job reference (for two-phase pipelines)
  - `job.namespace`: "medical-kg"
  - `job.name`: Pipeline name (e.g., "auto", "pdf-two-phase")

- Create dataset facets for each stage:
  - `inputDatasets`: Source documents consumed (e.g., raw adapter payloads)
  - `outputDatasets`: Produced artifacts (e.g., chunks, embeddings, graph nodes)
  - Custom facets for GPU usage, model versions, retry attempts

- Publish events to configurable backend (HTTP, Kafka, Marquez)

#### Scenario: OpenLineage tracks chunk stage dataset lineage

- **GIVEN** OpenLineage enabled and Marquez backend configured
- **WHEN** the chunk stage executes on document "clinicaltrials:NCT04267848"
- **THEN** an OpenLineage RunEvent is emitted with:
  - `inputs`: [{"namespace": "medical-kg", "name": "documents", "facets": {"dataSource": {"uri": "clinicaltrials:NCT04267848"}}}]
  - `outputs`: [{"namespace": "medical-kg", "name": "chunks", "facets": {"dataSource": {"uri": "chunks:NCT04267848:*"}}}]
  - `run.facets.processing`: {"input_count": 1, "output_count": 42, "duration_ms": 1200}
- **AND** the Marquez UI displays the lineage graph: document → chunk stage → chunks

#### Scenario: OpenLineage disabled, no overhead

- **GIVEN** feature flag `MK_ENABLE_OPENLINEAGE=false` (default)
- **WHEN** any stage executes
- **THEN** no OpenLineage events are generated
- **AND** no HTTP calls to Marquez are made
- **AND** stage execution latency is not impacted

---

### Requirement: Prometheus Dagster Metrics

The orchestration system MUST expose Prometheus metrics for Dagster job and stage execution. New metrics:

- `dagster_job_duration_seconds{pipeline, status}` - Histogram of job execution time
- `dagster_stage_duration_seconds{stage, status}` - Histogram of stage execution time
- `dagster_stage_retry_attempts_total{stage, reason}` - Counter of retry attempts per stage
- `dagster_circuit_breaker_state{stage, state}` - Gauge of circuit breaker state (0=closed, 1=open, 2=half-open)
- `dagster_rate_limit_wait_seconds{stage}` - Histogram of rate limit wait times
- `dagster_job_active{pipeline}` - Gauge of currently running jobs

Metrics SHALL be scraped from Dagster daemon at `/metrics` endpoint.

#### Scenario: Prometheus scrapes Dagster metrics

- **GIVEN** a Dagster daemon running with Prometheus exporter enabled
- **WHEN** Prometheus scrapes `http://dagster-daemon:9091/metrics`
- **THEN** metrics are returned in Prometheus text format
- **AND** histogram buckets for `dagster_stage_duration_seconds` cover 0.1s to 600s
- **AND** labels include `stage` (e.g., "chunk", "embed"), `status` (e.g., "success", "failed")

#### Scenario: Grafana dashboard visualizes stage durations

- **GIVEN** a Grafana dashboard configured with Prometheus datasource
- **WHEN** querying `histogram_quantile(0.95, rate(dagster_stage_duration_seconds_bucket[5m]))`
- **THEN** P95 latency for each stage is displayed (e.g., chunk: 1.2s, embed: 3.5s, index: 0.8s)
- **AND** alerts trigger if any stage exceeds SLO (e.g., embed > 5s)

---

### Requirement: AsyncAPI Queue Documentation

The system MUST document all Kafka topics used for orchestration in an AsyncAPI 3.0 (<https://www.asyncapi.com/>) specification. The spec SHALL be located at `docs/asyncapi.yaml` and include:

- **orchestration.events.v1**: CloudEvents stream for stage lifecycle
  - Schema: CloudEvent 1.0 with custom data fields per event type
  - Operations: publish (stages), subscribe (monitoring, lineage)

- **mineru.queue.v1**: PDF processing requests for GPU nodes
  - Schema: `{"job_id", "tenant_id", "pdf_url", "priority"}`
  - Operations: publish (download stage), subscribe (MinerU workers)

- **ingest.requests.v1**: Ingestion job submissions (existing topic, updated docs)
- **ingest.results.v1**: Ingestion completion notifications (existing topic, updated docs)

The AsyncAPI spec MUST be validated in CI/CD with `asyncapi validate docs/asyncapi.yaml`.

#### Scenario: AsyncAPI spec describes orchestration.events.v1

- **GIVEN** `docs/asyncapi.yaml` with orchestration.events.v1 channel definition
- **WHEN** the spec is parsed by AsyncAPI tools
- **THEN** the channel is documented with:
  - `address`: "orchestration.events.v1"
  - `messages`: stage.started, stage.completed, stage.failed, stage.retrying
  - `bindings.kafka`: {"partitions": 10, "replicas": 3, "retention": "7d"}
- **AND** JSON Schema definitions for each CloudEvent data payload are included

#### Scenario: AsyncAPI validation catches schema errors

- **GIVEN** a modified AsyncAPI spec with invalid message schema (missing required field)
- **WHEN** `asyncapi validate docs/asyncapi.yaml` runs in CI
- **THEN** the validation fails with error message indicating the missing field
- **AND** the pull request is blocked until the schema is fixed

---

## MODIFIED Requirements

### Requirement: Structured Logging with Correlation IDs

Structured logging MUST include orchestration-specific fields when Dagster is enabled. All log messages SHALL include:

- `correlation_id`: Dagster run ID (for tracing across stages)
- `job_id`: Job Ledger job ID
- `tenant_id`: Tenant identifier
- `stage`: Current stage name (e.g., "chunk", "embed")
- `pipeline`: Pipeline name (e.g., "auto", "pdf-two-phase")
- `dagster_run_id`: Dagster run ID (same as correlation_id but explicit)

Logs SHALL be emitted in JSON format and forwarded to existing Loki/CloudWatch infrastructure.

#### Scenario: Chunk stage logs include Dagster context

- **GIVEN** the chunk_op executing in a Dagster job
- **WHEN** a log message is emitted during chunking
- **THEN** the log entry includes:

  ```json
  {
    "level": "info",
    "timestamp": "2025-01-15T10:30:00Z",
    "message": "Chunking document",
    "correlation_id": "dagster-run-abc123",
    "job_id": "job-456",
    "tenant_id": "tenant-789",
    "stage": "chunk",
    "pipeline": "auto",
    "dagster_run_id": "dagster-run-abc123",
    "chunks_produced": 42
  }
  ```

#### Scenario: Log aggregation in Loki by Dagster run ID

- **GIVEN** logs forwarded to Loki from Dagster daemon and workers
- **WHEN** querying `{dagster_run_id="dagster-run-abc123"}`
- **THEN** all log entries for that run are returned across all stages
- **AND** logs are chronologically ordered by timestamp
- **AND** errors are highlighted with `level="error"`

---

## ADDED Requirements

### Requirement: Dagster UI Integration

The Dagster webserver MUST be accessible at `http://localhost:3000` (dev) or `https://dagster.medical-kg.example.com` (prod) for job monitoring. The UI SHALL provide:

- **Runs**: List of all job executions with status (success, failed, running)
- **Assets**: Materialized data products (chunks, embeddings, graph nodes) with lineage
- **Logs**: Real-time stage-level logs with filtering by job, stage, level
- **Sensors**: Status of `pdf_ir_ready_sensor` with last poll time and trigger count
- **Resources**: Configuration of resources (plugin_manager, job_ledger, kafka)

Access MUST be restricted to authenticated users via OAuth 2.0 (same auth as gateway).

#### Scenario: Engineer debugs failed job in Dagster UI

- **GIVEN** a job failed at the embed stage with GPU OOM error
- **WHEN** the engineer navigates to Dagster UI Runs page
- **THEN** the failed run is displayed with red status indicator
- **WHEN** clicking on the run ID
- **THEN** the job graph is displayed with chunk (success), embed (failed), index (not started)
- **WHEN** clicking on the embed op
- **THEN** logs show error message "CUDA out of memory" with timestamp and stack trace
- **AND** CloudEvents for `stage.failed` are visible in the timeline

#### Scenario: Engineer monitors sensor activity

- **GIVEN** the `pdf_ir_ready_sensor` running on Dagster daemon
- **WHEN** the engineer navigates to Sensors page in Dagster UI
- **THEN** the sensor status is "Running" with last poll time "2 seconds ago"
- **AND** trigger count shows "15 jobs triggered in last 24 hours"
- **AND** a list of recently triggered runs is displayed with job IDs and timestamps
