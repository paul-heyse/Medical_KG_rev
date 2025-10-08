# Orchestration Pipelines Guide

## Pipeline Architecture and Flow

- **Ingestion**: Documents enter via `/v1/ingest` or `/v1/pipelines/ingest`, are queued in Kafka (`ingest.requests.v1`), processed by chunking → embedding → indexing workers, and complete on `ingest.results.v1`. Server-sent events broadcast `jobs.started`, `jobs.progress`, and `jobs.completed` notifications to `/v1/jobs/{job_id}/events`.
- **Query**: Requests reach `/v1/retrieve`, `/v1/pipelines/query`, or the GraphQL `retrieve` mutation. The gateway now delegates directly to `Medical_KG_rev.services.retrieval.RetrievalService`, which fans out to BM25/SPLADE/dense components, fuses their responses, applies reranking, and returns final selections annotated with `RETRIEVAL_PIPELINE_VERSION` metadata and optional explain traces.
- **State & Observability**: All stages update the in-memory ledger, emit Prometheus metrics, trace spans, and structured logs keyed by correlation IDs. SSE consumers receive updates with stage names, statuses, and timestamps for live monitoring.

## Configuration Guide

- Pipeline definitions live in `config/orchestration/pipelines.yaml`. Each stage declares `name`, `kind`, `timeout_ms`, and stage-specific `options` (e.g., enabled retrieval strategies or rerank candidate counts).
- `PipelineConfigManager` hot-reloads the YAML and snapshots versions under `config/orchestration/versions/` for auditability.
- Per-stage overrides are supplied through the `config` map injected into the pipeline context. Overrides can tweak timeouts, strategy fan-out, or fusion algorithms without modifying code.
- Validation ensures referenced services exist; breaking changes should create new pipeline entries to preserve backwards compatibility while existing jobs drain on the original version.

## Profile Creation and Customisation

- Profiles aggregate ingestion and query pipeline selections. Define new profiles in `pipelines.yaml` under `profiles`, optionally inheriting from a base profile via `extends`.
- Use the `overrides` map to specialise stage behaviour (e.g., enable BM25 and dense retrieval for PMC while favouring SPLADE for DailyMed).
- The `ProfileDetector` supports explicit profile requests (`profile` field on API calls) and metadata-driven detection (e.g., `metadata.source = "clinicaltrials"`). Unknown profiles trigger RFC 7807 errors before execution.

## Troubleshooting

- **Unknown profile**: Ensure the requested profile exists in `pipelines.yaml`; the API returns `400` with `type=https://httpstatuses.com/400`.
- **Stage timeout**: Check stage timings in response metadata; timeouts surface as `partial=true` with detailed `errors` including the offending stage.
- **No SSE events**: Verify the job ID and tenant when subscribing to `/v1/jobs/{job_id}/events`; historical events are retained per job even if no subscribers were connected when emitted.
- **Partial ingestion**: Batch responses include per-item HTTP status and error payloads; downstream workers continue processing successfully queued jobs.

## Evaluation Harness Usage

- The evaluation tooling under `src/Medical_KG_rev/eval/` loads ground-truth datasets, computes retrieval metrics (nDCG, Recall, MRR, MAP), and compares pipeline variants.
- Use `EvalHarness` to execute nightly evaluations across profiles and strategies, capturing per-stage metrics and regression alerts.
- Ground-truth datasets are stored as versioned JSONL files; the harness supports annotation templates and experiment definitions for A/B testing.

## Operational Runbook

- Deploy workers alongside the API gateway; ensure Kafka, Redis/Postgres ledger backends, and vector stores are reachable.
- Monitor Prometheus metrics for ingestion throughput, query latency percentiles, circuit breaker states, and DLQ accumulation. Alerts should route to PagerDuty and Slack using the observability helpers.
- Use SSE streams and `/v1/jobs/{job_id}` for real-time debugging, and correlate with tracing spans (Jaeger) using the propagated correlation IDs.
- Apply configuration updates via the YAML file; safe changes hot-reload automatically, while major revisions should be versioned and rolled out with blue/green deploys.
