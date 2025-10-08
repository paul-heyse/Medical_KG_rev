# Typed Pipeline State Management

The orchestration runtime now persists pipeline state using a typed contract built on
Pydantic models, attrs-based caching, and structured logging. The state serialisation
flow validates payloads with `PipelineStateModel` before encoding them using `orjson`
and compressing snapshots for transport.

## Runtime Behaviour

* Stage execution enforces declared `depends_on` relationships prior to invoking the
  stage implementation. If a dependency is missing or failed the runtime raises a
  descriptive error to prevent out-of-order execution.
* PDF pipelines use the new `pdf-download` and `pdf-ir-gate` stage types to track gate
  progress. The pipeline state captures gate metadata and the job ledger is updated via
  `JobLedger.set_pdf_downloaded` and `JobLedger.set_pdf_ir_ready` when the stages
  complete.
* Snapshots are cached for 120 seconds using `PipelineStateCache`. Subsequent
  serialisation requests reuse cached payloads, reducing repeated `orjson` work during
  intensive telemetry reporting.
* Stage metrics are emitted to Prometheus counters and histograms through
  `record_stage_metrics`, providing duration, attempt counts, and failure statistics.
* The Dagster runtime persists snapshots to the job ledger through
  `PipelineStatePersister`, which applies tenacity-powered retries before surfacing a
  `StatePersistenceError`.

## Developer Guidance

1. Use `PipelineState.to_model()` to retrieve a validated representation that can be
   safely logged or passed to external systems.
2. When integrating new stages include a `depends_on` list in topology definitions so
   the runtime can enforce the execution order.
3. For PDF workflows emit dictionaries from gate stages to enrich the gate metadata.
4. Use `PipelineState.reset_pdf_gate()` to restart the gate when reprocessing a PDF
   payload.
5. Prefer `PipelineState.serialise_json()` for logging and
   `PipelineState.serialise_base64()` for durable storage.

```python
from Medical_KG_rev.orchestration.stages.contracts import PipelineState


def handle_state(state: PipelineState) -> None:
    model = state.to_model()
    print(model.context.tenant_id)
    if model.pdf_gate and model.pdf_gate.downloaded:
        ...
```
