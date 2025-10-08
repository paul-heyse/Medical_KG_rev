# Authoring Custom Orchestration Stages

The stage registry introduced by OpenSpec changes allows new orchestration
stages to be registered without modifying the Dagster runtime. This guide covers
the two built-in extensible stages—`download` and `gate`—and shows how to add
new custom stages.

## Stage Registry Fundamentals

- `StageMetadata` captures how a stage integrates with orchestration state:
  state keys, dependency hints, and output handlers.
- `StageRegistration` pairs metadata with a builder callable that turns a
  `StageDefinition` (from YAML) into an executable object.
- Builders run when a pipeline topology is loaded; they may capture shared
  resources such as HTTP clients or GPU pools.

Custom registrations can be added via `medical_kg.orchestration.stages` entry
points or by calling `StageRegistry.register_stage` during bootstrap.

## Download Stage

The download stage implements file acquisition with retry logic, ledger updates,
and Prometheus metrics. Key configuration keys:

| Key | Type | Description |
| --- | --- | --- |
| `storage.base_path` | string | Directory where files are written (created if missing). |
| `storage.filename_template` | string | Optional template using `{doc_id}`, `{job_id}`, `{tenant_id}`, `{timestamp}`. |
| `http.timeout_seconds` | number | Per-request timeout passed to the resilient HTTP client. |
| `http.max_attempts` | integer | Number of retry attempts. |
| `url_extractors` | list | Dot-paths describing where to find candidate URLs in upstream payloads or context metadata. |

The stage updates ledger metadata with `pdf_url`, `pdf_path`, `pdf_sha256`, and
emits metrics for download duration, size, and outcome. When a download fails,
the ledger records `pdf_download_error` and `pdf_last_attempt_url` to aid
troubleshooting.

## Gate Stage

Gates enforce external preconditions (e.g., waiting for MinerU IR). Configuration
fields:

- `gate`: symbolic gate name (must match an entry in the `gates:` section of the
  topology file).
- `field`: Job Ledger attribute or dot-path (`metadata.field`) to evaluate.
- `equals`: expected value.
- `resume_stage`: stage to continue executing after the gate passes.
- `timeout_seconds`, `poll_interval_seconds`: control wait strategy.

The stage polls the ledger, emits `pipeline_gate_wait_seconds` and
`pipeline_gate_events_total` metrics, and writes metadata such as
`gate.<name>.elapsed_seconds` when successful.

## Adding New Stages

1. Implement a class with an `execute(StageContext, payload)` method.
2. Register it with `StageRegistry.register_stage` or provide an entry point.
3. Define a topology YAML stage with a matching `type` and any custom config.
4. Provide a `StageMetadata` output handler to persist results into orchestration
   state for downstream stages.

Example snippet registering a `postprocess` stage:

```python
@dataclass
class PostProcessStage:
    def execute(self, ctx: StageContext, document: Document) -> Document:
        # mutate metadata, add annotations, etc.
        return document

metadata = StageMetadata(
    stage_type="postprocess",
    state_key="document",
    output_handler=lambda state, _, doc: state.__setitem__("document", doc),
    output_counter=lambda _: 1,
    description="Applies tenant-specific post-processing to IR documents",
    dependencies=("parse",),
)

registry.register_stage(metadata=metadata, builder=lambda _: PostProcessStage())
```

With the registration in place, the pipeline YAML can include:

```yaml
- name: postprocess
  type: postprocess
  policy: default
  depends_on:
    - parse
```

The stage factory automatically resolves and executes the new stage without
additional runtime changes.
