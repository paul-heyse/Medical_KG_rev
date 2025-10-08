# Authoring Custom Orchestration Stages

This guide walks through registering bespoke orchestration stages with the
pluggable registry introduced in the pipeline topology configuration.

## 1. Implement the Stage

1. Define a class that conforms to one of the contracts in
   `Medical_KG_rev.orchestration.stages.contracts`. The class must expose an
   `execute` method that accepts a `StageContext` plus the upstream payload.
2. Keep implementations framework agnostic—no Dagster dependencies should leak
   into the stage logic.
3. Ensure the stage returns serialisable data or objects with deterministic
   representations; results are stored in the pipeline run state for inspection.

```python
from dataclasses import dataclass
from Medical_KG_rev.orchestration.stages.contracts import StageContext

@dataclass(slots=True)
class ExampleStage:
    name: str

    def execute(self, ctx: StageContext, upstream: list[dict]) -> list[dict]:
        return [dict(item, stage=self.name) for item in upstream]
```

## 2. Provide Stage Metadata

Create a `StageMetadata` instance describing how the runtime should persist the
stage output and compute metrics.

```python
from Medical_KG_rev.orchestration.dagster.stage_registry import StageMetadata

def _handle_output(state: dict[str, object], stage_name: str, output: list[dict]) -> None:
    state["enriched_payloads"] = output

metadata = StageMetadata(
    stage_type="example-stage",
    state_key="enriched_payloads",
    output_handler=_handle_output,
    output_counter=len,
    description="Annotates payloads with the stage name",
)
```

## 3. Return a `StageRegistration`

Expose a callable (often named `register_<stage>()`) that returns a
`StageRegistration`. The callable receives the stage definition from the
pipeline YAML and should construct the stage instance with any configuration.

```python
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.stage_registry import StageRegistration


def register_example_stage() -> StageRegistration:
    def _builder(definition: StageDefinition) -> ExampleStage:
        return ExampleStage(name=definition.name)

    return StageRegistration(metadata=metadata, builder=_builder)
```

## 4. Declare an Entry Point

Third-party packages expose plugins via the `medical_kg.orchestration.stages`
entry point group. In `pyproject.toml`:

```toml
[project.entry-points."medical_kg.orchestration.stages"]
example-stage = "my_package.plugins:register_example_stage"
```

## 5. Reference the Plugin in Pipeline YAML

Pipelines opt into plugins through the `plugins` block:

```yaml
plugins:
  stages:
    - callable: "my_package.plugins:register_example_stage"
```

Per-stage `metadata_overrides` can adjust the registered metadata without
introducing a new plugin:

```yaml
stages:
  - name: example
    type: example-stage
    metadata_overrides:
      state_key: example_payloads
      dependencies: ["ingest"]
```

### Accessing Job Ledger State

Stages that need to inspect the job ledger can list `"@ledger"` in their
metadata dependencies. The runtime resolves this sentinel to the current ledger
snapshot (if the pipeline run is associated with a job ID) and includes it in
the upstream payload. For example, the built-in `gate` stage expects
`ledger.pdf_ir_ready` to be `true` before resuming downstream work.

## 6. Validate and Test

- Run `openspec validate <change-id> --strict` to confirm schema compliance.
- Add unit tests covering the stage contract and any orchestration adapters.
- Use the `tests/orchestration/test_stage_plugins_runtime.py` examples as a
  template for verifying behaviour.

> **Deprecation notice:** Direct imports of `build_default_stage_factory` or
> manual mutations of `StageRegistry` will be removed in the 2025-Q3 release.
> Always register stages via plugins or the `StageFactory.register_stage` API.
