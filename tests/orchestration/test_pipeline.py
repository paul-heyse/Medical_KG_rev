from __future__ import annotations

from pathlib import Path

import pytest

from Medical_KG_rev.orchestration.pipeline import (
    ParallelExecutor,
    PipelineConfig,
    PipelineContext,
    PipelineExecutor,
    StageFailure,
)


class DummyStage:
    def __init__(self, name: str, *, raise_error: bool = False) -> None:
        self.name = name
        self.timeout_ms = None
        self.raise_error = raise_error

    def execute(self, context: PipelineContext) -> PipelineContext:
        if self.raise_error:
            raise StageFailure("boom", stage=self.name, status=500)
        context.data.setdefault("visited", []).append(self.name)
        return context


def test_pipeline_executor_runs_in_order() -> None:
    context = PipelineContext(tenant_id="tenant", operation="ingest")
    executor = PipelineExecutor([
        DummyStage("first"),
        DummyStage("second"),
    ], operation="ingest", pipeline="test")

    result = executor.run(context)

    assert result.data["visited"] == ["first", "second"]
    assert "first" in result.stage_timings


def test_pipeline_executor_handles_failure() -> None:
    context = PipelineContext(tenant_id="tenant", operation="ingest")
    executor = PipelineExecutor([
        DummyStage("first"),
        DummyStage("second", raise_error=True),
    ], operation="ingest", pipeline="test")

    with pytest.raises(StageFailure):
        executor.run(context)

    assert context.partial is False  # failure happens before context mutated


def test_pipeline_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = """
    version: "1.0"
    ingestion:
      default:
        name: default
        stages:
          - name: stage-a
            kind: chunk
    query:
      q:
        name: q
        stages:
          - name: step
            kind: retrieval
    profiles:
      p:
        name: p
        ingestion: default
        query: q
    """
    config_path = tmp_path / "pipelines.yaml"
    config_path.write_text(yaml_content)

    config = PipelineConfig.from_yaml(config_path)

    assert "default" in config.ingestion
    assert config.profiles["p"].ingestion == "default"


def test_parallel_executor_collects_results() -> None:
    executor = ParallelExecutor(max_workers=2)
    context = PipelineContext(tenant_id="tenant", operation="test")

    def task_one() -> int:
        return 1

    def task_two() -> int:
        raise StageFailure("oops", stage="two", status=500)

    results = executor.run({"one": task_one, "two": task_two}, correlation_id=context.correlation_id)

    assert results["one"].value == 1
    assert results["two"].error is not None
