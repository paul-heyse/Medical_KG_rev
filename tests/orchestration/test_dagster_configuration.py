from __future__ import annotations

import json
from pathlib import Path

import pytest

from Medical_KG_rev.orchestration.dagster.configuration import (
    PipelineConfigLoader,
    PipelineTopologyConfig,
    ResiliencePolicyLoader,
)
from Medical_KG_rev.orchestration.stages.contracts import StageContext


def _write(tmp_path: Path, name: str, payload: str) -> Path:
    path = tmp_path / name
    path.write_text(payload)
    return path


def test_stage_context_metadata_helpers() -> None:
    ctx = StageContext(tenant_id="tenant-a", doc_id="doc-1", correlation_id="corr-1")
    updated = ctx.with_metadata(source="ingest", priority="high")

    assert updated.metadata == {"source": "ingest", "priority": "high"}
    assert updated.tenant_id == ctx.tenant_id


def test_pipeline_topology_validation_accepts_acyclic(tmp_path: Path) -> None:
    payload = """
name: demo
version: "2025-01-01"
stages:
  - name: ingest
    type: ingest
    policy: default
  - name: parse
    type: parse
    policy: default
    depends_on:
      - ingest
"""
    config = PipelineTopologyConfig.model_validate(json.loads(json.dumps({
        "name": "demo",
        "version": "2025-01-01",
        "stages": [
            {"name": "ingest", "type": "ingest", "policy": "default"},
            {"name": "parse", "type": "parse", "policy": "default", "depends_on": ["ingest"]},
        ],
    })))
    assert config.name == "demo"
    loader = PipelineConfigLoader(tmp_path)
    _write(tmp_path, "demo.yaml", payload)
    loaded = loader.load("demo")
    assert loaded.version == "2025-01-01"
    assert [stage.name for stage in loaded.stages] == ["ingest", "parse"]


def test_pipeline_topology_cycle_detection(tmp_path: Path) -> None:
    payload = """
name: cyclic
version: "2025-01-01"
stages:
  - name: a
    type: ingest
    policy: default
    depends_on:
      - c
  - name: b
    type: parse
    policy: default
    depends_on:
      - a
  - name: c
    type: chunk
    policy: default
    depends_on:
      - b
"""
    loader = PipelineConfigLoader(tmp_path)
    _write(tmp_path, "cyclic.yaml", payload)
    with pytest.raises(ValueError):
        loader.load("cyclic")


def test_pipeline_loader_reload_on_change(tmp_path: Path) -> None:
    loader = PipelineConfigLoader(tmp_path)
    _write(
        tmp_path,
        "auto.yaml",
        """
name: auto
version: "2025-01-01"
stages:
  - name: ingest
    type: ingest
    policy: default
""",
    )
    first = loader.load("auto")
    assert first.version == "2025-01-01"
    _write(
        tmp_path,
        "auto.yaml",
        """
name: auto
version: "2025-02-01"
stages:
  - name: ingest
    type: ingest
    policy: default
""",
    )
    second = loader.load("auto", force=True)
    assert second.version == "2025-02-01"


def test_resilience_policy_loader(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "resilience.yaml",
        """
policies:
  test:
    max_attempts: 2
    timeout_seconds: 5
    backoff:
      strategy: none
      initial: 0.0
      maximum: 0.0
      jitter: false
""",
    )
    loader = ResiliencePolicyLoader(tmp_path / "resilience.yaml")
    policies = loader.load()
    assert "test" in policies
    policy = loader.get("test")
    wrapped = loader.apply("test", "ingest", lambda value: value)
    assert wrapped(123) == 123


def test_resilience_policy_retry(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "resilience.yaml",
        """
policies:
  flaky:
    max_attempts: 3
    timeout_seconds: 5
    backoff:
      strategy: none
      initial: 0.0
      maximum: 0.0
      jitter: false
""",
    )
    loader = ResiliencePolicyLoader(tmp_path / "resilience.yaml")
    loader.load()

    calls = {"count": 0}

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("boom")
        return "ok"

    wrapped = loader.apply("flaky", "embed", flaky)
    assert wrapped() == "ok"
    assert calls["count"] == 2


def test_resilience_policy_invalid(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "resilience.yaml",
        """
policies:
  bad:
    max_attempts: 0
    timeout_seconds: 5
""",
    )
    loader = ResiliencePolicyLoader(tmp_path / "resilience.yaml")
    with pytest.raises(ValueError):
        loader.load()
