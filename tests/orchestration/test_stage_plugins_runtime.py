from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.stage_plugins import (
    GateConditionError,
    register_download_stage,
    register_gate_stage,
)
from Medical_KG_rev.orchestration.stages.contracts import StageContext


def _stage_definition(name: str, stage_type: str, config: dict | None = None) -> StageDefinition:
    return StageDefinition.model_validate({
        "name": name,
        "type": stage_type,
        "config": config or {},
    })


def test_download_stage_normalises_sources():
    registration = register_download_stage()
    definition = _stage_definition(
        "download", "download", config={"sources": [{"url": "a"}, "b", {"kind": "mirror", "url": "c"}]}
    )
    stage = registration.builder(definition)
    ctx = StageContext(tenant_id="tenant")

    result = stage.execute(ctx, upstream={"fallback": True})

    assert len(result) == 3
    assert result[0]["source"] == {"url": "a"}
    assert result[1]["source"] == {"value": "b"}
    assert result[2]["source"] == {"kind": "mirror", "url": "c"}


def test_gate_stage_conditions_pass_and_fail():
    registration = register_gate_stage()
    definition = _stage_definition(
        "gate", "gate", config={"conditions": [{"key": "ledger.pdf_ir_ready", "expected": True}]}
    )
    stage = registration.builder(definition)
    ctx = StageContext(tenant_id="tenant")

    stage.execute(ctx, upstream={"ledger": SimpleNamespace(pdf_ir_ready=True)})

    with pytest.raises(GateConditionError):
        stage.execute(ctx, upstream={"ledger": SimpleNamespace(pdf_ir_ready=False)})


def test_gate_stage_timeout_detection():
    registration = register_gate_stage()
    definition = _stage_definition(
        "gate", "gate", config={"timeout_seconds": 5, "conditions": ["value"]}
    )
    stage = registration.builder(definition)
    ctx = StageContext(tenant_id="tenant", metadata={"gate_elapsed_seconds": 10})

    with pytest.raises(GateConditionError) as excinfo:
        stage.execute(ctx, upstream=True)

    assert "timed out" in str(excinfo.value)
