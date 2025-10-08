from Medical_KG_rev.orchestration.dagster.configuration import PipelineConfigLoader


def test_pdf_pipeline_declares_gate_configuration() -> None:
    loader = PipelineConfigLoader("config/orchestration/pipelines")
    topology = loader.load("pdf-two-phase")

    stage_names = {stage.name for stage in topology.stages}
    assert {"download", "gate_pdf_ir_ready"}.issubset(stage_names)

    assert topology.gates, "expected gate definitions for pdf-two-phase"
    gate = next(g for g in topology.gates if g.name == "pdf_ir_ready")
    assert gate.resume_stage == "chunk"
    assert gate.condition.field == "pdf_ir_ready"
    assert gate.condition.equals is True
    assert gate.condition.timeout_seconds == 900
    assert gate.condition.poll_interval_seconds == 10.0
