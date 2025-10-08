import pytest

from Medical_KG_rev.orchestration.dagster.configuration import PipelineConfigLoader


def test_pdf_two_phase_pipeline_configuration() -> None:
    loader = PipelineConfigLoader()
    topology = loader.load("pdf-two-phase", force=True)

    assert topology.name == "pdf-two-phase"
    ingest = next(stage for stage in topology.stages if stage.name == "ingest")
    assert ingest.config["adapter"] == "openalex"
    assert "best_oa_location" in ingest.config["parameters"]["include"]

    download = next(stage for stage in topology.stages if stage.name == "download")
    assert download.config["storage"]["base_path"] == "/var/lib/medical-kg/pdfs"
    assert any(extractor["path"].endswith("pdf_url") for extractor in download.config["url_extractors"])

    gate = next(stage for stage in topology.stages if stage.name == "gate_pdf_ir_ready")
    assert gate.config["field"] == "pdf_ir_ready"
    assert gate.config["resume_stage"] == "chunk"
    assert pytest.approx(gate.config["timeout_seconds"], rel=0.0) == 1800.0

    gate_def = next(g for g in topology.gates if g.name == "pdf_ir_ready")
    assert gate_def.resume_stage == "chunk"
    assert gate_def.condition.field == "pdf_ir_ready"
