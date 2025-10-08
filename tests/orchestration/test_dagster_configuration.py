import pytest

from Medical_KG_rev.orchestration.dagster.configuration import (
    PipelineTopologyConfig,
    StageDefinition,
    build_stage_dependency_graph,
    derive_stage_execution_order,
)


def test_build_stage_dependency_graph_infers_typed_edges() -> None:
    stages = [
        StageDefinition(name="ingest", type="ingest"),
        StageDefinition(name="chunk", type="chunk"),
        StageDefinition(name="embed", type="embed"),
    ]

    graph = build_stage_dependency_graph(stages)
    assert graph["embed"] == {"chunk"}
    assert graph["chunk"] == {"ingest"}

    order = derive_stage_execution_order(stages)
    assert order.index("chunk") < order.index("embed")


def test_stage_type_without_provider_raises_validation_error() -> None:
    stages = [
        StageDefinition(name="ingest", type="ingest"),
        StageDefinition(name="embed", type="embed"),
    ]

    with pytest.raises(ValueError):
        derive_stage_execution_order(stages)


def test_pipeline_topology_config_uses_typed_dependency_resolution() -> None:
    config = PipelineTopologyConfig(
        name="typed-pipeline",
        version="2025-01-01",
        stages=[
            StageDefinition(name="ingest", type="ingest"),
            StageDefinition(name="chunk", type="chunk"),
            StageDefinition(name="embed", type="embed"),
        ],
    )

    order = derive_stage_execution_order(config.stages)
    assert order == ["ingest", "chunk", "embed"]
