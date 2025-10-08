from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.orchestration.dagster.runtime import StageFactory, StageResolutionError
from Medical_KG_rev.orchestration.dagster.stage_registry import (
    StageMetadata,
    StageRegistration,
    StageRegistry,
    StageRegistryError,
)


def _builder(_: Any) -> object:
    return object()


def test_stage_metadata_rejects_invalid_state_key():
    with pytest.raises(StageRegistryError):
        StageMetadata(
            stage_type="invalid",
            state_key="123-key",
            output_handler=lambda *_: None,
            output_counter=lambda _: 0,
            description="invalid",
        )


def test_stage_registry_register_and_lookup():
    registry = StageRegistry()
    metadata = StageMetadata(
        stage_type="custom",
        state_key="result",
        output_handler=lambda state, _, output: state.update({"result": output}),
        output_counter=lambda output: 1 if output else 0,
        description="Custom stage",
    )
    registry.register(StageRegistration(metadata=metadata, builder=_builder))

    resolved_metadata = registry.get_metadata("custom")
    assert resolved_metadata.stage_type == "custom"
    builder = registry.get_builder("custom")
    instance = builder(SimpleNamespace(name="stage", stage_type="custom", config={}))
    assert instance is not None


def test_stage_registry_plugin_loader_registers_plugins():
    metadata = StageMetadata(
        stage_type="plugin-stage",
        state_key="value",
        output_handler=lambda state, _, output: state.update({"value": output}),
        output_counter=lambda output: int(output or 0),
        description="Plugin provided stage",
    )

    def _plugin():
        return StageRegistration(metadata=metadata, builder=_builder)

    registry = StageRegistry(plugin_loader=lambda: [_plugin])
    loaded = registry.load_plugins()
    assert "plugin-stage" in loaded
    assert registry.get_metadata("plugin-stage").description == "Plugin provided stage"


def test_stage_factory_raises_on_unknown_stage():
    registry = StageRegistry()
    factory = StageFactory(registry)
    with pytest.raises(StageResolutionError):
        factory.resolve("pipeline", SimpleNamespace(name="missing", stage_type="missing", config={}))
