from __future__ import annotations

import time

import pytest

from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.stages.plugins import (
    StagePlugin,
    StagePluginHealth,
    StagePluginManager,
    StagePluginRegistration,
    StagePluginResources,
)


class _RecordingPlugin(StagePlugin):
    def __init__(
        self,
        name: str,
        *,
        dependencies: tuple[str, ...] = (),
        fail_build: bool = False,
    ) -> None:
        super().__init__(plugin_name=name, dependencies=dependencies)
        self.fail_build = fail_build
        self.invocations: list[str] = []
        self.cleaned = False

    def registrations(self, resources: StagePluginResources) -> tuple[StagePluginRegistration, ...]:
        def builder(definition: StageDefinition, _: StagePluginResources) -> dict[str, str]:
            self.invocations.append(definition.name)
            if self.fail_build:
                raise RuntimeError(f"{self.name} failed")
            return {"provider": self.name, "stage": definition.name}

        return (
            self.create_registration(
                stage_type="custom",
                builder=builder,
                capabilities=("test",),
            ),
        )

    def cleanup(self) -> None:
        self.cleaned = True

    def health_check(self) -> StagePluginHealth:
        return StagePluginHealth(status="ok", detail=self.name, timestamp=time.time())


@pytest.fixture()
def manager() -> StagePluginManager:
    resources = StagePluginResources(
        adapter_manager=AdapterPluginManager(),
        pipeline_resource=object(),
    )
    return StagePluginManager(resources=resources)


def _definition(name: str) -> StageDefinition:
    return StageDefinition(name=name, type="custom", config={})


def test_stage_plugin_manager_orders_dependencies(manager: StagePluginManager) -> None:
    primary = _RecordingPlugin("alpha", fail_build=True)
    dependent = _RecordingPlugin("beta", dependencies=("alpha.custom",))
    manager.register(primary)
    manager.register(dependent)

    stage = manager.build_stage(_definition("example"))

    assert stage == {"provider": "beta", "stage": "example"}
    assert len(primary.invocations) >= 1
    assert dependent.invocations == ["example"]


def test_stage_plugin_manager_unregister_cleans(manager: StagePluginManager) -> None:
    plugin = _RecordingPlugin("gamma")
    manager.register(plugin)

    assert "custom" in manager.available_stage_types()
    manager.unregister(plugin)

    assert plugin.cleaned is True
    assert manager.available_stage_types() == ()


def test_stage_plugin_manager_health_snapshot(manager: StagePluginManager) -> None:
    plugin = _RecordingPlugin("theta")
    manager.register(plugin)

    report = manager.check_health()

    key = f"{plugin.name}.custom"
    assert key in report
    assert report[key].status == "ok"

    descriptions = manager.describe_plugins()
    assert any(entry["name"] == key and entry["status"] == "initialized" for entry in descriptions)
