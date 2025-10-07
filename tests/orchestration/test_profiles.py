from __future__ import annotations

from Medical_KG_rev.orchestration.pipeline import PipelineConfig, ProfileDefinition
from Medical_KG_rev.orchestration.profiles import ProfileDetector, ProfileManager


def build_config() -> PipelineConfig:
    return PipelineConfig(
        version="1.0",
        ingestion={
            "default": {
                "name": "default",
                "stages": [],
            }
        },
        query={
            "hybrid": {
                "name": "hybrid",
                "stages": [],
            }
        },
        profiles={
            "base": ProfileDefinition(name="base", ingestion="default", query="hybrid"),
            "child": ProfileDefinition(name="child", extends="base", overrides={"top_k": 5}),
        },
    )


def test_profile_inheritance() -> None:
    config = build_config()
    manager = ProfileManager(config, config.profiles)

    profile = manager.get("child")

    assert profile.ingestion == "default"
    assert profile.overrides["top_k"] == 5


def test_profile_detector_defaults() -> None:
    config = build_config()
    manager = ProfileManager(config, config.profiles)
    detector = ProfileDetector(manager, default_profile="base")

    profile = detector.detect(metadata={"source": "DailyMed"})

    assert profile.name == "base"
