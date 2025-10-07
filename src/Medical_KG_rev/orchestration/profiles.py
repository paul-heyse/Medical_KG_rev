"""Profile configuration and selection for ingestion/query pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from Medical_KG_rev.utils.errors import ProblemDetail

from .pipeline import PipelineConfig, PipelineDefinition, ProfileDefinition


@dataclass(slots=True)
class PipelineProfile:
    """Resolved pipeline profile with applied inheritance."""

    name: str
    ingestion: str
    query: str
    overrides: dict[str, Any]

    def ingestion_definition(self, config: PipelineConfig) -> PipelineDefinition:
        try:
            return config.ingestion[self.ingestion]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise ValueError(f"Unknown ingestion pipeline '{self.ingestion}' for profile {self.name}") from exc

    def query_definition(self, config: PipelineConfig) -> PipelineDefinition:
        try:
            return config.query[self.query]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise ValueError(f"Unknown query pipeline '{self.query}' for profile {self.name}") from exc


class ProfileManager:
    """Loads and resolves orchestration profiles from YAML configuration."""

    def __init__(self, config: PipelineConfig, profiles: Mapping[str, ProfileDefinition]) -> None:
        self.config = config
        self._profiles = dict(profiles)
        self._resolved: dict[str, PipelineProfile] = {}

    @classmethod
    def from_yaml(
        cls,
        pipeline_config: PipelineConfig,
        path: str | Path | None = None,
    ) -> ProfileManager:
        resolved_path = Path(path).expanduser() if path else None
        data: dict[str, Any]
        if resolved_path and resolved_path.exists():
            data = yaml.safe_load(resolved_path.read_text()) or {}
        else:
            data = {
                profile.name: profile.model_dump()
                for profile in pipeline_config.profiles.values()
            }
        profiles = {
            name: ProfileDefinition.model_validate(value)
            for name, value in data.items()
        }
        return cls(pipeline_config, profiles)

    def list_profiles(self) -> list[str]:
        return sorted(self._profiles)

    def get(self, name: str) -> PipelineProfile:
        if name in self._resolved:
            return self._resolved[name]
        definition = self._profiles.get(name)
        if not definition:
            raise KeyError(f"Profile '{name}' is not defined")
        resolved = self._resolve_definition(definition)
        self._resolved[name] = resolved
        return resolved

    def _resolve_definition(self, definition: ProfileDefinition) -> PipelineProfile:
        if definition.extends:
            base = self.get(definition.extends)
            ingestion = definition.ingestion or base.ingestion
            query = definition.query or base.query
            overrides = {**base.overrides, **definition.overrides}
        else:
            ingestion = definition.ingestion or next(iter(self.config.ingestion))
            query = definition.query or next(iter(self.config.query))
            overrides = dict(definition.overrides)
        return PipelineProfile(
            name=definition.name,
            ingestion=ingestion,
            query=query,
            overrides=overrides,
        )


class ProfileDetector:
    """Detects profile name based on document metadata and request hints."""

    def __init__(self, manager: ProfileManager, *, default_profile: str) -> None:
        self.manager = manager
        self.default_profile = default_profile

    def detect(
        self,
        *,
        explicit: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PipelineProfile:
        if explicit:
            return self.manager.get(explicit)
        metadata = metadata or {}
        source = str(metadata.get("source", "")).lower()
        mapping = {
            "pmc": "pmc",
            "pubmed": "pmc",
            "dailymed": "dailymed",
            "clinicaltrials": "clinicaltrials",
            "openalex": "pmc",
        }
        profile_name = mapping.get(source, self.default_profile)
        return self.manager.get(profile_name)


def apply_profile_overrides(context: dict[str, Any], profile: PipelineProfile) -> dict[str, Any]:
    merged = dict(context)
    merged.setdefault("profile", profile.name)
    merged.setdefault("config", {}).update(profile.overrides)
    return merged


def validate_profile_reference(profile: str, manager: ProfileManager) -> None:
    try:
        manager.get(profile)
    except KeyError as exc:  # pragma: no cover - validation error
        problem = ProblemDetail(
            title="Unknown profile",
            status=400,
            detail=str(exc),
        )
        raise ValueError(problem.to_response()) from exc


__all__ = [
    "PipelineProfile",
    "ProfileDetector",
    "ProfileManager",
    "apply_profile_overrides",
    "validate_profile_reference",
]
