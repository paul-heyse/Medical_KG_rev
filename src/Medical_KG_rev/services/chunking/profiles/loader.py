"""Profile loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import Profile


class ProfileNotFoundError(RuntimeError):
    """Raised when a requested profile cannot be located."""


class ProfileRepository:
    """Loads and caches chunking profiles from YAML files."""

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory or DEFAULT_PROFILE_DIR
        self._cache: dict[str, Profile] = {}

    def get(self, profile_name: str) -> Profile:
        if profile_name in self._cache:
            return self._cache[profile_name]
        profile_path = self._resolve_path(profile_name)
        if not profile_path.exists():
            raise ProfileNotFoundError(
                f"Chunking profile '{profile_name}' not found in {self._directory}"
            )
        data = yaml.safe_load(profile_path.read_text()) or {}
        profile = Profile.model_validate(data)
        self._cache[profile.name] = profile
        return profile

    def _resolve_path(self, profile_name: str) -> Path:
        filename = f"{profile_name}.yaml"
        return self._directory / filename


DEFAULT_PROFILE_DIR = Path(__file__).resolve().parents[4] / "config" / "chunking" / "profiles"


def default_loader() -> ProfileRepository:
    return ProfileRepository()
