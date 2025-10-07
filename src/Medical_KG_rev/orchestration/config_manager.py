"""Pipeline configuration management with hot-reload and versioning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .pipeline import PipelineConfig


@dataclass
class PipelineConfigManager:
    config_path: Path
    history_dir: Path = field(init=False)
    _config: PipelineConfig = field(init=False)
    _mtime: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.config_path = self.config_path.expanduser()
        self.history_dir = self.config_path.parent / "versions"
        self._config = PipelineConfig.from_yaml(self.config_path)
        if self.config_path.exists():
            self._mtime = self.config_path.stat().st_mtime
            self._snapshot()

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def reload(self) -> PipelineConfig | None:
        """Reload configuration from disk if it changed."""

        if not self.config_path.exists():
            return None
        mtime = self.config_path.stat().st_mtime
        if self._mtime and mtime <= self._mtime:
            return None
        self._config = PipelineConfig.from_yaml(self.config_path)
        self._mtime = mtime
        self._snapshot()
        return self._config

    def migrate(self, transform: Callable[[dict[str, Any]], dict[str, Any]]) -> PipelineConfig:
        """Apply a migration function to the on-disk configuration."""

        raw = self.config_path.read_text(encoding="utf-8") if self.config_path.exists() else "{}"
        data = transform(PipelineConfig.from_yaml(None, text=raw).model_dump(mode="python"))
        serialised = PipelineConfig.model_validate(data)
        self.config_path.write_text(
            yaml.safe_dump(serialised.model_dump(mode="python"), sort_keys=False),
            encoding="utf-8",
        )
        self._config = serialised
        self._snapshot()
        return self._config

    def _snapshot(self) -> None:
        if not self.config_path.exists():
            return
        self.history_dir.mkdir(parents=True, exist_ok=True)
        version = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        snapshot_path = self.history_dir / f"{version}.yaml"
        snapshot_path.write_text(self.config_path.read_text(encoding="utf-8"), encoding="utf-8")


__all__ = ["PipelineConfigManager"]
