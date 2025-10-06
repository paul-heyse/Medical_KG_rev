"""Parser for declarative adapter configuration files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class AdapterConfig:
    name: str
    source: str
    requests: Any
    mapping: Dict[str, Any]


def load_adapter_config(path: Path) -> AdapterConfig:
    data = yaml.safe_load(path.read_text())
    if not data:
        raise ValueError("Adapter configuration is empty")
    return AdapterConfig(
        name=data.get("name") or path.stem,
        source=data["source"],
        requests=data.get("requests", []),
        mapping=data.get("mapping", {}),
    )
