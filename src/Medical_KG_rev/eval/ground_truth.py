"""Ground truth dataset loading utilities for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import json


@dataclass(slots=True)
class GroundTruthRecord:
    query_id: str
    query: str
    relevant_documents: Sequence[str]
    judgments: Mapping[str, float]


class GroundTruthManager:
    """Loads ground truth datasets from JSONL files and caches them in memory."""

    def __init__(self) -> None:
        self._datasets: dict[str, list[GroundTruthRecord]] = {}

    def load(self, name: str, path: str | Path) -> list[GroundTruthRecord]:
        resolved = Path(path).expanduser()
        records: list[GroundTruthRecord] = []
        with resolved.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                record = GroundTruthRecord(
                    query_id=str(payload["query_id"]),
                    query=str(payload["query"]),
                    relevant_documents=list(payload.get("relevant_documents", [])),
                    judgments=dict(payload.get("judgments", {})),
                )
                records.append(record)
        self._datasets[name] = records
        return records

    def dataset(self, name: str) -> list[GroundTruthRecord]:
        try:
            return self._datasets[name]
        except KeyError as exc:
            raise KeyError(f"Ground truth dataset '{name}' has not been loaded") from exc

    def queries(self, name: str) -> Iterable[GroundTruthRecord]:
        return iter(self.dataset(name))


__all__ = ["GroundTruthManager", "GroundTruthRecord"]
