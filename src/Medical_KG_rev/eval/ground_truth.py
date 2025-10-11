"""Ground truth dataset loading utilities for retrieval evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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

    def create_annotation_template(
        self,
        name: str,
        queries: Sequence[str],
        directory: str | Path,
    ) -> Path:
        """Create a JSONL template for manual relevance annotation."""
        target_dir = Path(directory).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = target_dir / f"{name}-{timestamp}-template.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index, query in enumerate(queries, start=1):
                payload = {
                    "query_id": f"{name}-{index:04d}",
                    "query": query,
                    "relevant_documents": [],
                    "judgments": {},
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    def save(self, name: str, records: Sequence[GroundTruthRecord], directory: str | Path) -> Path:
        """Persist a dataset to a versioned JSONL file."""
        target_dir = Path(directory).expanduser() / name
        target_dir.mkdir(parents=True, exist_ok=True)
        version = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = target_dir / f"{version}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                payload = {
                    "query_id": record.query_id,
                    "query": record.query,
                    "relevant_documents": list(record.relevant_documents),
                    "judgments": dict(record.judgments),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._datasets[name] = list(records)
        return path


__all__ = ["GroundTruthManager", "GroundTruthRecord"]
