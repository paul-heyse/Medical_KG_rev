"""Lightweight FAISS-like index backed by NumPy for tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class FAISSIndex:
    dimension: int
    vectors: list[np.ndarray] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    metadata: list[Mapping[str, object]] = field(default_factory=list)

    def add(
        self, vector_id: str, vector: Sequence[float], metadata: Mapping[str, object] | None = None
    ) -> None:
        array = np.asarray(vector, dtype=float)
        if array.shape != (self.dimension,):
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {array.shape}"
            )
        self.vectors.append(array)
        self.ids.append(vector_id)
        self.metadata.append(metadata or {})

    def search(
        self, query_vector: Sequence[float], k: int = 5
    ) -> list[tuple[str, float, Mapping[str, object]]]:
        if not self.vectors:
            return []
        query = np.asarray(query_vector, dtype=float)
        if query.shape != (self.dimension,):
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {query.shape}"
            )
        matrix = np.vstack(self.vectors)
        scores = matrix @ query
        indices = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i]), self.metadata[i]) for i in indices]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_bytes(
            json.dumps(
                {
                    "dimension": self.dimension,
                    "ids": self.ids,
                    "vectors": [vector.tolist() for vector in self.vectors],
                    "metadata": self.metadata,
                }
            ).encode("utf-8")
        )

    @classmethod
    def load(cls, path: str | Path) -> FAISSIndex:
        data = json.loads(Path(path).read_text("utf-8"))
        index = cls(dimension=data["dimension"])
        for vector_id, vector_values, metadata in zip(
            data["ids"], data["vectors"], data["metadata"], strict=False
        ):
            index.add(vector_id, vector_values, metadata)
        return index

    def clear(self) -> None:
        self.vectors.clear()
        self.ids.clear()
        self.metadata.clear()
