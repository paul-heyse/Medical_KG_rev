"""GPU-backed FAISS index wrapper with metadata persistence.

Key Responsibilities:
    - Enforce GPU availability for FAISS operations.
    - Provide serialization helpers for index metadata.
    - Expose convenience methods for vector add/search operations (legacy stub).

Collaborators:
    - Downstream: FAISS GPU libraries, filesystem for metadata persistence.
    - Upstream: Retrieval services performing dense vector search.

Side Effects:
    - Allocates GPU memory when configured to use GPU resources.
    - Reads/writes JSON metadata alongside index files.

Thread Safety:
    - Not thread-safe: FAISS index operations mutate shared state.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import json

import faiss  # type: ignore import-not-found
import numpy as np

from Medical_KG_rev.services import GpuNotAvailableError


_METADATA_SUFFIX = ".meta.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return (
        value if isinstance(value, (str, int, float, bool, type(None), list, dict)) else str(value)
    )


class FAISSIndex:
    """Thin wrapper around FAISS with GPU enforcement and metadata storage.

    Attributes:
        dimension: Dimensionality of vectors managed by the index.
        metric: FAISS distance metric identifier (e.g., ``ip`` or ``l2``).
        use_gpu: Whether GPU acceleration is enabled.
        gpu_device: GPU device ordinal used when ``use_gpu`` is true.
        index: Underlying FAISS index instance.
    """

    def __init__(
        self,
        dimension: int,
        *,
        metric: str = "ip",
        use_gpu: bool = True,
        gpu_device: int = 0,
    ) -> None:
        """Create a FAISS index instance with optional GPU acceleration.

        Args:
            dimension: Dimensionality of vectors stored in the index.
            metric: FAISS metric identifier (e.g., ``ip`` or ``l2``).
            use_gpu: Whether to allocate the index on a GPU device.
            gpu_device: GPU device ordinal used when GPU mode is enabled.

        Raises:
            ValueError: If an invalid dimension or metric is provided.
            GpuNotAvailableError: When GPU mode is requested without GPU support.
        """
        if dimension <= 0:
            raise ValueError("FAISS dimension must be positive")
        self.dimension = int(dimension)
        self.metric = metric
        self._use_gpu = use_gpu
        self._gpu_device = gpu_device
        self._resources: faiss.GpuResources | None = None
        self._gpu_index: faiss.Index | None = None
        self._cpu_index = self._build_cpu_index()
        self._dirty_gpu = True
        self._id_to_internal: dict[str, int] = {}
        self._internal_to_id: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self.ids: list[str] = []
        self._next_internal_id: int = 1
        if self._use_gpu:
            self._ensure_gpu_available()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_cpu_index(self) -> faiss.Index:
        if self.metric == "ip":
            base = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            base = faiss.IndexFlatL2(self.dimension)
        else:  # pragma: no cover - defensive programming
            raise ValueError(f"Unsupported FAISS metric '{self.metric}'")
        return faiss.IndexIDMap2(base)

    def _ensure_gpu_available(self) -> None:
        if faiss.get_num_gpus() <= 0:
            raise GpuNotAvailableError("FAISS GPU requested but no CUDA devices detected")

    def _gpu_index_instance(self) -> faiss.Index:
        if not self._use_gpu:
            return self._cpu_index
        self._ensure_gpu_available()
        if self._resources is None:
            self._resources = faiss.StandardGpuResources()
        if self._gpu_index is None or self._dirty_gpu:
            cloned = faiss.clone_index(self._cpu_index)
            self._gpu_index = faiss.index_cpu_to_gpu(
                self._resources,
                self._gpu_device,
                cloned,
            )
            self._dirty_gpu = False
        return self._gpu_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(
        self,
        vector_id: str,
        vector: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(vector_id, str) or not vector_id:
            raise ValueError("Vector id must be a non-empty string")
        if vector_id in self._id_to_internal:
            self.remove(vector_id)
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {array.shape}"
            )
        internal_id = self._next_internal_id
        self._next_internal_id += 1
        self._cpu_index.add_with_ids(
            array.reshape(1, -1), np.asarray([internal_id], dtype=np.int64)
        )
        self._id_to_internal[vector_id] = internal_id
        self._internal_to_id[internal_id] = vector_id
        meta_dict = dict(metadata or {})
        self._metadata[vector_id] = meta_dict
        self.ids.append(vector_id)
        self._dirty_gpu = True

    def remove(self, vector_id: str) -> bool:
        internal_id = self._id_to_internal.pop(vector_id, None)
        if internal_id is None:
            return False
        selector = faiss.IDSelectorArray(np.asarray([internal_id], dtype=np.int64))
        self._cpu_index.remove_ids(selector)
        self._internal_to_id.pop(internal_id, None)
        self._metadata.pop(vector_id, None)
        try:
            self.ids.remove(vector_id)
        except ValueError:  # pragma: no cover - defensive cleanup
            self.ids = [vid for vid in self.ids if vid != vector_id]
        self._dirty_gpu = True
        return True

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 5,
    ) -> list[tuple[str, float, Mapping[str, Any]]]:
        if self._cpu_index.ntotal == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {query.shape}"
            )
        distances, indices = self._gpu_index_instance().search(query.reshape(1, -1), k)
        results: list[tuple[str, float, Mapping[str, Any]]] = []
        for score, internal_id in zip(distances[0], indices[0], strict=False):
            if internal_id == -1:
                continue
            vector_id = self._internal_to_id.get(int(internal_id))
            if vector_id is None:
                continue
            metadata = self._metadata.get(vector_id, {})
            results.append((vector_id, float(score), metadata))
        return results

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._cpu_index, str(target))
        metadata_payload = {
            "ids": self.ids,
            "id_to_internal": self._id_to_internal,
            "metadata": self._metadata,
            "next_id": self._next_internal_id,
            "metric": self.metric,
        }
        metadata_path = target.with_suffix(target.suffix + _METADATA_SUFFIX)
        metadata_path.write_text(
            json.dumps(metadata_payload, default=_json_default), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        use_gpu: bool = True,
        metric: str = "ip",
        gpu_device: int = 0,
    ) -> FAISSIndex:
        target = Path(path)
        cpu_index = faiss.read_index(str(target))
        instance = cls.__new__(cls)
        instance.dimension = cpu_index.d
        instance.metric = metric
        instance._use_gpu = use_gpu
        instance._gpu_device = gpu_device
        instance._resources = None
        instance._gpu_index = None
        instance._cpu_index = cpu_index
        instance._dirty_gpu = True
        instance._id_to_internal = {}
        instance._internal_to_id = {}
        instance._metadata = {}
        instance.ids = []
        instance._next_internal_id = 1
        metadata_path = target.with_suffix(target.suffix + _METADATA_SUFFIX)
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text("utf-8"))
            instance.ids = [str(value) for value in payload.get("ids", [])]
            id_to_internal = {
                str(key): int(value)
                for key, value in (payload.get("id_to_internal", {}) or {}).items()
            }
            instance._id_to_internal = id_to_internal
            instance._internal_to_id = {
                internal: vector_id for vector_id, internal in id_to_internal.items()
            }
            metadata_payload = payload.get("metadata", {}) or {}
            instance._metadata = {
                str(key): dict(value) if isinstance(value, Mapping) else dict(value)
                for key, value in metadata_payload.items()
            }
            instance._next_internal_id = int(
                payload.get("next_id", max(id_to_internal.values(), default=0) + 1)
            )
            instance.metric = str(payload.get("metric", instance.metric))
        else:  # pragma: no cover - defensive for manual migrations
            raise FileNotFoundError(f"Missing metadata file for FAISS index at {metadata_path}")
        if instance._use_gpu:
            instance._ensure_gpu_available()
        return instance

    def clear(self) -> None:
        self._cpu_index.reset()
        self.ids.clear()
        self._id_to_internal.clear()
        self._internal_to_id.clear()
        self._metadata.clear()
        self._next_internal_id = 1
        self._dirty_gpu = True

    @property
    def metadata(self) -> list[Mapping[str, Any]]:
        return [self._metadata.get(vector_id, {}) for vector_id in self.ids]

    @property
    def ntotal(self) -> int:
        return self._cpu_index.ntotal
