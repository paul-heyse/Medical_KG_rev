"""Compression utilities and orchestration for vector payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .models import CompressionPolicy



class CompressionError(RuntimeError):
    """Raised when a compression policy cannot be satisfied."""


class Quantizer(Protocol):
    """Callable protocol for quantisation functions."""

    def __call__(self, vectors: Sequence[Sequence[float]]) -> dict[str, object]:
        """Return a compressed representation for the provided vectors."""


def _as_float32(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    if not vectors:
        raise CompressionError("compression requires at least one vector")
    return np.asarray(vectors, dtype=np.float32)


def quantize_int8(vectors: Sequence[Sequence[float]]) -> dict[str, object]:
    """Symmetric per-vector int8 quantisation with scale tracking."""
    matrix = _as_float32(vectors)
    scales = np.maximum(np.abs(matrix).max(axis=1, keepdims=True), 1e-6)
    quantized = np.clip(np.round(matrix / scales * 127.0), -128, 127).astype(np.int8)
    return {"values": quantized, "scales": scales.astype(np.float32)}


def quantize_fp16(vectors: Sequence[Sequence[float]]) -> dict[str, object]:
    """Lossy fp16 conversion retaining fp32 restore metadata."""
    matrix = _as_float32(vectors)
    return {"values": matrix.astype(np.float16), "dtype": "float16"}


def binary_quantize(vectors: Sequence[Sequence[float]]) -> dict[str, object]:
    """Convert vectors into packed binary signatures for Hamming search."""
    matrix = _as_float32(vectors)
    thresholded = (matrix >= 0).astype(np.uint8)
    packed = np.packbits(thresholded, axis=1)
    return {"values": packed, "bits": thresholded}


def _split_subvectors(matrix: np.ndarray, m: int) -> np.ndarray:
    if matrix.shape[1] % m:
        raise CompressionError("pq requires dimensionality divisible by m")
    dim = matrix.shape[1] // m
    return matrix.reshape(matrix.shape[0], m, dim)


def train_pq(vectors: Sequence[Sequence[float]], *, m: int, nbits: int) -> dict[str, object]:
    """Train a simple product quantiser via k-means on subvectors."""
    matrix = _as_float32(vectors)
    subvectors = _split_subvectors(matrix, m)
    codebooks: list[np.ndarray] = []
    for subspace in range(m):
        data = subvectors[:, subspace, :]
        clusters = min(2**nbits, data.shape[0])
        # Initialise centroids deterministically for reproducibility
        indices = np.linspace(0, data.shape[0] - 1, clusters, dtype=int)
        centroids = data[indices]
        for _ in range(5):  # limited iterations for efficiency
            distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(distances, axis=1)
            for centroid_id in range(clusters):
                members = data[assignments == centroid_id]
                if members.size:
                    centroids[centroid_id] = members.mean(axis=0)
        codebooks.append(centroids.astype(np.float32))
    return {"codebooks": codebooks, "m": m, "nbits": nbits}


def apply_pq(vectors: Sequence[Sequence[float]], codebooks: Sequence[np.ndarray]) -> np.ndarray:
    matrix = _as_float32(vectors)
    m = len(codebooks)
    subvectors = _split_subvectors(matrix, m)
    codes = np.empty((matrix.shape[0], m), dtype=np.uint8)
    for subspace, centroids in enumerate(codebooks):
        distances = np.linalg.norm(
            subvectors[:, subspace, :, None] - centroids[None, None, :, :], axis=2
        )
        codes[:, subspace] = np.argmin(distances, axis=2)[:, 0]
    return codes


def learn_opq_rotation(vectors: Sequence[Sequence[float]], *, m: int) -> dict[str, object]:
    """Estimate an orthogonal rotation prior to PQ (OPQ)."""
    matrix = _as_float32(vectors)
    cov = np.cov(matrix, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    rotation = eigvecs[:, order][:, : m * (matrix.shape[1] // m)]
    return {"rotation": rotation.astype(np.float32), "m": m}


def two_stage_reorder(
    coarse: Sequence[tuple[str, float, dict[str, object]]],
    reorder_embeddings: Mapping[str, Sequence[float]],
    *,
    top_k: int,
    metric: str = "cosine",
    query_vector: Sequence[float] | None = None,
) -> list[tuple[str, float, dict[str, object]]]:
    """Reorder coarse ANN hits using float vectors."""
    if metric not in {"cosine", "l2"}:
        raise CompressionError(f"unsupported metric '{metric}' for reorder")
    if not coarse:
        return []
    if query_vector is None:
        for _, _, payload in coarse:
            candidate = payload.get("query") if isinstance(payload, Mapping) else None
            if candidate is not None:
                query_vector = candidate
                break
    if query_vector is None:
        raise CompressionError("query vector required for reorder")
    query_array = np.asarray(query_vector, dtype=np.float32)
    reordered: list[tuple[str, float, dict[str, object]]] = []
    for identifier, _score, payload in coarse:
        vector_data = reorder_embeddings.get(identifier)
        if vector_data is None:
            continue
        vector = np.asarray(vector_data, dtype=np.float32)
        if vector.size == 0:
            reordered.append((identifier, _score, payload))
            continue
        if metric == "cosine":
            numerator = float(np.dot(query_array, vector))
            denom = float(np.linalg.norm(query_array) * np.linalg.norm(vector)) or 1e-6
            score = numerator / denom
        else:
            diff = query_array - vector
            score = -float(np.dot(diff, diff))
        reordered.append((identifier, score, payload))
    reordered.sort(key=lambda item: item[1], reverse=True)
    return reordered[:top_k]


@dataclass(slots=True)
class CompressionManager:
    """Routes compression policies to the appropriate quantisers."""

    int8_quantizer: Quantizer = quantize_int8
    fp16_quantizer: Quantizer = quantize_fp16

    def compress(
        self,
        vectors: Sequence[Sequence[float]],
        policy: CompressionPolicy,
    ) -> dict[str, object]:
        if not policy or policy.kind == "none":
            return {"values": _as_float32(vectors)}
        if policy.kind == "int8":
            return self.int8_quantizer(vectors)
        if policy.kind == "fp16":
            return self.fp16_quantizer(vectors)
        if policy.kind == "binary":
            return binary_quantize(vectors)
        if policy.kind == "pq":
            if not policy.pq_m or not policy.pq_nbits:
                raise CompressionError("pq policy requires m and nbits")
            return {
                "pq": train_pq(vectors, m=policy.pq_m, nbits=policy.pq_nbits),
                "values": vectors,
            }
        if policy.kind == "opq":
            if not policy.pq_m or not policy.pq_nbits or not policy.opq_m:
                raise CompressionError("opq policy requires m, nbits, and opq_m")
            rotation = learn_opq_rotation(vectors, m=policy.opq_m)
            rotated = (rotation["rotation"].T @ _as_float32(vectors).T).T
            pq = train_pq(rotated, m=policy.pq_m, nbits=policy.pq_nbits)
            return {"rotation": rotation, "pq": pq, "values": vectors}
        raise CompressionError(f"unsupported compression kind '{policy.kind}'")

    def validate(
        self,
        vectors: Sequence[Sequence[float]],
        policy: CompressionPolicy,
    ) -> None:
        _ = self.compress(vectors, policy)

    def decompress(
        self,
        payload: dict[str, object],
    ) -> np.ndarray:
        values = payload.get("values")
        if isinstance(values, np.ndarray):
            return values.astype(np.float32)
        if isinstance(values, list):
            return np.asarray(values, dtype=np.float32)
        raise CompressionError("unknown compression payload")


def batch_vectors(
    records: Iterable[Sequence[float]], *, batch_size: int
) -> Iterable[list[Sequence[float]]]:
    """Yield fixed-size batches suitable for GPU uploads."""
    batch: list[Sequence[float]] = []
    for vector in records:
        batch.append(vector)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
