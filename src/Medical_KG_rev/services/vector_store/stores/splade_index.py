"""SPLADE impact index storage implementation.

This module implements Lucene impact index storage for SPLADE vectors,
providing efficient storage and retrieval of sparse vectors with
quantized weights and impact scores.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from Medical_KG_rev.services.retrieval.splade_service import SPLADEVector
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
SPLADE_INDEX_SECONDS = Histogram(
    "splade_index_seconds",
    "Time spent on SPLADE index operations",
    ["operation", "status"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

SPLADE_INDEX_OPERATIONS = Counter(
    "splade_index_operations_total",
    "Total number of SPLADE index operations",
    ["operation", "status"],
)

SPLADE_INDEX_SIZE_BYTES = Histogram(
    "splade_index_size_bytes",
    "Size of SPLADE index in bytes",
    ["operation"],
    buckets=[1024, 10240, 102400, 1024000, 10240000, 102400000],
)


class SPLADEIndexManifest:
    """Manifest for SPLADE impact index metadata."""

    def __init__(
        self,
        index_path: str,
        model_name: str = "naver/splade-v3",
        tokenizer_name: str = "naver/splade-v3",
        sparsity_threshold: float = 0.01,
        quantization_scale: int = 1000,
        max_terms_per_chunk: int = 10000,
        created_at: str | None = None,
        version: str = "1.0",
    ):
        """Initialize SPLADE index manifest.

        Args:
            index_path: Path to the index directory
            model_name: Name of the SPLADE model used
            tokenizer_name: Name of the tokenizer used
            sparsity_threshold: Sparsity threshold applied
            quantization_scale: Quantization scale factor
            max_terms_per_chunk: Maximum terms per chunk
            created_at: Creation timestamp
            version: Index version

        """
        self.index_path = index_path
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.sparsity_threshold = sparsity_threshold
        self.quantization_scale = quantization_scale
        self.max_terms_per_chunk = max_terms_per_chunk
        self.created_at = created_at or time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        self.version = version

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "index_path": self.index_path,
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "sparsity_threshold": self.sparsity_threshold,
            "quantization_scale": self.quantization_scale,
            "max_terms_per_chunk": self.max_terms_per_chunk,
            "created_at": self.created_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SPLADEIndexManifest":
        """Create manifest from dictionary."""
        return cls(
            index_path=data["index_path"],
            model_name=data["model_name"],
            tokenizer_name=data["tokenizer_name"],
            sparsity_threshold=data["sparsity_threshold"],
            quantization_scale=data["quantization_scale"],
            max_terms_per_chunk=data["max_terms_per_chunk"],
            created_at=data["created_at"],
            version=data["version"],
        )

    def save(self, manifest_path: str) -> None:
        """Save manifest to file."""
        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, manifest_path: str) -> "SPLADEIndexManifest":
        """Load manifest from file."""
        with open(manifest_path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class SPLADEImpactIndex:
    """SPLADE impact index storage using Lucene-style impact format.

    This class implements efficient storage and retrieval of SPLADE vectors
    using Lucene impact index format with quantized weights and impact scores.
    """

    def __init__(
        self,
        index_path: str,
        model_name: str = "naver/splade-v3",
        tokenizer_name: str = "naver/splade-v3",
        sparsity_threshold: float = 0.01,
        quantization_scale: int = 1000,
        max_terms_per_chunk: int = 10000,
    ):
        """Initialize SPLADE impact index.

        Args:
            index_path: Path to the index directory
            model_name: Name of the SPLADE model used
            tokenizer_name: Name of the tokenizer used
            sparsity_threshold: Sparsity threshold applied
            quantization_scale: Quantization scale factor
            max_terms_per_chunk: Maximum terms per chunk

        """
        self.index_path = Path(index_path)
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.sparsity_threshold = sparsity_threshold
        self.quantization_scale = quantization_scale
        self.max_terms_per_chunk = max_terms_per_chunk

        # Create index directory
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Index storage files
        self.vectors_file = self.index_path / "vectors.jsonl"
        self.manifest_file = self.index_path / "manifest.json"
        self.stats_file = self.index_path / "stats.json"

        # Load or create manifest
        if self.manifest_file.exists():
            self.manifest = SPLADEIndexManifest.load(str(self.manifest_file))
        else:
            self.manifest = SPLADEIndexManifest(
                index_path=str(self.index_path),
                model_name=self.model_name,
                tokenizer_name=self.tokenizer_name,
                sparsity_threshold=self.sparsity_threshold,
                quantization_scale=self.quantization_scale,
                max_terms_per_chunk=self.max_terms_per_chunk,
            )
            self.manifest.save(str(self.manifest_file))

        # Index statistics
        self._stats = self._load_stats()

        logger.info(
            "Initialized SPLADE impact index",
            extra={
                "index_path": str(self.index_path),
                "model_name": self.model_name,
                "tokenizer_name": self.tokenizer_name,
                "sparsity_threshold": self.sparsity_threshold,
                "quantization_scale": self.quantization_scale,
                "max_terms_per_chunk": self.max_terms_per_chunk,
            },
        )

    def _load_stats(self) -> dict[str, Any]:
        """Load index statistics."""
        if self.stats_file.exists():
            with open(self.stats_file) as f:
                return json.load(f)
        return {
            "total_chunks": 0,
            "total_terms": 0,
            "avg_terms_per_chunk": 0,
            "index_size_bytes": 0,
            "last_updated": None,
        }

    def _save_stats(self) -> None:
        """Save index statistics."""
        with open(self.stats_file, "w") as f:
            json.dump(self._stats, f, indent=2)

    def add_vector(
        self,
        chunk_id: str,
        vector: SPLADEVector,
    ) -> None:
        """Add a SPLADE vector to the index.

        Args:
            chunk_id: Unique identifier for the chunk
            vector: SPLADE vector to add

        """
        start_time = time.perf_counter()

        try:
            # Validate vector
            if not vector.terms:
                logger.warning("Empty SPLADE vector provided", extra={"chunk_id": chunk_id})
                return

            # Create index entry
            entry = {
                "chunk_id": chunk_id,
                "terms": vector.terms,
                "tokenizer_name": vector.tokenizer_name,
                "model_name": vector.model_name,
                "sparsity_threshold": vector.sparsity_threshold,
                "quantization_scale": vector.quantization_scale,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }

            # Append to vectors file
            with open(self.vectors_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

            # Update statistics
            self._stats["total_chunks"] += 1
            self._stats["total_terms"] += len(vector.terms)
            self._stats["avg_terms_per_chunk"] = (
                self._stats["total_terms"] / self._stats["total_chunks"]
            )
            self._stats["index_size_bytes"] = self.vectors_file.stat().st_size
            self._stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Save statistics
            self._save_stats()

            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="add_vector", status="ok").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="add_vector", status="ok").inc()
            SPLADE_INDEX_SIZE_BYTES.labels(operation="add_vector").observe(
                self._stats["index_size_bytes"]
            )

            logger.info(
                "SPLADE vector added to index",
                extra={
                    "chunk_id": chunk_id,
                    "terms_count": len(vector.terms),
                    "processing_time_seconds": processing_time,
                },
            )

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="add_vector", status="error").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="add_vector", status="error").inc()

            logger.error(
                "Failed to add SPLADE vector to index",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def get_vector(self, chunk_id: str) -> SPLADEVector | None:
        """Get a SPLADE vector from the index.

        Args:
            chunk_id: Unique identifier for the chunk

        Returns:
            SPLADE vector if found, None otherwise

        """
        start_time = time.perf_counter()

        try:
            if not self.vectors_file.exists():
                return None

            with open(self.vectors_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry["chunk_id"] == chunk_id:
                        processing_time = time.perf_counter() - start_time

                        SPLADE_INDEX_SECONDS.labels(operation="get_vector", status="ok").observe(
                            processing_time
                        )
                        SPLADE_INDEX_OPERATIONS.labels(operation="get_vector", status="ok").inc()

                        return SPLADEVector(
                            terms=entry["terms"],
                            tokenizer_name=entry["tokenizer_name"],
                            model_name=entry["model_name"],
                            sparsity_threshold=entry["sparsity_threshold"],
                            quantization_scale=entry["quantization_scale"],
                        )

            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="get_vector", status="not_found").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="get_vector", status="not_found").inc()

            return None

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="get_vector", status="error").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="get_vector", status="error").inc()

            logger.error(
                "Failed to get SPLADE vector from index",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def search_vectors(
        self,
        query_terms: dict[int, float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search for similar vectors using dot product similarity.

        Args:
            query_terms: Query terms with weights
            top_k: Number of top results to return

        Returns:
            List of (chunk_id, score) tuples sorted by score

        """
        start_time = time.perf_counter()

        try:
            if not self.vectors_file.exists():
                return []

            scores = {}

            with open(self.vectors_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    chunk_id = entry["chunk_id"]
                    vector_terms = entry["terms"]

                    # Calculate dot product similarity
                    score = 0.0
                    for term_id, query_weight in query_terms.items():
                        if term_id in vector_terms:
                            vector_weight = vector_terms[term_id]
                            score += query_weight * vector_weight

                    if score > 0:
                        scores[chunk_id] = score

            # Sort by score and return top-k
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            results = sorted_scores[:top_k]

            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="search_vectors", status="ok").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="search_vectors", status="ok").inc()

            logger.info(
                "SPLADE vector search completed",
                extra={
                    "query_terms": len(query_terms),
                    "results_count": len(results),
                    "processing_time_seconds": processing_time,
                },
            )

            return results

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="search_vectors", status="error").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="search_vectors", status="error").inc()

            logger.error(
                "Failed to search SPLADE vectors",
                extra={
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def get_all_vectors(self) -> list[tuple[str, SPLADEVector]]:
        """Get all vectors from the index.

        Returns:
            List of (chunk_id, vector) tuples

        """
        start_time = time.perf_counter()

        try:
            if not self.vectors_file.exists():
                return []

            vectors = []

            with open(self.vectors_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    chunk_id = entry["chunk_id"]

                    vector = SPLADEVector(
                        terms=entry["terms"],
                        tokenizer_name=entry["tokenizer_name"],
                        model_name=entry["model_name"],
                        sparsity_threshold=entry["sparsity_threshold"],
                        quantization_scale=entry["quantization_scale"],
                    )

                    vectors.append((chunk_id, vector))

            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="get_all_vectors", status="ok").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="get_all_vectors", status="ok").inc()

            logger.info(
                "Retrieved all SPLADE vectors",
                extra={
                    "vectors_count": len(vectors),
                    "processing_time_seconds": processing_time,
                },
            )

            return vectors

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="get_all_vectors", status="error").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="get_all_vectors", status="error").inc()

            logger.error(
                "Failed to get all SPLADE vectors",
                extra={
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def remove_vector(self, chunk_id: str) -> bool:
        """Remove a vector from the index.

        Args:
            chunk_id: Unique identifier for the chunk

        Returns:
            True if vector was removed, False if not found

        """
        start_time = time.perf_counter()

        try:
            if not self.vectors_file.exists():
                return False

            # Read all vectors except the one to remove
            remaining_vectors = []
            removed = False

            with open(self.vectors_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry["chunk_id"] != chunk_id:
                        remaining_vectors.append(entry)
                    else:
                        removed = True

            if not removed:
                processing_time = time.perf_counter() - start_time
                SPLADE_INDEX_SECONDS.labels(operation="remove_vector", status="not_found").observe(
                    processing_time
                )
                SPLADE_INDEX_OPERATIONS.labels(operation="remove_vector", status="not_found").inc()
                return False

            # Write remaining vectors back to file
            with open(self.vectors_file, "w") as f:
                for entry in remaining_vectors:
                    f.write(json.dumps(entry) + "\n")

            # Update statistics
            self._stats["total_chunks"] = len(remaining_vectors)
            self._stats["total_terms"] = sum(len(entry["terms"]) for entry in remaining_vectors)
            self._stats["avg_terms_per_chunk"] = (
                self._stats["total_terms"] / self._stats["total_chunks"]
                if self._stats["total_chunks"] > 0
                else 0
            )
            self._stats["index_size_bytes"] = self.vectors_file.stat().st_size
            self._stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Save statistics
            self._save_stats()

            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="remove_vector", status="ok").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="remove_vector", status="ok").inc()

            logger.info(
                "SPLADE vector removed from index",
                extra={
                    "chunk_id": chunk_id,
                    "processing_time_seconds": processing_time,
                },
            )

            return True

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="remove_vector", status="error").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="remove_vector", status="error").inc()

            logger.error(
                "Failed to remove SPLADE vector from index",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def clear_index(self) -> None:
        """Clear all vectors from the index."""
        start_time = time.perf_counter()

        try:
            # Remove vectors file
            if self.vectors_file.exists():
                self.vectors_file.unlink()

            # Reset statistics
            self._stats = {
                "total_chunks": 0,
                "total_terms": 0,
                "avg_terms_per_chunk": 0,
                "index_size_bytes": 0,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }

            # Save statistics
            self._save_stats()

            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="clear_index", status="ok").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="clear_index", status="ok").inc()

            logger.info("SPLADE index cleared", extra={"processing_time_seconds": processing_time})

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_INDEX_SECONDS.labels(operation="clear_index", status="error").observe(
                processing_time
            )
            SPLADE_INDEX_OPERATIONS.labels(operation="clear_index", status="error").inc()

            logger.error(
                "Failed to clear SPLADE index",
                extra={
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def get_index_stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with index statistics

        """
        return {
            **self._stats,
            "manifest": self.manifest.to_dict(),
            "index_path": str(self.index_path),
            "vectors_file_exists": self.vectors_file.exists(),
            "manifest_file_exists": self.manifest_file.exists(),
            "stats_file_exists": self.stats_file.exists(),
        }

    def validate_index(self) -> list[str]:
        """Validate index integrity.

        Returns:
            List of validation error messages

        """
        errors = []

        try:
            # Check if files exist
            if not self.vectors_file.exists():
                errors.append("Vectors file does not exist")

            if not self.manifest_file.exists():
                errors.append("Manifest file does not exist")

            if not self.stats_file.exists():
                errors.append("Stats file does not exist")

            # Validate manifest
            if self.manifest_file.exists():
                try:
                    manifest = SPLADEIndexManifest.load(str(self.manifest_file))
                    if manifest.model_name != self.model_name:
                        errors.append(
                            f"Manifest model name mismatch: {manifest.model_name} != {self.model_name}"
                        )
                    if manifest.tokenizer_name != self.tokenizer_name:
                        errors.append(
                            f"Manifest tokenizer name mismatch: {manifest.tokenizer_name} != {self.tokenizer_name}"
                        )
                except Exception as e:
                    errors.append(f"Failed to load manifest: {e}")

            # Validate vectors file
            if self.vectors_file.exists():
                try:
                    chunk_ids = set()
                    with open(self.vectors_file) as f:
                        for line_num, line in enumerate(f, 1):
                            try:
                                entry = json.loads(line.strip())
                                if "chunk_id" not in entry:
                                    errors.append(f"Line {line_num}: Missing chunk_id")
                                elif entry["chunk_id"] in chunk_ids:
                                    errors.append(
                                        f"Line {line_num}: Duplicate chunk_id: {entry['chunk_id']}"
                                    )
                                else:
                                    chunk_ids.add(entry["chunk_id"])

                                if "terms" not in entry:
                                    errors.append(f"Line {line_num}: Missing terms")
                                elif not isinstance(entry["terms"], dict):
                                    errors.append(f"Line {line_num}: Terms must be a dictionary")

                            except json.JSONDecodeError as e:
                                errors.append(f"Line {line_num}: Invalid JSON: {e}")
                except Exception as e:
                    errors.append(f"Failed to validate vectors file: {e}")

            # Check statistics consistency
            if self.stats_file.exists():
                try:
                    stats = self._load_stats()
                    if stats["total_chunks"] < 0:
                        errors.append("Invalid total_chunks in stats")
                    if stats["total_terms"] < 0:
                        errors.append("Invalid total_terms in stats")
                except Exception as e:
                    errors.append(f"Failed to validate stats: {e}")

        except Exception as e:
            errors.append(f"Index validation failed: {e}")

        return errors
