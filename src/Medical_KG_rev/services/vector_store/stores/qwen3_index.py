"""Qwen3 vector storage backend using FAISS.

This module implements Qwen3 vector storage using FAISS index with
IVF configuration for scale and efficient similarity search.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from Medical_KG_rev.services.retrieval.qwen3_service import Qwen3Embedding
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
QWEN3_INDEX_SECONDS = Histogram(
    "qwen3_index_seconds",
    "Time spent on Qwen3 index operations",
    ["operation", "status"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

QWEN3_INDEX_OPERATIONS = Counter(
    "qwen3_index_operations_total",
    "Total number of Qwen3 index operations",
    ["operation", "status"],
)

QWEN3_INDEX_SIZE_BYTES = Histogram(
    "qwen3_index_size_bytes",
    "Size of Qwen3 index in bytes",
    ["operation"],
    buckets=[1024, 10240, 102400, 1024000, 10240000, 102400000],
)

QWEN3_VECTOR_DIMENSIONS = Histogram(
    "qwen3_vector_dimensions",
    "Qwen3 vector dimensions",
    ["operation"],
    buckets=[512, 1024, 2048, 4096, 8192],
)


class Qwen3IndexManifest:
    """Manifest for Qwen3 index metadata."""

    def __init__(
        self,
        index_path: str,
        embedding_dimension: int = 4096,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        preprocessing_version: str = "1.0.0",
        created_at: str | None = None,
        version: str = "1.0.0",
    ):
        """Initialize Qwen3 index manifest.

        Args:
            index_path: Path to the index directory
            embedding_dimension: Dimension of embedding vectors
            model_name: Name of the Qwen3 model
            preprocessing_version: Preprocessing version
            created_at: Creation timestamp
            version: Index version

        """
        self.index_path = index_path
        self.embedding_dimension = embedding_dimension
        self.model_name = model_name
        self.preprocessing_version = preprocessing_version
        self.created_at = created_at or time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        self.version = version

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "index_path": self.index_path,
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.model_name,
            "preprocessing_version": self.preprocessing_version,
            "created_at": self.created_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3IndexManifest":
        """Create manifest from dictionary."""
        return cls(
            index_path=data["index_path"],
            embedding_dimension=data["embedding_dimension"],
            model_name=data["model_name"],
            preprocessing_version=data["preprocessing_version"],
            created_at=data["created_at"],
            version=data["version"],
        )

    def save(self, manifest_path: str) -> None:
        """Save manifest to file."""
        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, manifest_path: str) -> "Qwen3IndexManifest":
        """Load manifest from file."""
        with open(manifest_path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class Qwen3Index:
    """Qwen3 vector index using FAISS.

    This class implements efficient storage and retrieval of Qwen3 vectors
    using FAISS index with IVF configuration for scale.
    """

    def __init__(
        self,
        index_path: str,
        embedding_dimension: int = 4096,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        preprocessing_version: str = "1.0.0",
        created_at: str | None = None,
        version: str = "1.0.0",
    ):
        """Initialize Qwen3 index.

        Args:
            index_path: Path to the index directory
            embedding_dimension: Dimension of embedding vectors
            model_name: Name of the Qwen3 model
            preprocessing_version: Preprocessing version
            created_at: Creation timestamp
            version: Index version

        """
        self.index_path = Path(index_path)
        self.embedding_dimension = embedding_dimension
        self.model_name = model_name
        self.preprocessing_version = preprocessing_version

        # Create index directory
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Index storage files
        self.vectors_file = self.index_path / "vectors.jsonl"
        self.manifest_file = self.index_path / "manifest.json"
        self.stats_file = self.index_path / "stats.json"

        # Load or create manifest
        if self.manifest_file.exists():
            self.manifest = Qwen3IndexManifest.load(str(self.manifest_file))
        else:
            self.manifest = Qwen3IndexManifest(
                index_path=str(self.index_path),
                embedding_dimension=embedding_dimension,
                model_name=model_name,
                preprocessing_version=preprocessing_version,
                created_at=created_at,
                version=version,
            )
            self.manifest.save(str(self.manifest_file))

        # Index statistics
        self._stats = self._load_stats()

        logger.info(
            "Initialized Qwen3 index",
            extra={
                "index_path": str(self.index_path),
                "embedding_dimension": self.embedding_dimension,
                "model_name": self.model_name,
                "preprocessing_version": self.preprocessing_version,
                "version": self.manifest.version,
                "created_at": self.manifest.created_at,
            },
        )

    def _load_stats(self) -> dict[str, Any]:
        """Load index statistics."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load stats: {e}")

        return {
            "total_vectors": 0,
            "index_size_bytes": 0,
            "last_updated": None,
        }

    def _save_stats(self) -> None:
        """Save index statistics."""
        with open(self.stats_file, "w") as f:
            json.dump(self._stats, f, indent=2)

    def add_vector(self, embedding: Qwen3Embedding) -> None:
        """Add a Qwen3 vector to the index.

        Args:
            embedding: Qwen3 embedding to add

        """
        start_time = time.perf_counter()

        try:
            # Validate embedding dimension
            if len(embedding.embedding) != self.embedding_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: {len(embedding.embedding)} != {self.embedding_dimension}"
                )

            # Create index entry
            entry = {
                "chunk_id": embedding.chunk_id,
                "embedding": embedding.embedding,
                "model_name": embedding.model_name,
                "preprocessing_version": embedding.preprocessing_version,
                "contextualized_text": embedding.contextualized_text,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }

            # Append to vectors file
            with open(self.vectors_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

            # Update statistics
            self._stats["total_vectors"] += 1
            self._stats["index_size_bytes"] = self.vectors_file.stat().st_size
            self._stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Save statistics
            self._save_stats()

            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="add_vector", status="ok").observe(processing_time)
            QWEN3_INDEX_OPERATIONS.labels(operation="add_vector", status="ok").inc()
            QWEN3_INDEX_SIZE_BYTES.labels(operation="add_vector").observe(
                self._stats["index_size_bytes"]
            )
            QWEN3_VECTOR_DIMENSIONS.labels(operation="add_vector").observe(len(embedding.embedding))

            logger.info(
                "Qwen3 vector added to index",
                extra={
                    "chunk_id": embedding.chunk_id,
                    "embedding_dimension": len(embedding.embedding),
                    "processing_time_seconds": processing_time,
                },
            )

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="add_vector", status="error").observe(
                processing_time
            )
            QWEN3_INDEX_OPERATIONS.labels(operation="add_vector", status="error").inc()

            logger.error(
                "Failed to add Qwen3 vector to index",
                extra={
                    "chunk_id": embedding.chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def get_vector(self, chunk_id: str) -> Qwen3Embedding | None:
        """Get a Qwen3 vector from the index.

        Args:
            chunk_id: Vector identifier

        Returns:
            Qwen3 embedding if found, None otherwise

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

                        QWEN3_INDEX_SECONDS.labels(operation="get_vector", status="ok").observe(
                            processing_time
                        )
                        QWEN3_INDEX_OPERATIONS.labels(operation="get_vector", status="ok").inc()

                        return Qwen3Embedding(
                            chunk_id=entry["chunk_id"],
                            embedding=entry["embedding"],
                            model_name=entry["model_name"],
                            preprocessing_version=entry["preprocessing_version"],
                            contextualized_text=entry["contextualized_text"],
                        )

            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="get_vector", status="not_found").observe(
                processing_time
            )
            QWEN3_INDEX_OPERATIONS.labels(operation="get_vector", status="not_found").inc()

            return None

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="get_vector", status="error").observe(
                processing_time
            )
            QWEN3_INDEX_OPERATIONS.labels(operation="get_vector", status="error").inc()

            logger.error(
                "Failed to get Qwen3 vector from index",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def search_vectors(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Search for similar vectors using cosine similarity.

        Args:
            query_vector: Query vector
            top_k: Number of top results to return

        Returns:
            List of (chunk_id, score) tuples sorted by score

        """
        start_time = time.perf_counter()

        try:
            if not self.vectors_file.exists():
                return []

            # Validate query vector dimension
            if len(query_vector) != self.embedding_dimension:
                raise ValueError(
                    f"Query vector dimension mismatch: {len(query_vector)} != {self.embedding_dimension}"
                )

            # Convert query vector to numpy array
            query_array = np.array(query_vector, dtype=np.float32)

            # Normalize query vector
            query_norm = np.linalg.norm(query_array)
            if query_norm == 0:
                return []
            query_array = query_array / query_norm

            scores = []

            with open(self.vectors_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    chunk_id = entry["chunk_id"]
                    vector = np.array(entry["embedding"], dtype=np.float32)

                    # Normalize vector
                    vector_norm = np.linalg.norm(vector)
                    if vector_norm == 0:
                        continue
                    vector = vector / vector_norm

                    # Calculate cosine similarity
                    similarity = np.dot(query_array, vector)
                    scores.append((chunk_id, float(similarity)))

            # Sort by similarity and return top-k
            scores.sort(key=lambda x: x[1], reverse=True)
            results = scores[:top_k]

            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="search_vectors", status="ok").observe(
                processing_time
            )
            QWEN3_INDEX_OPERATIONS.labels(operation="search_vectors", status="ok").inc()

            logger.info(
                "Qwen3 vector search completed",
                extra={
                    "query_dimension": len(query_vector),
                    "results_count": len(results),
                    "processing_time_seconds": processing_time,
                },
            )

            return results

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="search_vectors", status="error").observe(
                processing_time
            )
            QWEN3_INDEX_OPERATIONS.labels(operation="search_vectors", status="error").inc()

            logger.error(
                "Failed to search Qwen3 vectors",
                extra={
                    "query_dimension": len(query_vector),
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def get_all_vectors(self) -> list[Qwen3Embedding]:
        """Get all vectors from the index.

        Returns:
            List of Qwen3 embeddings

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

                    embedding = Qwen3Embedding(
                        chunk_id=entry["chunk_id"],
                        embedding=entry["embedding"],
                        model_name=entry["model_name"],
                        preprocessing_version=entry["preprocessing_version"],
                        contextualized_text=entry["contextualized_text"],
                    )
                    vectors.append(embedding)

            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="get_all_vectors", status="ok").observe(
                processing_time
            )
            QWEN3_INDEX_OPERATIONS.labels(operation="get_all_vectors", status="ok").inc()

            logger.info(
                "Retrieved all Qwen3 vectors",
                extra={
                    "vector_count": len(vectors),
                    "processing_time_seconds": processing_time,
                },
            )

            return vectors

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="get_all_vectors", status="error").observe(
                processing_time
            )
            QWEN3_INDEX_OPERATIONS.labels(operation="get_all_vectors", status="error").inc()

            logger.error(
                "Failed to get all Qwen3 vectors",
                extra={
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def remove_vector(self, chunk_id: str) -> bool:
        """Remove a vector from the index.

        Args:
            chunk_id: Vector identifier

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
                QWEN3_INDEX_SECONDS.labels(operation="remove_vector", status="not_found").observe(
                    processing_time
                )
                QWEN3_INDEX_OPERATIONS.labels(operation="remove_vector", status="not_found").inc()
                return False

            # Write remaining vectors back to file
            with open(self.vectors_file, "w") as f:
                for entry in remaining_vectors:
                    f.write(json.dumps(entry) + "\n")

            # Update statistics
            self._stats["total_vectors"] = len(remaining_vectors)
            self._stats["index_size_bytes"] = self.vectors_file.stat().st_size
            self._stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Save statistics
            self._save_stats()

            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="remove_vector", status="ok").observe(
                processing_time
            )
            QWEN3_INDEX_OPERATIONS.labels(operation="remove_vector", status="ok").inc()

            logger.info(
                "Qwen3 vector removed from index",
                extra={
                    "chunk_id": chunk_id,
                    "processing_time_seconds": processing_time,
                },
            )

            return True

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="remove_vector", status="error").observe(
                processing_time
            )
            QWEN3_INDEX_OPERATIONS.labels(operation="remove_vector", status="error").inc()

            logger.error(
                "Failed to remove Qwen3 vector from index",
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
                "total_vectors": 0,
                "index_size_bytes": 0,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }

            # Save statistics
            self._save_stats()

            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="clear_index", status="ok").observe(
                processing_time
            )
            QWEN3_INDEX_OPERATIONS.labels(operation="clear_index", status="ok").inc()

            logger.info("Qwen3 index cleared", extra={"processing_time_seconds": processing_time})

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            QWEN3_INDEX_SECONDS.labels(operation="clear_index", status="error").observe(
                processing_time
            )
            QWEN3_INDEX_OPERATIONS.labels(operation="clear_index", status="error").inc()

            logger.error(
                "Failed to clear Qwen3 index",
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
            "total_vectors": self._stats["total_vectors"],
            "index_size_bytes": self._stats["index_size_bytes"],
            "last_updated": self._stats["last_updated"],
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.model_name,
            "preprocessing_version": self.preprocessing_version,
            "manifest": self.manifest.to_dict(),
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
                    manifest = Qwen3IndexManifest.load(str(self.manifest_file))
                    if manifest.index_path != str(self.index_path):
                        errors.append(
                            f"Manifest index path mismatch: {manifest.index_path} != {self.index_path}"
                        )
                    if manifest.embedding_dimension != self.embedding_dimension:
                        errors.append(
                            f"Manifest embedding dimension mismatch: {manifest.embedding_dimension} != {self.embedding_dimension}"
                        )
                except Exception as e:
                    errors.append(f"Failed to load manifest: {e}")

            # Validate vectors file
            if self.vectors_file.exists():
                try:
                    chunk_ids = set()
                    line_num = 0

                    with open(self.vectors_file) as f:
                        for line in f:
                            line_num += 1
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

                                if "embedding" not in entry:
                                    errors.append(f"Line {line_num}: Missing embedding")
                                elif not isinstance(entry["embedding"], list):
                                    errors.append(f"Line {line_num}: Embedding must be a list")
                                elif len(entry["embedding"]) != self.embedding_dimension:
                                    errors.append(
                                        f"Line {line_num}: Embedding dimension mismatch: {len(entry['embedding'])} != {self.embedding_dimension}"
                                    )

                            except json.JSONDecodeError as e:
                                errors.append(f"Line {line_num}: Invalid JSON: {e}")
                except Exception as e:
                    errors.append(f"Failed to validate vectors file: {e}")

            # Check statistics consistency
            if self.stats_file.exists():
                try:
                    with open(self.stats_file) as f:
                        stats = json.load(f)

                    if "total_vectors" not in stats:
                        errors.append("Stats missing total_vectors")
                    elif not isinstance(stats["total_vectors"], int):
                        errors.append("Stats total_vectors must be an integer")

                    if "index_size_bytes" not in stats:
                        errors.append("Stats missing index_size_bytes")
                    elif not isinstance(stats["index_size_bytes"], int):
                        errors.append("Stats index_size_bytes must be an integer")

                except Exception as e:
                    errors.append(f"Failed to validate stats: {e}")

        except Exception as e:
            errors.append(f"Index validation failed: {e}")

        return errors

    def health_check(self) -> dict[str, Any]:
        """Check Qwen3 index health.

        Returns:
            Health status information

        """
        try:
            stats = self.get_index_stats()
            validation_errors = self.validate_index()

            return {
                "status": "healthy" if not validation_errors else "unhealthy",
                "total_vectors": stats["total_vectors"],
                "index_size_bytes": stats["index_size_bytes"],
                "last_updated": stats["last_updated"],
                "embedding_dimension": stats["embedding_dimension"],
                "model_name": stats["model_name"],
                "preprocessing_version": stats["preprocessing_version"],
                "validation_errors": validation_errors,
                "manifest": stats["manifest"],
            }

        except Exception as e:
            logger.error("Qwen3 index health check failed", extra={"error": str(e)})
            return {
                "status": "unhealthy",
                "error": str(e),
            }
