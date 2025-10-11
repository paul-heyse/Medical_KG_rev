"""BM25 index storage and management.

This module implements BM25 index storage using Lucene-style format
with multi-field configuration and manifest-based version tracking.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from Medical_KG_rev.services.retrieval.bm25_field_mapping import BM25FieldMapper
from Medical_KG_rev.services.retrieval.bm25_service import BM25Document
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
BM25_INDEX_SECONDS = Histogram(
    "bm25_index_seconds",
    "Time spent on BM25 index operations",
    ["operation", "status"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

BM25_INDEX_OPERATIONS = Counter(
    "bm25_index_operations_total", "Total number of BM25 index operations", ["operation", "status"]
)

BM25_INDEX_SIZE_BYTES = Histogram(
    "bm25_index_size_bytes",
    "Size of BM25 index in bytes",
    ["operation"],
    buckets=[1024, 10240, 102400, 1024000, 10240000, 102400000],
)


class BM25IndexManifest:
    """Manifest for BM25 index metadata."""

    def __init__(
        self,
        index_path: str,
        created_at: str | None = None,
        version: str = "1.0.0",
    ):
        """Initialize BM25 index manifest.

        Args:
            index_path: Path to the index directory
            created_at: Creation timestamp
            version: Index version

        """
        self.index_path = index_path
        self.created_at = created_at or time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        self.version = version

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "index_path": self.index_path,
            "created_at": self.created_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BM25IndexManifest":
        """Create manifest from dictionary."""
        return cls(
            index_path=data["index_path"],
            created_at=data["created_at"],
            version=data["version"],
        )

    def save(self, manifest_path: str) -> None:
        """Save manifest to file."""
        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, manifest_path: str) -> "BM25IndexManifest":
        """Load manifest from file."""
        with open(manifest_path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class BM25Index:
    """BM25 index storage using Lucene-style format.

    This class implements efficient storage and retrieval of BM25 documents
    using Lucene-style index format with multi-field configuration.
    """

    def __init__(
        self,
        index_path: str,
        field_mapper: BM25FieldMapper | None = None,
        created_at: str | None = None,
        version: str = "1.0.0",
    ):
        """Initialize BM25 index.

        Args:
            index_path: Path to the index directory
            field_mapper: Field mapper for document processing
            created_at: Creation timestamp
            version: Index version

        """
        self.index_path = Path(index_path)
        self.field_mapper = field_mapper or BM25FieldMapper()

        # Create index directory
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Index storage files
        self.documents_file = self.index_path / "documents.jsonl"
        self.manifest_file = self.index_path / "manifest.json"
        self.stats_file = self.index_path / "stats.json"

        # Load or create manifest
        if self.manifest_file.exists():
            self.manifest = BM25IndexManifest.load(str(self.manifest_file))
        else:
            self.manifest = BM25IndexManifest(
                index_path=str(self.index_path),
                created_at=created_at,
                version=version,
            )
            self.manifest.save(str(self.manifest_file))

        # Index statistics
        self._stats = self._load_stats()

        logger.info(
            "Initialized BM25 index",
            extra={
                "index_path": str(self.index_path),
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
            "total_documents": 0,
            "total_fields": 0,
            "index_size_bytes": 0,
            "last_updated": None,
        }

    def _save_stats(self) -> None:
        """Save index statistics."""
        with open(self.stats_file, "w") as f:
            json.dump(self._stats, f, indent=2)

    def add_document(self, document: BM25Document) -> None:
        """Add a BM25 document to the index.

        Args:
            document: BM25 document to add

        """
        start_time = time.perf_counter()

        try:
            # Create index entry
            entry = {
                "chunk_id": document.chunk_id,
                "title": document.title,
                "section_headers": document.section_headers,
                "paragraph": document.paragraph,
                "caption": document.caption,
                "table_text": document.table_text,
                "footnote": document.footnote,
                "refs_text": document.refs_text,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }

            # Append to documents file
            with open(self.documents_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

            # Update statistics
            self._stats["total_documents"] += 1
            self._stats["total_fields"] += 7  # 7 fields per document
            self._stats["index_size_bytes"] = self.documents_file.stat().st_size
            self._stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Save statistics
            self._save_stats()

            processing_time = time.perf_counter() - start_time

            BM25_INDEX_SECONDS.labels(operation="add_document", status="ok").observe(
                processing_time
            )
            BM25_INDEX_OPERATIONS.labels(operation="add_document", status="ok").inc()
            BM25_INDEX_SIZE_BYTES.labels(operation="add_document").observe(
                self._stats["index_size_bytes"]
            )

            logger.info(
                "BM25 document added to index",
                extra={
                    "chunk_id": document.chunk_id,
                    "processing_time_seconds": processing_time,
                },
            )

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            BM25_INDEX_SECONDS.labels(operation="add_document", status="error").observe(
                processing_time
            )
            BM25_INDEX_OPERATIONS.labels(operation="add_document", status="error").inc()

            logger.error(
                "Failed to add BM25 document to index",
                extra={
                    "chunk_id": document.chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def get_document(self, chunk_id: str) -> BM25Document | None:
        """Get a BM25 document from the index.

        Args:
            chunk_id: Document identifier

        Returns:
            BM25 document if found, None otherwise

        """
        start_time = time.perf_counter()

        try:
            if not self.documents_file.exists():
                return None

            with open(self.documents_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry["chunk_id"] == chunk_id:
                        processing_time = time.perf_counter() - start_time

                        BM25_INDEX_SECONDS.labels(operation="get_document", status="ok").observe(
                            processing_time
                        )
                        BM25_INDEX_OPERATIONS.labels(operation="get_document", status="ok").inc()

                        return BM25Document(
                            chunk_id=entry["chunk_id"],
                            title=entry["title"],
                            section_headers=entry["section_headers"],
                            paragraph=entry["paragraph"],
                            caption=entry["caption"],
                            table_text=entry["table_text"],
                            footnote=entry["footnote"],
                            refs_text=entry["refs_text"],
                        )

            processing_time = time.perf_counter() - start_time

            BM25_INDEX_SECONDS.labels(operation="get_document", status="not_found").observe(
                processing_time
            )
            BM25_INDEX_OPERATIONS.labels(operation="get_document", status="not_found").inc()

            return None

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            BM25_INDEX_SECONDS.labels(operation="get_document", status="error").observe(
                processing_time
            )
            BM25_INDEX_OPERATIONS.labels(operation="get_document", status="error").inc()

            logger.error(
                "Failed to get BM25 document from index",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def get_all_documents(self) -> list[BM25Document]:
        """Get all documents from the index.

        Returns:
            List of BM25 documents

        """
        start_time = time.perf_counter()

        try:
            if not self.documents_file.exists():
                return []

            documents = []

            with open(self.documents_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    document = BM25Document(
                        chunk_id=entry["chunk_id"],
                        title=entry["title"],
                        section_headers=entry["section_headers"],
                        paragraph=entry["paragraph"],
                        caption=entry["caption"],
                        table_text=entry["table_text"],
                        footnote=entry["footnote"],
                        refs_text=entry["refs_text"],
                    )
                    documents.append(document)

            processing_time = time.perf_counter() - start_time

            BM25_INDEX_SECONDS.labels(operation="get_all_documents", status="ok").observe(
                processing_time
            )
            BM25_INDEX_OPERATIONS.labels(operation="get_all_documents", status="ok").inc()

            logger.info(
                "Retrieved all BM25 documents",
                extra={
                    "document_count": len(documents),
                    "processing_time_seconds": processing_time,
                },
            )

            return documents

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            BM25_INDEX_SECONDS.labels(operation="get_all_documents", status="error").observe(
                processing_time
            )
            BM25_INDEX_OPERATIONS.labels(operation="get_all_documents", status="error").inc()

            logger.error(
                "Failed to get all BM25 documents",
                extra={
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def remove_document(self, chunk_id: str) -> bool:
        """Remove a document from the index.

        Args:
            chunk_id: Document identifier

        Returns:
            True if document was removed, False if not found

        """
        start_time = time.perf_counter()

        try:
            if not self.documents_file.exists():
                return False

            # Read all documents except the one to remove
            remaining_documents = []
            removed = False

            with open(self.documents_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry["chunk_id"] != chunk_id:
                        remaining_documents.append(entry)
                    else:
                        removed = True

            if not removed:
                processing_time = time.perf_counter() - start_time
                BM25_INDEX_SECONDS.labels(operation="remove_document", status="not_found").observe(
                    processing_time
                )
                BM25_INDEX_OPERATIONS.labels(operation="remove_document", status="not_found").inc()
                return False

            # Write remaining documents back to file
            with open(self.documents_file, "w") as f:
                for entry in remaining_documents:
                    f.write(json.dumps(entry) + "\n")

            # Update statistics
            self._stats["total_documents"] = len(remaining_documents)
            self._stats["total_fields"] = len(remaining_documents) * 7
            self._stats["index_size_bytes"] = self.documents_file.stat().st_size
            self._stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Save statistics
            self._save_stats()

            processing_time = time.perf_counter() - start_time

            BM25_INDEX_SECONDS.labels(operation="remove_document", status="ok").observe(
                processing_time
            )
            BM25_INDEX_OPERATIONS.labels(operation="remove_document", status="ok").inc()

            logger.info(
                "BM25 document removed from index",
                extra={
                    "chunk_id": chunk_id,
                    "processing_time_seconds": processing_time,
                },
            )

            return True

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            BM25_INDEX_SECONDS.labels(operation="remove_document", status="error").observe(
                processing_time
            )
            BM25_INDEX_OPERATIONS.labels(operation="remove_document", status="error").inc()

            logger.error(
                "Failed to remove BM25 document from index",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def clear_index(self) -> None:
        """Clear all documents from the index."""
        start_time = time.perf_counter()

        try:
            # Remove documents file
            if self.documents_file.exists():
                self.documents_file.unlink()

            # Reset statistics
            self._stats = {
                "total_documents": 0,
                "total_fields": 0,
                "index_size_bytes": 0,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }

            # Save statistics
            self._save_stats()

            processing_time = time.perf_counter() - start_time

            BM25_INDEX_SECONDS.labels(operation="clear_index", status="ok").observe(processing_time)
            BM25_INDEX_OPERATIONS.labels(operation="clear_index", status="ok").inc()

            logger.info("BM25 index cleared", extra={"processing_time_seconds": processing_time})

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            BM25_INDEX_SECONDS.labels(operation="clear_index", status="error").observe(
                processing_time
            )
            BM25_INDEX_OPERATIONS.labels(operation="clear_index", status="error").inc()

            logger.error(
                "Failed to clear BM25 index",
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
            "total_documents": self._stats["total_documents"],
            "total_fields": self._stats["total_fields"],
            "index_size_bytes": self._stats["index_size_bytes"],
            "last_updated": self._stats["last_updated"],
            "manifest": self.manifest.to_dict(),
            "field_boosts": self.field_mapper.get_field_boosts(),
        }

    def validate_index(self) -> list[str]:
        """Validate index integrity.

        Returns:
            List of validation error messages

        """
        errors = []

        try:
            # Check if files exist
            if not self.documents_file.exists():
                errors.append("Documents file does not exist")

            if not self.manifest_file.exists():
                errors.append("Manifest file does not exist")

            if not self.stats_file.exists():
                errors.append("Stats file does not exist")

            # Validate manifest
            if self.manifest_file.exists():
                try:
                    manifest = BM25IndexManifest.load(str(self.manifest_file))
                    if manifest.index_path != str(self.index_path):
                        errors.append(
                            f"Manifest index path mismatch: {manifest.index_path} != {self.index_path}"
                        )
                except Exception as e:
                    errors.append(f"Failed to load manifest: {e}")

            # Validate documents file
            if self.documents_file.exists():
                try:
                    chunk_ids = set()
                    line_num = 0

                    with open(self.documents_file) as f:
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

                                # Check required fields
                                required_fields = [
                                    "title",
                                    "section_headers",
                                    "paragraph",
                                    "caption",
                                    "table_text",
                                    "footnote",
                                    "refs_text",
                                ]
                                for field in required_fields:
                                    if field not in entry:
                                        errors.append(f"Line {line_num}: Missing field: {field}")
                                    elif not isinstance(entry[field], str):
                                        errors.append(
                                            f"Line {line_num}: Field {field} must be a string"
                                        )

                            except json.JSONDecodeError as e:
                                errors.append(f"Line {line_num}: Invalid JSON: {e}")
                except Exception as e:
                    errors.append(f"Failed to validate documents file: {e}")

            # Check statistics consistency
            if self.stats_file.exists():
                try:
                    with open(self.stats_file) as f:
                        stats = json.load(f)

                    if "total_documents" not in stats:
                        errors.append("Stats missing total_documents")
                    elif not isinstance(stats["total_documents"], int):
                        errors.append("Stats total_documents must be an integer")

                    if "index_size_bytes" not in stats:
                        errors.append("Stats missing index_size_bytes")
                    elif not isinstance(stats["index_size_bytes"], int):
                        errors.append("Stats index_size_bytes must be an integer")

                except Exception as e:
                    errors.append(f"Failed to validate stats: {e}")

        except Exception as e:
            errors.append(f"Index validation failed: {e}")

        return errors

    def rebuild_index(self) -> None:
        """Rebuild the index from scratch.

        This method clears the index and rebuilds it from the stored documents.
        """
        start_time = time.perf_counter()

        try:
            # Get all documents
            documents = self.get_all_documents()

            # Clear index
            self.clear_index()

            # Rebuild index
            for document in documents:
                self.add_document(document)

            processing_time = time.perf_counter() - start_time

            logger.info(
                "BM25 index rebuilt",
                extra={
                    "document_count": len(documents),
                    "processing_time_seconds": processing_time,
                },
            )

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            logger.error(
                "Failed to rebuild BM25 index",
                extra={
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def health_check(self) -> dict[str, Any]:
        """Check BM25 index health.

        Returns:
            Health status information

        """
        try:
            stats = self.get_index_stats()
            validation_errors = self.validate_index()

            return {
                "status": "healthy" if not validation_errors else "unhealthy",
                "total_documents": stats["total_documents"],
                "index_size_bytes": stats["index_size_bytes"],
                "last_updated": stats["last_updated"],
                "validation_errors": validation_errors,
                "manifest": stats["manifest"],
                "field_boosts": stats["field_boosts"],
            }

        except Exception as e:
            logger.error("BM25 index health check failed", extra={"error": str(e)})
            return {
                "status": "unhealthy",
                "error": str(e),
            }
