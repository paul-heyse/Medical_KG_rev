"""Storage layout and manifest management.

This module implements storage layout management for the hybrid retrieval system,
including manifest creation, validation, and storage organization.
"""

from pathlib import Path
from typing import Any
import json
import logging
import time

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class StorageManifest(BaseModel):
    """Storage manifest for tracking storage layout and metadata."""

    version: str = Field(default="1.0", description="Manifest version")
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        description="Creation timestamp",
    )
    updated_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        description="Last update timestamp",
    )

    # Storage paths
    chunk_store_path: str = Field(..., description="Path to chunk store database")
    bm25_index_path: str = Field(..., description="Path to BM25 index")
    splade_index_path: str = Field(..., description="Path to SPLADE index")
    qwen3_index_path: str = Field(..., description="Path to Qwen3 index")

    # Raw artifacts
    raw_artifacts_path: str = Field(..., description="Path to raw artifacts storage")
    doctags_path: str = Field(..., description="Path to DocTags storage")

    # Metadata
    total_documents: int = Field(default=0, description="Total number of documents")
    total_chunks: int = Field(default=0, description="Total number of chunks")
    total_tokens: int = Field(default=0, description="Total number of tokens")

    # Index metadata
    bm25_documents: int = Field(default=0, description="Number of documents in BM25 index")
    splade_documents: int = Field(default=0, description="Number of documents in SPLADE index")
    qwen3_documents: int = Field(default=0, description="Number of documents in Qwen3 index")

    # Storage sizes
    chunk_store_size_bytes: int = Field(default=0, description="Chunk store size in bytes")
    bm25_index_size_bytes: int = Field(default=0, description="BM25 index size in bytes")
    splade_index_size_bytes: int = Field(default=0, description="SPLADE index size in bytes")
    qwen3_index_size_bytes: int = Field(default=0, description="Qwen3 index size in bytes")
    raw_artifacts_size_bytes: int = Field(default=0, description="Raw artifacts size in bytes")
    doctags_size_bytes: int = Field(default=0, description="DocTags size in bytes")

    # Configuration
    config: dict[str, Any] = Field(default_factory=dict, description="Storage configuration")

    # Validation
    is_valid: bool = Field(default=True, description="Whether the storage layout is valid")
    validation_errors: list[str] = Field(default_factory=list, description="Validation errors")


class StorageLayoutConfig(BaseModel):
    """Configuration for storage layout management."""

    base_path: str = Field(default="storage", description="Base storage path")
    manifest_filename: str = Field(default="storage_manifest.json", description="Manifest filename")
    enable_validation: bool = Field(default=True, description="Enable layout validation")
    enable_compression: bool = Field(default=True, description="Enable storage compression")
    max_file_size_mb: int = Field(default=1000, description="Maximum file size in MB")
    enable_backup: bool = Field(default=True, description="Enable manifest backup")
    backup_retention_days: int = Field(default=30, description="Backup retention in days")


class StorageLayout:
    """Storage layout manager for hybrid retrieval system.

    This class manages the storage layout, including manifest creation,
    validation, and storage organization.
    """

    def __init__(self, config: StorageLayoutConfig | None = None):
        """Initialize storage layout manager.

        Args:
        ----
            config: Configuration for storage layout

        """
        self.config = config or StorageLayoutConfig()

        # Create base path
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Define storage paths
        self.chunk_store_path = self.base_path / "chunk_store"
        self.bm25_index_path = self.base_path / "bm25_index"
        self.splade_index_path = self.base_path / "splade_index"
        self.qwen3_index_path = self.base_path / "qwen3_index"
        self.raw_artifacts_path = self.base_path / "raw_artifacts"
        self.doctags_path = self.base_path / "doctags"

        # Create storage directories
        self._create_storage_directories()

        # Load or create manifest
        self.manifest = self._load_or_create_manifest()

        logger.info(
            "Initialized storage layout",
            extra={
                "base_path": str(self.base_path),
                "manifest_path": str(self.manifest_path),
                "enable_validation": self.config.enable_validation,
                "enable_compression": self.config.enable_compression,
            },
        )

    @property
    def manifest_path(self) -> Path:
        """Get manifest file path."""
        return self.base_path / self.config.manifest_filename

    def _create_storage_directories(self) -> None:
        """Create storage directories."""
        directories = [
            self.chunk_store_path,
            self.bm25_index_path,
            self.splade_index_path,
            self.qwen3_index_path,
            self.raw_artifacts_path,
            self.doctags_path,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created storage directory: {directory}")

    def _load_or_create_manifest(self) -> StorageManifest:
        """Load existing manifest or create a new one."""
        if self.manifest_path.exists():
            try:
                with self.manifest_path.open("r", encoding="utf-8") as f:
                    manifest_data = json.load(f)

                manifest = StorageManifest(**manifest_data)
                logger.info("Loaded existing storage manifest")
                return manifest

            except Exception as e:
                logger.warning(f"Failed to load existing manifest: {e}. Creating new one.")

        # Create new manifest
        manifest = StorageManifest(
            chunk_store_path=str(self.chunk_store_path),
            bm25_index_path=str(self.bm25_index_path),
            splade_index_path=str(self.splade_index_path),
            qwen3_index_path=str(self.qwen3_index_path),
            raw_artifacts_path=str(self.raw_artifacts_path),
            doctags_path=str(self.doctags_path),
            config=self.config.dict(),
        )

        self._save_manifest(manifest)
        logger.info("Created new storage manifest")
        return manifest

    def _save_manifest(self, manifest: StorageManifest) -> None:
        """Save manifest to disk."""
        try:
            # Update timestamp
            manifest.updated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Save manifest
            with self.manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest.dict(), f, indent=2, ensure_ascii=False)

            # Create backup if enabled
            if self.config.enable_backup:
                self._create_manifest_backup(manifest)

            logger.debug("Saved storage manifest")

        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            raise

    def _create_manifest_backup(self, manifest: StorageManifest) -> None:
        """Create manifest backup."""
        try:
            backup_dir = self.base_path / "backups"
            backup_dir.mkdir(exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            backup_path = backup_dir / f"storage_manifest_{timestamp}.json"

            with backup_path.open("w", encoding="utf-8") as f:
                json.dump(manifest.dict(), f, indent=2, ensure_ascii=False)

            # Clean up old backups
            self._cleanup_old_backups(backup_dir)

            logger.debug(f"Created manifest backup: {backup_path}")

        except Exception as e:
            logger.warning(f"Failed to create manifest backup: {e}")

    def _cleanup_old_backups(self, backup_dir: Path) -> None:
        """Clean up old backup files."""
        try:
            cutoff_time = time.time() - (self.config.backup_retention_days * 24 * 60 * 60)

            for backup_file in backup_dir.glob("storage_manifest_*.json"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    logger.debug(f"Removed old backup: {backup_file}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    def validate_layout(self) -> bool:
        """Validate storage layout.

        Returns
        -------
            True if layout is valid, False otherwise

        """
        if not self.config.enable_validation:
            return True

        errors = []

        # Check storage directories exist
        required_dirs = [
            self.chunk_store_path,
            self.bm25_index_path,
            self.splade_index_path,
            self.qwen3_index_path,
            self.raw_artifacts_path,
            self.doctags_path,
        ]

        for directory in required_dirs:
            if not directory.exists():
                errors.append(f"Storage directory missing: {directory}")
            elif not directory.is_dir():
                errors.append(f"Storage path is not a directory: {directory}")

        # Check file sizes
        for directory in required_dirs:
            if directory.exists():
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        if file_size_mb > self.config.max_file_size_mb:
                            errors.append(f"File too large: {file_path} ({file_size_mb:.1f} MB)")

        # Check manifest consistency
        if self.manifest.chunk_store_path != str(self.chunk_store_path):
            errors.append("Manifest chunk store path mismatch")
        if self.manifest.bm25_index_path != str(self.bm25_index_path):
            errors.append("Manifest BM25 index path mismatch")
        if self.manifest.splade_index_path != str(self.splade_index_path):
            errors.append("Manifest SPLADE index path mismatch")
        if self.manifest.qwen3_index_path != str(self.qwen3_index_path):
            errors.append("Manifest Qwen3 index path mismatch")
        if self.manifest.raw_artifacts_path != str(self.raw_artifacts_path):
            errors.append("Manifest raw artifacts path mismatch")
        if self.manifest.doctags_path != str(self.doctags_path):
            errors.append("Manifest DocTags path mismatch")

        # Update manifest validation status
        self.manifest.is_valid = len(errors) == 0
        self.manifest.validation_errors = errors

        if errors:
            logger.warning(f"Storage layout validation failed: {errors}")
        else:
            logger.info("Storage layout validation passed")

        return len(errors) == 0

    def update_manifest(self, **kwargs: Any) -> None:
        """Update manifest with new data.

        Args:
        ----
            **kwargs: Fields to update in manifest

        """
        try:
            # Update manifest fields
            for key, value in kwargs.items():
                if hasattr(self.manifest, key):
                    setattr(self.manifest, key, value)
                else:
                    logger.warning(f"Unknown manifest field: {key}")

            # Recalculate storage sizes
            self._update_storage_sizes()

            # Save updated manifest
            self._save_manifest(self.manifest)

            logger.info("Updated storage manifest", extra=kwargs)

        except Exception as e:
            logger.error(f"Failed to update manifest: {e}")
            raise

    def _update_storage_sizes(self) -> None:
        """Update storage sizes in manifest."""
        try:
            # Update chunk store size
            chunk_store_size = self._calculate_directory_size(self.chunk_store_path)
            self.manifest.chunk_store_size_bytes = chunk_store_size

            # Update index sizes
            self.manifest.bm25_index_size_bytes = self._calculate_directory_size(
                self.bm25_index_path
            )
            self.manifest.splade_index_size_bytes = self._calculate_directory_size(
                self.splade_index_path
            )
            self.manifest.qwen3_index_size_bytes = self._calculate_directory_size(
                self.qwen3_index_path
            )

            # Update raw artifacts size
            self.manifest.raw_artifacts_size_bytes = self._calculate_directory_size(
                self.raw_artifacts_path
            )
            self.manifest.doctags_size_bytes = self._calculate_directory_size(self.doctags_path)

        except Exception as e:
            logger.warning(f"Failed to update storage sizes: {e}")

    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory.

        Args:
        ----
            directory: Directory path

        Returns:
        -------
            Total size in bytes

        """
        if not directory.exists():
            return 0

        total_size = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns
        -------
            Dictionary with storage statistics

        """
        try:
            # Update storage sizes
            self._update_storage_sizes()

            total_size = (
                self.manifest.chunk_store_size_bytes
                + self.manifest.bm25_index_size_bytes
                + self.manifest.splade_index_size_bytes
                + self.manifest.qwen3_index_size_bytes
                + self.manifest.raw_artifacts_size_bytes
                + self.manifest.doctags_size_bytes
            )

            stats = {
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_size_gb": total_size / (1024 * 1024 * 1024),
                "chunk_store_size_bytes": self.manifest.chunk_store_size_bytes,
                "bm25_index_size_bytes": self.manifest.bm25_index_size_bytes,
                "splade_index_size_bytes": self.manifest.splade_index_size_bytes,
                "qwen3_index_size_bytes": self.manifest.qwen3_index_size_bytes,
                "raw_artifacts_size_bytes": self.manifest.raw_artifacts_size_bytes,
                "doctags_size_bytes": self.manifest.doctags_size_bytes,
                "total_documents": self.manifest.total_documents,
                "total_chunks": self.manifest.total_chunks,
                "total_tokens": self.manifest.total_tokens,
                "bm25_documents": self.manifest.bm25_documents,
                "splade_documents": self.manifest.splade_documents,
                "qwen3_documents": self.manifest.qwen3_documents,
                "is_valid": self.manifest.is_valid,
                "validation_errors": self.manifest.validation_errors,
                "created_at": self.manifest.created_at,
                "updated_at": self.manifest.updated_at,
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}

    def get_path(self, storage_type: str) -> Path:
        """Get path for storage type.

        Args:
        ----
            storage_type: Type of storage (chunk_store, bm25_index, etc.)

        Returns:
        -------
            Path for the storage type

        """
        path_mapping = {
            "chunk_store": self.chunk_store_path,
            "bm25_index": self.bm25_index_path,
            "splade_index": self.splade_index_path,
            "qwen3_index": self.qwen3_index_path,
            "raw_artifacts": self.raw_artifacts_path,
            "doctags": self.doctags_path,
        }

        if storage_type not in path_mapping:
            raise ValueError(f"Unknown storage type: {storage_type}")

        return path_mapping[storage_type]

    def health_check(self) -> dict[str, Any]:
        """Check storage layout health.

        Returns
        -------
            Health status information

        """
        try:
            stats = self.get_storage_stats()
            is_valid = self.validate_layout()

            return {
                "status": "healthy" if is_valid else "unhealthy",
                "is_valid": is_valid,
                "validation_errors": self.manifest.validation_errors,
                "total_size_bytes": stats.get("total_size_bytes", 0),
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "created_at": stats.get("created_at"),
                "updated_at": stats.get("updated_at"),
            }

        except Exception as e:
            logger.error("Storage layout health check failed", extra={"error": str(e)})
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def cleanup(self) -> None:
        """Clean up storage layout."""
        try:
            # Clean up old backups
            if self.config.enable_backup:
                backup_dir = self.base_path / "backups"
                if backup_dir.exists():
                    self._cleanup_old_backups(backup_dir)

            logger.info("Storage layout cleanup completed")

        except Exception as e:
            logger.error(f"Storage layout cleanup failed: {e}")

    def __enter__(self) -> "StorageLayout":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.cleanup()
