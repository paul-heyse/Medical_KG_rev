"""Manifest management system for hybrid retrieval storage.

This module implements a comprehensive manifest management system for tracking
and managing storage manifests across the hybrid retrieval system.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from Medical_KG_rev.storage.layout import StorageLayout, StorageManifest

logger = logging.getLogger(__name__)


class ManifestManagerConfig(BaseModel):
    """Configuration for manifest manager."""

    storage_layout_path: str = Field(default="storage", description="Path to storage layout")
    enable_auto_sync: bool = Field(
        default=True, description="Enable automatic manifest synchronization"
    )
    sync_interval_seconds: int = Field(default=300, description="Sync interval in seconds")
    enable_validation: bool = Field(default=True, description="Enable manifest validation")
    enable_backup: bool = Field(default=True, description="Enable manifest backup")
    max_manifest_age_hours: int = Field(default=24, description="Maximum manifest age in hours")
    enable_compression: bool = Field(default=True, description="Enable manifest compression")


class ManifestManager:
    """Manifest manager for hybrid retrieval storage.

    This class provides comprehensive manifest management capabilities,
    including synchronization, validation, backup, and recovery.
    """

    def __init__(self, config: ManifestManagerConfig | None = None):
        """Initialize manifest manager.

        Args:
            config: Configuration for manifest manager

        """
        self.config = config or ManifestManagerConfig()

        # Initialize storage layout
        self.storage_layout = StorageLayout()

        # Load current manifest
        self.current_manifest = self.storage_layout.manifest

        # Initialize sync state
        self.last_sync_time = time.time()
        self.sync_in_progress = False

        logger.info(
            "Initialized manifest manager",
            extra={
                "storage_layout_path": self.config.storage_layout_path,
                "enable_auto_sync": self.config.enable_auto_sync,
                "sync_interval_seconds": self.config.sync_interval_seconds,
                "enable_validation": self.config.enable_validation,
                "enable_backup": self.config.enable_backup,
            },
        )

    def sync_manifest(self, force: bool = False) -> bool:
        """Synchronize manifest with storage layout.

        Args:
            force: Force sync even if recently synced

        Returns:
            True if sync was successful, False otherwise

        """
        if self.sync_in_progress:
            logger.warning("Manifest sync already in progress")
            return False

        if not force and not self._should_sync():
            logger.debug("Manifest sync skipped - too recent")
            return True

        self.sync_in_progress = True

        try:
            logger.info("Starting manifest synchronization")

            # Update storage sizes
            self.storage_layout._update_storage_sizes()

            # Update manifest with current storage state
            self.current_manifest = self.storage_layout.manifest

            # Validate manifest if enabled
            if self.config.enable_validation:
                is_valid = self.storage_layout.validate_layout()
                if not is_valid:
                    logger.warning("Manifest validation failed during sync")

            # Save updated manifest
            self.storage_layout._save_manifest(self.current_manifest)

            # Update sync time
            self.last_sync_time = time.time()

            logger.info("Manifest synchronization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Manifest synchronization failed: {e}")
            return False

        finally:
            self.sync_in_progress = False

    def _should_sync(self) -> bool:
        """Check if manifest should be synchronized.

        Returns:
            True if sync is needed, False otherwise

        """
        if not self.config.enable_auto_sync:
            return False

        time_since_last_sync = time.time() - self.last_sync_time
        return time_since_last_sync >= self.config.sync_interval_seconds

    def validate_manifest(self) -> dict[str, Any]:
        """Validate current manifest.

        Returns:
            Validation results

        """
        try:
            logger.info("Starting manifest validation")

            # Validate storage layout
            is_valid = self.storage_layout.validate_layout()

            # Check manifest age
            manifest_age_hours = self._get_manifest_age_hours()
            is_fresh = manifest_age_hours <= self.config.max_manifest_age_hours

            # Check manifest consistency
            consistency_errors = self._check_manifest_consistency()

            validation_result = {
                "is_valid": is_valid and is_fresh and len(consistency_errors) == 0,
                "layout_valid": is_valid,
                "is_fresh": is_fresh,
                "manifest_age_hours": manifest_age_hours,
                "consistency_errors": consistency_errors,
                "validation_errors": self.current_manifest.validation_errors,
                "last_sync_time": self.last_sync_time,
                "sync_in_progress": self.sync_in_progress,
            }

            if validation_result["is_valid"]:
                logger.info("Manifest validation passed")
            else:
                logger.warning(f"Manifest validation failed: {validation_result}")

            return validation_result

        except Exception as e:
            logger.error(f"Manifest validation failed: {e}")
            return {
                "is_valid": False,
                "error": str(e),
                "last_sync_time": self.last_sync_time,
                "sync_in_progress": self.sync_in_progress,
            }

    def _get_manifest_age_hours(self) -> float:
        """Get manifest age in hours.

        Returns:
            Manifest age in hours

        """
        try:
            # Parse manifest timestamp
            manifest_time = time.strptime(self.current_manifest.updated_at, "%Y-%m-%d %H:%M:%S UTC")
            manifest_timestamp = time.mktime(manifest_time)

            # Calculate age
            current_time = time.time()
            age_seconds = current_time - manifest_timestamp
            age_hours = age_seconds / 3600

            return age_hours

        except Exception as e:
            logger.warning(f"Failed to calculate manifest age: {e}")
            return float("inf")  # Treat as very old if we can't parse

    def _check_manifest_consistency(self) -> list[str]:
        """Check manifest consistency.

        Returns:
            List of consistency errors

        """
        errors = []

        try:
            # Check path consistency
            if self.current_manifest.chunk_store_path != str(self.storage_layout.chunk_store_path):
                errors.append("Chunk store path mismatch")
            if self.current_manifest.bm25_index_path != str(self.storage_layout.bm25_index_path):
                errors.append("BM25 index path mismatch")
            if self.current_manifest.splade_index_path != str(
                self.storage_layout.splade_index_path
            ):
                errors.append("SPLADE index path mismatch")
            if self.current_manifest.qwen3_index_path != str(self.storage_layout.qwen3_index_path):
                errors.append("Qwen3 index path mismatch")
            if self.current_manifest.raw_artifacts_path != str(
                self.storage_layout.raw_artifacts_path
            ):
                errors.append("Raw artifacts path mismatch")
            if self.current_manifest.doctags_path != str(self.storage_layout.doctags_path):
                errors.append("DocTags path mismatch")

            # Check size consistency
            current_sizes = self.storage_layout.get_storage_stats()
            if (
                current_sizes.get("chunk_store_size_bytes")
                != self.current_manifest.chunk_store_size_bytes
            ):
                errors.append("Chunk store size mismatch")
            if (
                current_sizes.get("bm25_index_size_bytes")
                != self.current_manifest.bm25_index_size_bytes
            ):
                errors.append("BM25 index size mismatch")
            if (
                current_sizes.get("splade_index_size_bytes")
                != self.current_manifest.splade_index_size_bytes
            ):
                errors.append("SPLADE index size mismatch")
            if (
                current_sizes.get("qwen3_index_size_bytes")
                != self.current_manifest.qwen3_index_size_bytes
            ):
                errors.append("Qwen3 index size mismatch")
            if (
                current_sizes.get("raw_artifacts_size_bytes")
                != self.current_manifest.raw_artifacts_size_bytes
            ):
                errors.append("Raw artifacts size mismatch")
            if current_sizes.get("doctags_size_bytes") != self.current_manifest.doctags_size_bytes:
                errors.append("DocTags size mismatch")

        except Exception as e:
            errors.append(f"Consistency check failed: {e}")

        return errors

    def backup_manifest(self) -> bool:
        """Create manifest backup.

        Returns:
            True if backup was successful, False otherwise

        """
        if not self.config.enable_backup:
            logger.debug("Manifest backup disabled")
            return True

        try:
            logger.info("Creating manifest backup")

            # Create backup directory
            backup_dir = self.storage_layout.base_path / "backups"
            backup_dir.mkdir(exist_ok=True)

            # Create backup filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            backup_filename = f"storage_manifest_backup_{timestamp}.json"
            backup_path = backup_dir / backup_filename

            # Write backup
            with backup_path.open("w", encoding="utf-8") as f:
                json.dump(self.current_manifest.dict(), f, indent=2, ensure_ascii=False)

            # Clean up old backups
            self._cleanup_old_backups(backup_dir)

            logger.info(f"Manifest backup created: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Manifest backup failed: {e}")
            return False

    def _cleanup_old_backups(self, backup_dir: Path) -> None:
        """Clean up old backup files.

        Args:
            backup_dir: Backup directory path

        """
        try:
            cutoff_time = time.time() - (self.config.max_manifest_age_hours * 3600)

            for backup_file in backup_dir.glob("storage_manifest_backup_*.json"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    logger.debug(f"Removed old backup: {backup_file}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    def restore_manifest(self, backup_path: str | Path) -> bool:
        """Restore manifest from backup.

        Args:
            backup_path: Path to backup file

        Returns:
            True if restore was successful, False otherwise

        """
        try:
            logger.info(f"Restoring manifest from backup: {backup_path}")

            # Load backup
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            with backup_path.open("r", encoding="utf-8") as f:
                backup_data = json.load(f)

            # Create manifest from backup
            restored_manifest = StorageManifest(**backup_data)

            # Save restored manifest
            self.storage_layout._save_manifest(restored_manifest)

            # Update current manifest
            self.current_manifest = restored_manifest

            # Sync with storage layout
            self.sync_manifest(force=True)

            logger.info("Manifest restore completed successfully")
            return True

        except Exception as e:
            logger.error(f"Manifest restore failed: {e}")
            return False

    def get_manifest_info(self) -> dict[str, Any]:
        """Get manifest information.

        Returns:
            Manifest information

        """
        try:
            validation_result = self.validate_manifest()

            info = {
                "manifest": self.current_manifest.dict(),
                "validation": validation_result,
                "sync_status": {
                    "last_sync_time": self.last_sync_time,
                    "sync_in_progress": self.sync_in_progress,
                    "auto_sync_enabled": self.config.enable_auto_sync,
                    "sync_interval_seconds": self.config.sync_interval_seconds,
                },
                "config": self.config.dict(),
                "storage_stats": self.storage_layout.get_storage_stats(),
            }

            return info

        except Exception as e:
            logger.error(f"Failed to get manifest info: {e}")
            return {"error": str(e)}

    def update_manifest(self, **kwargs: Any) -> bool:
        """Update manifest with new data.

        Args:
            **kwargs: Fields to update in manifest

        Returns:
            True if update was successful, False otherwise

        """
        try:
            logger.info("Updating manifest", extra=kwargs)

            # Update manifest
            self.storage_layout.update_manifest(**kwargs)

            # Update current manifest
            self.current_manifest = self.storage_layout.manifest

            # Sync if auto-sync is enabled
            if self.config.enable_auto_sync:
                self.sync_manifest(force=True)

            logger.info("Manifest update completed successfully")
            return True

        except Exception as e:
            logger.error(f"Manifest update failed: {e}")
            return False

    def health_check(self) -> dict[str, Any]:
        """Check manifest manager health.

        Returns:
            Health status information

        """
        try:
            validation_result = self.validate_manifest()

            health = {
                "status": "healthy" if validation_result["is_valid"] else "unhealthy",
                "validation": validation_result,
                "sync_status": {
                    "last_sync_time": self.last_sync_time,
                    "sync_in_progress": self.sync_in_progress,
                    "auto_sync_enabled": self.config.enable_auto_sync,
                },
                "storage_health": self.storage_layout.health_check(),
                "config": self.config.dict(),
            }

            return health

        except Exception as e:
            logger.error("Manifest manager health check failed", extra={"error": str(e)})
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def cleanup(self) -> None:
        """Clean up manifest manager."""
        try:
            # Sync manifest one last time
            if self.config.enable_auto_sync:
                self.sync_manifest(force=True)

            # Clean up storage layout
            self.storage_layout.cleanup()

            logger.info("Manifest manager cleanup completed")

        except Exception as e:
            logger.error(f"Manifest manager cleanup failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.cleanup()
