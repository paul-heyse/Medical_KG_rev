"""Tests for manifest management system."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from Medical_KG_rev.storage.manifest_manager import ManifestManager, ManifestManagerConfig


class TestManifestManagerConfig:
    """Test ManifestManagerConfig model."""

    def test_default_config(self):
        """Test default configuration."""
        config = ManifestManagerConfig()

        assert config.storage_layout_path == "storage"
        assert config.enable_auto_sync is True
        assert config.sync_interval_seconds == 300
        assert config.enable_validation is True
        assert config.enable_backup is True
        assert config.max_manifest_age_hours == 24
        assert config.enable_compression is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ManifestManagerConfig(
            storage_layout_path="/custom/storage",
            enable_auto_sync=False,
            sync_interval_seconds=600,
            enable_validation=False,
            enable_backup=False,
            max_manifest_age_hours=48,
            enable_compression=False,
        )

        assert config.storage_layout_path == "/custom/storage"
        assert config.enable_auto_sync is False
        assert config.sync_interval_seconds == 600
        assert config.enable_validation is False
        assert config.enable_backup is False
        assert config.max_manifest_age_hours == 48
        assert config.enable_compression is False


@patch("Medical_KG_rev.storage.manifest_manager.StorageLayout")
class TestManifestManager:
    """Test ManifestManager class."""

    def test_manifest_manager_initialization(self, mock_storage_layout):
        """Test ManifestManager initialization."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        config = ManifestManagerConfig(storage_layout_path="/test/storage")
        manager = ManifestManager(config)

        assert manager.config == config
        assert manager.storage_layout == mock_layout
        assert manager.current_manifest == mock_manifest
        assert manager.last_sync_time > 0
        assert manager.sync_in_progress is False

    def test_manifest_manager_default_config(self, mock_storage_layout):
        """Test ManifestManager with default config."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        assert isinstance(manager.config, ManifestManagerConfig)
        assert manager.config.storage_layout_path == "storage"

    def test_sync_manifest(self, mock_storage_layout):
        """Test manifest synchronization."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        result = manager.sync_manifest(force=True)

        assert result is True
        assert manager.sync_in_progress is False
        mock_layout._update_storage_sizes.assert_called_once()
        mock_layout._save_manifest.assert_called_once_with(mock_manifest)

    def test_sync_manifest_already_in_progress(self, mock_storage_layout):
        """Test manifest sync when already in progress."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()
        manager.sync_in_progress = True

        result = manager.sync_manifest()

        assert result is False
        mock_layout._update_storage_sizes.assert_not_called()

    def test_sync_manifest_error(self, mock_storage_layout):
        """Test manifest sync with error."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_layout._update_storage_sizes.side_effect = Exception("Sync error")
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        result = manager.sync_manifest(force=True)

        assert result is False
        assert manager.sync_in_progress is False

    def test_should_sync(self, mock_storage_layout):
        """Test sync timing logic."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        config = ManifestManagerConfig(sync_interval_seconds=1)
        manager = ManifestManager(config)

        # Should sync immediately
        assert manager._should_sync() is True

        # Update sync time
        manager.last_sync_time = time.time()

        # Should not sync immediately after
        assert manager._should_sync() is False

        # Wait and should sync again
        time.sleep(1.1)
        assert manager._should_sync() is True

    def test_should_sync_disabled(self, mock_storage_layout):
        """Test sync timing when auto-sync is disabled."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        config = ManifestManagerConfig(enable_auto_sync=False)
        manager = ManifestManager(config)

        assert manager._should_sync() is False

    def test_validate_manifest(self, mock_storage_layout):
        """Test manifest validation."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_manifest.updated_at = "2024-01-01 00:00:00 UTC"
        mock_manifest.validation_errors = []
        mock_layout.manifest = mock_manifest
        mock_layout.validate_layout.return_value = True
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        # Mock manifest age calculation
        with patch.object(manager, "_get_manifest_age_hours", return_value=1.0):
            with patch.object(manager, "_check_manifest_consistency", return_value=[]):
                result = manager.validate_manifest()

        assert "is_valid" in result
        assert "layout_valid" in result
        assert "is_fresh" in result
        assert "manifest_age_hours" in result
        assert "consistency_errors" in result
        assert "validation_errors" in result
        assert "last_sync_time" in result
        assert "sync_in_progress" in result

    def test_get_manifest_age_hours(self, mock_storage_layout):
        """Test manifest age calculation."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_manifest.updated_at = "2024-01-01 00:00:00 UTC"
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        # Mock current time
        with patch(
            "time.time",
            return_value=time.mktime(time.strptime("2024-01-01 01:00:00", "%Y-%m-%d %H:%M:%S")),
        ):
            age_hours = manager._get_manifest_age_hours()

        assert age_hours == 1.0

    def test_get_manifest_age_hours_error(self, mock_storage_layout):
        """Test manifest age calculation with error."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_manifest.updated_at = "invalid_timestamp"
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        age_hours = manager._get_manifest_age_hours()

        assert age_hours == float("inf")

    def test_check_manifest_consistency(self, mock_storage_layout):
        """Test manifest consistency check."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_manifest.chunk_store_path = "/storage/chunk_store"
        mock_manifest.bm25_index_path = "/storage/bm25_index"
        mock_manifest.splade_index_path = "/storage/splade_index"
        mock_manifest.qwen3_index_path = "/storage/qwen3_index"
        mock_manifest.raw_artifacts_path = "/storage/raw_artifacts"
        mock_manifest.doctags_path = "/storage/doctags"
        mock_manifest.chunk_store_size_bytes = 1000
        mock_manifest.bm25_index_size_bytes = 2000
        mock_manifest.splade_index_size_bytes = 3000
        mock_manifest.qwen3_index_size_bytes = 4000
        mock_manifest.raw_artifacts_size_bytes = 5000
        mock_manifest.doctags_size_bytes = 6000
        mock_layout.manifest = mock_manifest
        mock_layout.chunk_store_path = Path("/storage/chunk_store")
        mock_layout.bm25_index_path = Path("/storage/bm25_index")
        mock_layout.splade_index_path = Path("/storage/splade_index")
        mock_layout.qwen3_index_path = Path("/storage/qwen3_index")
        mock_layout.raw_artifacts_path = Path("/storage/raw_artifacts")
        mock_layout.doctags_path = Path("/storage/doctags")
        mock_layout.get_storage_stats.return_value = {
            "chunk_store_size_bytes": 1000,
            "bm25_index_size_bytes": 2000,
            "splade_index_size_bytes": 3000,
            "qwen3_index_size_bytes": 4000,
            "raw_artifacts_size_bytes": 5000,
            "doctags_size_bytes": 6000,
        }
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        errors = manager._check_manifest_consistency()

        assert errors == []

    def test_check_manifest_consistency_with_errors(self, mock_storage_layout):
        """Test manifest consistency check with errors."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_manifest.chunk_store_path = "/storage/chunk_store"
        mock_manifest.bm25_index_path = "/storage/bm25_index"
        mock_manifest.splade_index_path = "/storage/splade_index"
        mock_manifest.qwen3_index_path = "/storage/qwen3_index"
        mock_manifest.raw_artifacts_path = "/storage/raw_artifacts"
        mock_manifest.doctags_path = "/storage/doctags"
        mock_manifest.chunk_store_size_bytes = 1000
        mock_manifest.bm25_index_size_bytes = 2000
        mock_manifest.splade_index_size_bytes = 3000
        mock_manifest.qwen3_index_size_bytes = 4000
        mock_manifest.raw_artifacts_size_bytes = 5000
        mock_manifest.doctags_size_bytes = 6000
        mock_layout.manifest = mock_manifest
        mock_layout.chunk_store_path = Path("/different/chunk_store")  # Mismatch
        mock_layout.bm25_index_path = Path("/storage/bm25_index")
        mock_layout.splade_index_path = Path("/storage/splade_index")
        mock_layout.qwen3_index_path = Path("/storage/qwen3_index")
        mock_layout.raw_artifacts_path = Path("/storage/raw_artifacts")
        mock_layout.doctags_path = Path("/storage/doctags")
        mock_layout.get_storage_stats.return_value = {
            "chunk_store_size_bytes": 2000,  # Mismatch
            "bm25_index_size_bytes": 2000,
            "splade_index_size_bytes": 3000,
            "qwen3_index_size_bytes": 4000,
            "raw_artifacts_size_bytes": 5000,
            "doctags_size_bytes": 6000,
        }
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        errors = manager._check_manifest_consistency()

        assert "Chunk store path mismatch" in errors
        assert "Chunk store size mismatch" in errors

    def test_backup_manifest(self, mock_storage_layout):
        """Test manifest backup creation."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_manifest.dict.return_value = {"version": "1.0", "test": "data"}
        mock_layout.manifest = mock_manifest
        mock_layout.base_path = Path("/storage")
        mock_storage_layout.return_value = mock_layout

        config = ManifestManagerConfig(enable_backup=True)
        manager = ManifestManager(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(mock_layout, "base_path", Path(temp_dir)):
                result = manager.backup_manifest()

        assert result is True
        mock_manifest.dict.assert_called_once()

    def test_backup_manifest_disabled(self, mock_storage_layout):
        """Test manifest backup when disabled."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        config = ManifestManagerConfig(enable_backup=False)
        manager = ManifestManager(config)

        result = manager.backup_manifest()

        assert result is True
        mock_manifest.dict.assert_not_called()

    def test_backup_manifest_error(self, mock_storage_layout):
        """Test manifest backup with error."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_manifest.dict.side_effect = Exception("Backup error")
        mock_layout.manifest = mock_manifest
        mock_layout.base_path = Path("/storage")
        mock_storage_layout.return_value = mock_layout

        config = ManifestManagerConfig(enable_backup=True)
        manager = ManifestManager(config)

        result = manager.backup_manifest()

        assert result is False

    def test_restore_manifest(self, mock_storage_layout):
        """Test manifest restore."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            backup_path = Path(temp_dir) / "backup.json"
            backup_data = {"version": "1.0", "test": "data"}

            with backup_path.open("w", encoding="utf-8") as f:
                json.dump(backup_data, f)

            with patch(
                "Medical_KG_rev.storage.manifest_manager.StorageManifest"
            ) as mock_storage_manifest_class:
                mock_restored_manifest = Mock()
                mock_storage_manifest_class.return_value = mock_restored_manifest

                result = manager.restore_manifest(backup_path)

        assert result is True
        mock_layout._save_manifest.assert_called_once_with(mock_restored_manifest)
        mock_layout._update_storage_sizes.assert_called_once()

    def test_restore_manifest_file_not_found(self, mock_storage_layout):
        """Test manifest restore with missing file."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        result = manager.restore_manifest("/nonexistent/backup.json")

        assert result is False

    def test_get_manifest_info(self, mock_storage_layout):
        """Test getting manifest information."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_manifest.dict.return_value = {"version": "1.0", "test": "data"}
        mock_layout.manifest = mock_manifest
        mock_layout.get_storage_stats.return_value = {"total_size_bytes": 1000}
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        with patch.object(manager, "validate_manifest", return_value={"is_valid": True}):
            info = manager.get_manifest_info()

        assert "manifest" in info
        assert "validation" in info
        assert "sync_status" in info
        assert "config" in info
        assert "storage_stats" in info

    def test_update_manifest(self, mock_storage_layout):
        """Test updating manifest."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        config = ManifestManagerConfig(enable_auto_sync=True)
        manager = ManifestManager(config)

        with patch.object(manager, "sync_manifest", return_value=True):
            result = manager.update_manifest(total_documents=100)

        assert result is True
        mock_layout.update_manifest.assert_called_once_with(total_documents=100)

    def test_update_manifest_error(self, mock_storage_layout):
        """Test updating manifest with error."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_layout.update_manifest.side_effect = Exception("Update error")
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        result = manager.update_manifest(total_documents=100)

        assert result is False

    def test_health_check(self, mock_storage_layout):
        """Test health check."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_layout.health_check.return_value = {"status": "healthy"}
        mock_storage_layout.return_value = mock_layout

        manager = ManifestManager()

        with patch.object(manager, "validate_manifest", return_value={"is_valid": True}):
            health = manager.health_check()

        assert "status" in health
        assert "validation" in health
        assert "sync_status" in health
        assert "storage_health" in health
        assert "config" in health

    def test_cleanup(self, mock_storage_layout):
        """Test cleanup."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        config = ManifestManagerConfig(enable_auto_sync=True)
        manager = ManifestManager(config)

        with patch.object(manager, "sync_manifest", return_value=True):
            manager.cleanup()

        mock_layout.cleanup.assert_called_once()

    def test_context_manager(self, mock_storage_layout):
        """Test context manager functionality."""
        mock_layout = Mock()
        mock_manifest = Mock()
        mock_layout.manifest = mock_manifest
        mock_storage_layout.return_value = mock_layout

        with ManifestManager() as manager:
            assert manager.storage_layout == mock_layout
            assert manager.current_manifest == mock_manifest

        mock_layout.cleanup.assert_called_once()
