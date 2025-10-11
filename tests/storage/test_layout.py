"""Tests for storage layout management."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from Medical_KG_rev.storage.layout import StorageLayout, StorageLayoutConfig, StorageManifest


class TestStorageManifest:
    """Test StorageManifest model."""

    def test_storage_manifest_creation(self):
        """Test creating a StorageManifest."""
        manifest = StorageManifest(
            chunk_store_path="/storage/chunk_store",
            bm25_index_path="/storage/bm25_index",
            splade_index_path="/storage/splade_index",
            qwen3_index_path="/storage/qwen3_index",
            raw_artifacts_path="/storage/raw_artifacts",
            doctags_path="/storage/doctags",
        )

        assert manifest.version == "1.0"
        assert manifest.chunk_store_path == "/storage/chunk_store"
        assert manifest.bm25_index_path == "/storage/bm25_index"
        assert manifest.splade_index_path == "/storage/splade_index"
        assert manifest.qwen3_index_path == "/storage/qwen3_index"
        assert manifest.raw_artifacts_path == "/storage/raw_artifacts"
        assert manifest.doctags_path == "/storage/doctags"
        assert manifest.total_documents == 0
        assert manifest.total_chunks == 0
        assert manifest.is_valid is True
        assert manifest.validation_errors == []

    def test_storage_manifest_with_metadata(self):
        """Test creating a StorageManifest with metadata."""
        manifest = StorageManifest(
            chunk_store_path="/storage/chunk_store",
            bm25_index_path="/storage/bm25_index",
            splade_index_path="/storage/splade_index",
            qwen3_index_path="/storage/qwen3_index",
            raw_artifacts_path="/storage/raw_artifacts",
            doctags_path="/storage/doctags",
            total_documents=100,
            total_chunks=1000,
            total_tokens=50000,
            bm25_documents=100,
            splade_documents=100,
            qwen3_documents=100,
            chunk_store_size_bytes=1024000,
            bm25_index_size_bytes=512000,
            splade_index_size_bytes=256000,
            qwen3_index_size_bytes=128000,
            raw_artifacts_size_bytes=2048000,
            doctags_size_bytes=1024000,
        )

        assert manifest.total_documents == 100
        assert manifest.total_chunks == 1000
        assert manifest.total_tokens == 50000
        assert manifest.bm25_documents == 100
        assert manifest.splade_documents == 100
        assert manifest.qwen3_documents == 100
        assert manifest.chunk_store_size_bytes == 1024000
        assert manifest.bm25_index_size_bytes == 512000
        assert manifest.splade_index_size_bytes == 256000
        assert manifest.qwen3_index_size_bytes == 128000
        assert manifest.raw_artifacts_size_bytes == 2048000
        assert manifest.doctags_size_bytes == 1024000


class TestStorageLayoutConfig:
    """Test StorageLayoutConfig model."""

    def test_default_config(self):
        """Test default configuration."""
        config = StorageLayoutConfig()

        assert config.base_path == "storage"
        assert config.manifest_filename == "storage_manifest.json"
        assert config.enable_validation is True
        assert config.enable_compression is True
        assert config.max_file_size_mb == 1000
        assert config.enable_backup is True
        assert config.backup_retention_days == 30

    def test_custom_config(self):
        """Test custom configuration."""
        config = StorageLayoutConfig(
            base_path="/custom/storage",
            manifest_filename="custom_manifest.json",
            enable_validation=False,
            enable_compression=False,
            max_file_size_mb=500,
            enable_backup=False,
            backup_retention_days=7,
        )

        assert config.base_path == "/custom/storage"
        assert config.manifest_filename == "custom_manifest.json"
        assert config.enable_validation is False
        assert config.enable_compression is False
        assert config.max_file_size_mb == 500
        assert config.enable_backup is False
        assert config.backup_retention_days == 7


class TestStorageLayout:
    """Test StorageLayout class."""

    def test_storage_layout_initialization(self):
        """Test StorageLayout initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)
            layout = StorageLayout(config)

            assert layout.config == config
            assert layout.base_path == Path(temp_dir)
            assert layout.chunk_store_path == Path(temp_dir) / "chunk_store"
            assert layout.bm25_index_path == Path(temp_dir) / "bm25_index"
            assert layout.splade_index_path == Path(temp_dir) / "splade_index"
            assert layout.qwen3_index_path == Path(temp_dir) / "qwen3_index"
            assert layout.raw_artifacts_path == Path(temp_dir) / "raw_artifacts"
            assert layout.doctags_path == Path(temp_dir) / "doctags"

            # Check directories were created
            assert layout.chunk_store_path.exists()
            assert layout.bm25_index_path.exists()
            assert layout.splade_index_path.exists()
            assert layout.qwen3_index_path.exists()
            assert layout.raw_artifacts_path.exists()
            assert layout.doctags_path.exists()

    def test_storage_layout_default_config(self):
        """Test StorageLayout with default config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("Medical_KG_rev.storage.layout.Path") as mock_path:
                mock_path.return_value = Path(temp_dir)
                layout = StorageLayout()

                assert isinstance(layout.config, StorageLayoutConfig)
                assert layout.config.base_path == "storage"

    def test_manifest_path_property(self):
        """Test manifest path property."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)
            layout = StorageLayout(config)

            expected_path = Path(temp_dir) / "storage_manifest.json"
            assert layout.manifest_path == expected_path

    def test_load_existing_manifest(self):
        """Test loading existing manifest."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)
            layout = StorageLayout(config)

            # Create a manifest file
            manifest_data = {
                "version": "1.0",
                "chunk_store_path": str(layout.chunk_store_path),
                "bm25_index_path": str(layout.bm25_index_path),
                "splade_index_path": str(layout.splade_index_path),
                "qwen3_index_path": str(layout.qwen3_index_path),
                "raw_artifacts_path": str(layout.raw_artifacts_path),
                "doctags_path": str(layout.doctags_path),
                "total_documents": 50,
                "total_chunks": 500,
            }

            with layout.manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest_data, f)

            # Create new layout instance to load manifest
            layout2 = StorageLayout(config)

            assert layout2.manifest.total_documents == 50
            assert layout2.manifest.total_chunks == 500

    def test_create_new_manifest(self):
        """Test creating new manifest."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)
            layout = StorageLayout(config)

            # Check manifest was created
            assert layout.manifest_path.exists()

            # Check manifest content
            with layout.manifest_path.open("r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            assert manifest_data["version"] == "1.0"
            assert manifest_data["chunk_store_path"] == str(layout.chunk_store_path)
            assert manifest_data["bm25_index_path"] == str(layout.bm25_index_path)
            assert manifest_data["splade_index_path"] == str(layout.splade_index_path)
            assert manifest_data["qwen3_index_path"] == str(layout.qwen3_index_path)
            assert manifest_data["raw_artifacts_path"] == str(layout.raw_artifacts_path)
            assert manifest_data["doctags_path"] == str(layout.doctags_path)

    def test_validate_layout(self):
        """Test layout validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)
            layout = StorageLayout(config)

            # Layout should be valid
            assert layout.validate_layout() is True
            assert layout.manifest.is_valid is True
            assert layout.manifest.validation_errors == []

    def test_validate_layout_missing_directories(self):
        """Test layout validation with missing directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)
            layout = StorageLayout(config)

            # Remove a directory
            layout.chunk_store_path.rmdir()

            # Layout should be invalid
            assert layout.validate_layout() is False
            assert layout.manifest.is_valid is False
            assert len(layout.manifest.validation_errors) > 0

    def test_update_manifest(self):
        """Test updating manifest."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)
            layout = StorageLayout(config)

            # Update manifest
            layout.update_manifest(
                total_documents=100,
                total_chunks=1000,
                total_tokens=50000,
            )

            # Check updates
            assert layout.manifest.total_documents == 100
            assert layout.manifest.total_chunks == 1000
            assert layout.manifest.total_tokens == 50000

            # Check manifest was saved
            with layout.manifest_path.open("r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            assert manifest_data["total_documents"] == 100
            assert manifest_data["total_chunks"] == 1000
            assert manifest_data["total_tokens"] == 50000

    def test_get_storage_stats(self):
        """Test getting storage statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)
            layout = StorageLayout(config)

            # Create some test files
            test_file = layout.chunk_store_path / "test.txt"
            test_file.write_text("test content")

            stats = layout.get_storage_stats()

            assert "total_size_bytes" in stats
            assert "total_size_mb" in stats
            assert "total_size_gb" in stats
            assert "chunk_store_size_bytes" in stats
            assert "bm25_index_size_bytes" in stats
            assert "splade_index_size_bytes" in stats
            assert "qwen3_index_size_bytes" in stats
            assert "raw_artifacts_size_bytes" in stats
            assert "doctags_size_bytes" in stats
            assert "total_documents" in stats
            assert "total_chunks" in stats
            assert "total_tokens" in stats
            assert "is_valid" in stats
            assert "created_at" in stats
            assert "updated_at" in stats

    def test_get_path(self):
        """Test getting path for storage type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)
            layout = StorageLayout(config)

            assert layout.get_path("chunk_store") == layout.chunk_store_path
            assert layout.get_path("bm25_index") == layout.bm25_index_path
            assert layout.get_path("splade_index") == layout.splade_index_path
            assert layout.get_path("qwen3_index") == layout.qwen3_index_path
            assert layout.get_path("raw_artifacts") == layout.raw_artifacts_path
            assert layout.get_path("doctags") == layout.doctags_path

    def test_get_path_invalid_type(self):
        """Test getting path for invalid storage type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)
            layout = StorageLayout(config)

            with pytest.raises(ValueError, match="Unknown storage type"):
                layout.get_path("invalid_type")

    def test_health_check(self):
        """Test health check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)
            layout = StorageLayout(config)

            health = layout.health_check()

            assert "status" in health
            assert "is_valid" in health
            assert "validation_errors" in health
            assert "total_size_bytes" in health
            assert "total_documents" in health
            assert "total_chunks" in health
            assert "created_at" in health
            assert "updated_at" in health

    def test_context_manager(self):
        """Test context manager functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir)

            with StorageLayout(config) as layout:
                assert layout.base_path == Path(temp_dir)
                assert layout.chunk_store_path.exists()
                assert layout.bm25_index_path.exists()
                assert layout.splade_index_path.exists()
                assert layout.qwen3_index_path.exists()
                assert layout.raw_artifacts_path.exists()
                assert layout.doctags_path.exists()

    def test_backup_creation(self):
        """Test manifest backup creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir, enable_backup=True)
            layout = StorageLayout(config)

            # Update manifest to trigger backup
            layout.update_manifest(total_documents=100)

            # Check backup was created
            backup_dir = layout.base_path / "backups"
            assert backup_dir.exists()

            backup_files = list(backup_dir.glob("storage_manifest_*.json"))
            assert len(backup_files) > 0

    def test_backup_disabled(self):
        """Test manifest backup when disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir, enable_backup=False)
            layout = StorageLayout(config)

            # Update manifest
            layout.update_manifest(total_documents=100)

            # Check no backup was created
            backup_dir = layout.base_path / "backups"
            assert not backup_dir.exists()

    def test_file_size_validation(self):
        """Test file size validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageLayoutConfig(base_path=temp_dir, max_file_size_mb=1)
            layout = StorageLayout(config)

            # Create a large file
            large_file = layout.chunk_store_path / "large.txt"
            large_file.write_text("x" * (2 * 1024 * 1024))  # 2MB file

            # Layout should be invalid due to large file
            assert layout.validate_layout() is False
            assert layout.manifest.is_valid is False
            assert len(layout.manifest.validation_errors) > 0
