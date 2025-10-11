"""Unit and integration tests for the Index stage implementation.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from Medical_KG_rev.config.retrieval_config import BM25Config, Qwen3Config, SPLADEConfig
from Medical_KG_rev.models.chunk import Chunk
from Medical_KG_rev.pipeline.index_stage import IndexStageConfig, IndexStageImpl


class TestIndexStageImpl:
    """Test cases for IndexStageImpl."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for testing."""
        return tmp_path

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        chunk_store_path = temp_dir / "chunks.db"
        index_output_dir = temp_dir / "indexes"
        return IndexStageConfig(
            chunk_store_path=chunk_store_path,
            index_output_dir=index_output_dir,
            bm25_config=BM25Config(),
            splade_config=SPLADEConfig(),
            qwen3_config=Qwen3Config(),
            batch_size=2,
            max_workers=1,
        )

    @pytest.fixture
    def mock_chunk(self):
        """Create mock chunk for testing."""
        chunk = Mock(spec=Chunk)
        chunk.chunk_id = "test_chunk_1"
        chunk.doc_id = "test_doc_1"
        chunk.page_number = 1
        chunk.section_path = "Introduction"
        chunk.contextualized_text = "This is a test chunk with medical content."
        chunk.content_only_text = "This is a test chunk with medical content."

        # Add feature attributes
        chunk.bm25_fields = {"title": "Test Title", "paragraph": "Test content"}
        chunk.splade_vector = {
            "vector": {"term1": 0.5, "term2": 0.3},
            "model_name": "splade-v3",
            "tokenizer_name": "naver/splade-v3",
            "segment_count": 1,
        }
        chunk.qwen3_embedding = [0.1, 0.2, 0.3, 0.4]

        return chunk

    @pytest.fixture
    def stage(self, config):
        """Create IndexStageImpl instance."""
        return IndexStageImpl(config)

    def test_initialization(self, stage):
        """Test stage initialization."""
        assert stage.config is not None
        assert stage.chunk_store is None
        assert stage.manifest_manager is None
        assert stage.bm25_index is None
        assert stage.splade_index is None
        assert stage.qwen3_index is None
        assert stage.bm25_service is None
        assert stage.splade_service is None
        assert stage.qwen3_service is None

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Index")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEImpactIndex")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Index")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Service")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEService")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Service")
    def test_initialize_success(
        self,
        mock_qwen3_service,
        mock_splade_service,
        mock_bm25_service,
        mock_qwen3_index,
        mock_splade_index,
        mock_bm25_index,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
    ):
        """Test successful initialization."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_manifest_manager.return_value = Mock()
        mock_bm25_index.return_value = Mock()
        mock_splade_index.return_value = Mock()
        mock_qwen3_index.return_value = Mock()
        mock_bm25_service.return_value = Mock()
        mock_splade_service.return_value = Mock()
        mock_qwen3_service.return_value = Mock()

        # Initialize
        stage.initialize()

        # Verify initialization
        assert stage.chunk_store is not None
        assert stage.manifest_manager is not None
        assert stage.bm25_index is not None
        assert stage.splade_index is not None
        assert stage.qwen3_index is not None
        assert stage.bm25_service is not None
        assert stage.splade_service is not None
        assert stage.qwen3_service is not None

    def test_initialize_failure(self, stage):
        """Test initialization failure."""
        # Mock Path.mkdir to raise exception
        with patch("pathlib.Path.mkdir", side_effect=Exception("Permission denied")):
            with pytest.raises(Exception):
                stage.initialize()

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Index")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEImpactIndex")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Index")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Service")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEService")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Service")
    def test_build_indexes_success(
        self,
        mock_qwen3_service,
        mock_splade_service,
        mock_bm25_service,
        mock_qwen3_index,
        mock_splade_index,
        mock_bm25_index,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
        mock_chunk,
    ):
        """Test successful index building."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_chunk_store.return_value.get_all_chunks.return_value = [mock_chunk]
        mock_manifest_manager.return_value = Mock()
        mock_bm25_index.return_value.add_document.return_value = None
        mock_splade_index.return_value.add_vector.return_value = None
        mock_qwen3_index.return_value.add_vector.return_value = None

        # Initialize
        stage.initialize()

        # Build indexes
        results = stage.build_indexes()

        # Verify results
        assert results["total_chunks"] == 1
        assert results["indexed_chunks"] == 1
        assert results["bm25_indexed"] == 1
        assert results["splade_indexed"] == 1
        assert results["qwen3_indexed"] == 1
        assert len(results["errors"]) == 0

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Index")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEImpactIndex")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Index")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Service")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEService")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Service")
    def test_build_indexes_with_errors(
        self,
        mock_qwen3_service,
        mock_splade_service,
        mock_bm25_service,
        mock_qwen3_index,
        mock_splade_index,
        mock_bm25_index,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
        mock_chunk,
    ):
        """Test index building with errors."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_chunk_store.return_value.get_all_chunks.return_value = [mock_chunk]
        mock_manifest_manager.return_value = Mock()
        mock_bm25_index.return_value.add_document.side_effect = Exception("BM25 error")
        mock_splade_index.return_value.add_vector.side_effect = Exception("SPLADE error")
        mock_qwen3_index.return_value.add_vector.side_effect = Exception("Qwen3 error")

        # Initialize
        stage.initialize()

        # Build indexes
        results = stage.build_indexes()

        # Verify results
        assert results["total_chunks"] == 1
        assert results["indexed_chunks"] == 0
        assert results["bm25_indexed"] == 0
        assert results["splade_indexed"] == 0
        assert results["qwen3_indexed"] == 0
        assert len(results["errors"]) == 3  # One error per index type

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    def test_build_indexes_not_initialized(self, mock_manifest_manager, mock_chunk_store, stage):
        """Test index building when stage is not initialized."""
        with pytest.raises(RuntimeError, match="Index stage not initialized"):
            stage.build_indexes()

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Index")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEImpactIndex")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Index")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Service")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEService")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Service")
    def test_build_bm25_index_success(
        self,
        mock_qwen3_service,
        mock_splade_service,
        mock_bm25_service,
        mock_qwen3_index,
        mock_splade_index,
        mock_bm25_index,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
        mock_chunk,
    ):
        """Test successful BM25 index building."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_manifest_manager.return_value = Mock()
        mock_bm25_index.return_value.add_document.return_value = None

        # Initialize
        stage.initialize()

        # Build BM25 index
        results = stage._build_bm25_index([mock_chunk])

        # Verify results
        assert results["indexed"] == 1
        assert len(results["errors"]) == 0

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Index")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEImpactIndex")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Index")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Service")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEService")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Service")
    def test_build_splade_index_success(
        self,
        mock_qwen3_service,
        mock_splade_service,
        mock_bm25_service,
        mock_qwen3_index,
        mock_splade_index,
        mock_bm25_index,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
        mock_chunk,
    ):
        """Test successful SPLADE index building."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_manifest_manager.return_value = Mock()
        mock_splade_index.return_value.add_vector.return_value = None

        # Initialize
        stage.initialize()

        # Build SPLADE index
        results = stage._build_splade_index([mock_chunk])

        # Verify results
        assert results["indexed"] == 1
        assert len(results["errors"]) == 0

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Index")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEImpactIndex")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Index")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Service")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEService")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Service")
    def test_build_qwen3_index_success(
        self,
        mock_qwen3_service,
        mock_splade_service,
        mock_bm25_service,
        mock_qwen3_index,
        mock_splade_index,
        mock_bm25_index,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
        mock_chunk,
    ):
        """Test successful Qwen3 index building."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_manifest_manager.return_value = Mock()
        mock_qwen3_index.return_value.add_vector.return_value = None

        # Initialize
        stage.initialize()

        # Build Qwen3 index
        results = stage._build_qwen3_index([mock_chunk])

        # Verify results
        assert results["indexed"] == 1
        assert len(results["errors"]) == 0

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Index")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEImpactIndex")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Index")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Service")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEService")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Service")
    def test_validate_indexes_success(
        self,
        mock_qwen3_service,
        mock_splade_service,
        mock_bm25_service,
        mock_qwen3_index,
        mock_splade_index,
        mock_bm25_index,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
    ):
        """Test successful index validation."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_manifest_manager.return_value = Mock()
        mock_bm25_index.return_value.validate_index.return_value = True
        mock_splade_index.return_value.validate_index.return_value = True
        mock_qwen3_index.return_value.validate_index.return_value = True

        # Initialize
        stage.initialize()

        # Validate indexes
        results = stage.validate_indexes()

        # Verify results
        assert results["bm25_valid"] is True
        assert results["splade_valid"] is True
        assert results["qwen3_valid"] is True
        assert len(results["errors"]) == 0

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    def test_create_manifest_success(self, mock_manifest_manager, mock_chunk_store, stage):
        """Test successful manifest creation."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_manifest_manager.return_value.create_manifest.return_value = {"manifest": "created"}

        # Initialize
        stage.initialize()

        # Create manifest
        manifest = stage.create_manifest()

        # Verify manifest
        assert manifest is not None
        assert "manifest" in manifest

    def test_create_manifest_not_initialized(self, stage):
        """Test manifest creation when stage is not initialized."""
        with pytest.raises(RuntimeError, match="Manifest manager not initialized"):
            stage.create_manifest()

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    def test_validate_manifest_success(self, mock_manifest_manager, mock_chunk_store, stage):
        """Test successful manifest validation."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_manifest_manager.return_value.validate_manifest.return_value = True

        # Initialize
        stage.initialize()

        # Validate manifest
        result = stage.validate_manifest(Path("test_manifest.json"))

        # Verify result
        assert result is True

    def test_validate_manifest_not_initialized(self, stage):
        """Test manifest validation when stage is not initialized."""
        result = stage.validate_manifest(Path("test_manifest.json"))
        assert result is False

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    def test_cleanup_success(self, mock_manifest_manager, mock_chunk_store, stage):
        """Test successful cleanup."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_chunk_store.return_value.close.return_value = None
        mock_manifest_manager.return_value = Mock()

        # Initialize
        stage.initialize()

        # Cleanup
        stage.cleanup()

        # Verify cleanup was called
        mock_chunk_store.return_value.close.assert_called_once()

    @patch("Medical_KG_rev.pipeline.index_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.index_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Index")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEImpactIndex")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Index")
    @patch("Medical_KG_rev.pipeline.index_stage.BM25Service")
    @patch("Medical_KG_rev.pipeline.index_stage.SPLADEService")
    @patch("Medical_KG_rev.pipeline.index_stage.Qwen3Service")
    def test_health_check_success(
        self,
        mock_qwen3_service,
        mock_splade_service,
        mock_bm25_service,
        mock_qwen3_index,
        mock_splade_index,
        mock_bm25_index,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
    ):
        """Test successful health check."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_chunk_store.return_value.health_check.return_value = {"status": "healthy"}
        mock_manifest_manager.return_value = Mock()
        mock_bm25_index.return_value.health_check.return_value = {"status": "healthy"}
        mock_splade_index.return_value.health_check.return_value = {"status": "healthy"}
        mock_qwen3_index.return_value.health_check.return_value = {"status": "healthy"}

        # Initialize
        stage.initialize()

        # Health check
        health = stage.health_check()

        # Verify health status
        assert health["stage"] == "index"
        assert health["status"] == "healthy"
        assert "components" in health
        assert "chunk_store" in health["components"]
        assert "bm25_index" in health["components"]
        assert "splade_index" in health["components"]
        assert "qwen3_index" in health["components"]

    def test_health_check_not_initialized(self, stage):
        """Test health check when stage is not initialized."""
        health = stage.health_check()

        # Verify health status
        assert health["stage"] == "index"
        assert health["status"] == "healthy"  # Default status
        assert "components" in health
