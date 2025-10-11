"""Unit and integration tests for the Features stage implementation.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from Medical_KG_rev.config.retrieval_config import BM25Config, Qwen3Config, SPLADEConfig
from Medical_KG_rev.models.chunk import Chunk
from Medical_KG_rev.pipeline.features_stage import FeaturesStageConfig, FeaturesStageImpl


class TestFeaturesStageImpl:
    """Test cases for FeaturesStageImpl."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for testing."""
        return tmp_path

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        chunk_store_path = temp_dir / "chunks.db"
        return FeaturesStageConfig(
            chunk_store_path=chunk_store_path,
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
        return chunk

    @pytest.fixture
    def stage(self, config):
        """Create FeaturesStageImpl instance."""
        return FeaturesStageImpl(config)

    def test_initialization(self, stage):
        """Test stage initialization."""
        assert stage.config is not None
        assert stage.chunk_store is None
        assert stage.manifest_manager is None
        assert stage.bm25_field_mapper is None
        assert stage.splade_segmenter is None
        assert stage.splade_aggregator is None
        assert stage.qwen3_contextualizer is None
        assert stage.qwen3_service is None

    @patch("Medical_KG_rev.pipeline.features_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.features_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.features_stage.BM25FieldMapper")
    @patch("Medical_KG_rev.pipeline.features_stage.SPLADESegmenter")
    @patch("Medical_KG_rev.pipeline.features_stage.SPLADEAggregator")
    @patch("Medical_KG_rev.pipeline.features_stage.Qwen3Contextualizer")
    @patch("Medical_KG_rev.pipeline.features_stage.Qwen3Service")
    def test_initialize_success(
        self,
        mock_qwen3_service,
        mock_qwen3_contextualizer,
        mock_splade_aggregator,
        mock_splade_segmenter,
        mock_bm25_field_mapper,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
    ):
        """Test successful initialization."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_manifest_manager.return_value = Mock()
        mock_bm25_field_mapper.return_value = Mock()
        mock_splade_segmenter.return_value = Mock()
        mock_splade_aggregator.return_value = Mock()
        mock_qwen3_contextualizer.return_value = Mock()
        mock_qwen3_service.return_value = Mock()

        # Initialize
        stage.initialize()

        # Verify initialization
        assert stage.chunk_store is not None
        assert stage.manifest_manager is not None
        assert stage.bm25_field_mapper is not None
        assert stage.splade_segmenter is not None
        assert stage.splade_aggregator is not None
        assert stage.qwen3_contextualizer is not None
        assert stage.qwen3_service is not None

    def test_initialize_failure(self, stage):
        """Test initialization failure."""
        # Mock Path.mkdir to raise exception
        with patch("pathlib.Path.mkdir", side_effect=Exception("Permission denied")):
            with pytest.raises(Exception):
                stage.initialize()

    @patch("Medical_KG_rev.pipeline.features_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.features_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.features_stage.BM25FieldMapper")
    @patch("Medical_KG_rev.pipeline.features_stage.SPLADESegmenter")
    @patch("Medical_KG_rev.pipeline.features_stage.SPLADEAggregator")
    @patch("Medical_KG_rev.pipeline.features_stage.Qwen3Contextualizer")
    @patch("Medical_KG_rev.pipeline.features_stage.Qwen3Service")
    def test_process_chunks_success(
        self,
        mock_qwen3_service,
        mock_qwen3_contextualizer,
        mock_splade_aggregator,
        mock_splade_segmenter,
        mock_bm25_field_mapper,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
        mock_chunk,
    ):
        """Test successful chunk processing."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_chunk_store.return_value.get_all_chunks.return_value = [mock_chunk]
        mock_manifest_manager.return_value = Mock()
        mock_bm25_field_mapper.return_value.map_chunk_to_bm25_fields.return_value = {
            "title": "Test Title",
            "paragraph": "Test content",
        }
        mock_splade_segmenter.return_value.segment_chunk_for_splade.return_value = [
            {"text": "Test segment", "tokens": 10}
        ]
        mock_splade_aggregator.return_value.aggregate_splade_segments.return_value = {
            "vector": {"term1": 0.5, "term2": 0.3},
            "metadata": {},
        }
        mock_qwen3_contextualizer.return_value.get_contextualized_text.return_value = (
            "Contextualized text"
        )
        mock_qwen3_service.return_value.generate_embedding.return_value = [0.1, 0.2, 0.3]

        # Initialize
        stage.initialize()

        # Process chunks
        results = stage.process_chunks()

        # Verify results
        assert results["total_chunks"] == 1
        assert results["processed_chunks"] == 1
        assert results["bm25_features"] == 1
        assert results["splade_features"] == 1
        assert results["qwen3_features"] == 1
        assert len(results["errors"]) == 0

    @patch("Medical_KG_rev.pipeline.features_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.features_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.features_stage.BM25FieldMapper")
    @patch("Medical_KG_rev.pipeline.features_stage.SPLADESegmenter")
    @patch("Medical_KG_rev.pipeline.features_stage.SPLADEAggregator")
    @patch("Medical_KG_rev.pipeline.features_stage.Qwen3Contextualizer")
    @patch("Medical_KG_rev.pipeline.features_stage.Qwen3Service")
    def test_process_chunks_with_errors(
        self,
        mock_qwen3_service,
        mock_qwen3_contextualizer,
        mock_splade_aggregator,
        mock_splade_segmenter,
        mock_bm25_field_mapper,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
        mock_chunk,
    ):
        """Test chunk processing with errors."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_chunk_store.return_value.get_all_chunks.return_value = [mock_chunk]
        mock_manifest_manager.return_value = Mock()
        mock_bm25_field_mapper.return_value.map_chunk_to_bm25_fields.side_effect = Exception(
            "BM25 error"
        )
        mock_splade_segmenter.return_value.segment_chunk_for_splade.side_effect = Exception(
            "SPLADE error"
        )
        mock_qwen3_contextualizer.return_value.get_contextualized_text.side_effect = Exception(
            "Qwen3 error"
        )

        # Initialize
        stage.initialize()

        # Process chunks
        results = stage.process_chunks()

        # Verify results
        assert results["total_chunks"] == 1
        assert results["processed_chunks"] == 0
        assert results["bm25_features"] == 0
        assert results["splade_features"] == 0
        assert results["qwen3_features"] == 0
        assert len(results["errors"]) == 3  # One error per feature type

    @patch("Medical_KG_rev.pipeline.features_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.features_stage.ManifestManager")
    def test_process_chunks_not_initialized(self, mock_manifest_manager, mock_chunk_store, stage):
        """Test chunk processing when stage is not initialized."""
        with pytest.raises(RuntimeError, match="Features stage not initialized"):
            stage.process_chunks()

    @patch("Medical_KG_rev.pipeline.features_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.features_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.features_stage.BM25FieldMapper")
    @patch("Medical_KG_rev.pipeline.features_stage.SPLADESegmenter")
    @patch("Medical_KG_rev.pipeline.features_stage.SPLADEAggregator")
    @patch("Medical_KG_rev.pipeline.features_stage.Qwen3Contextualizer")
    @patch("Medical_KG_rev.pipeline.features_stage.Qwen3Service")
    def test_generate_splade_vector_success(
        self,
        mock_qwen3_service,
        mock_qwen3_contextualizer,
        mock_splade_aggregator,
        mock_splade_segmenter,
        mock_bm25_field_mapper,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
        mock_chunk,
    ):
        """Test successful SPLADE vector generation."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_manifest_manager.return_value = Mock()
        mock_splade_segmenter.return_value.segment_chunk_for_splade.return_value = [
            {"text": "Test segment", "tokens": 10}
        ]
        mock_splade_aggregator.return_value.aggregate_splade_segments.return_value = {
            "vector": {"term1": 0.5, "term2": 0.3},
            "metadata": {},
        }

        # Initialize
        stage.initialize()

        # Generate SPLADE vector
        result = stage._generate_splade_vector(mock_chunk)

        # Verify result
        assert result is not None
        assert "vector" in result
        assert "segment_count" in result
        assert "model_name" in result
        assert "tokenizer_name" in result

    @patch("Medical_KG_rev.pipeline.features_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.features_stage.ManifestManager")
    @patch("Medical_KG_rev.pipeline.features_stage.BM25FieldMapper")
    @patch("Medical_KG_rev.pipeline.features_stage.SPLADESegmenter")
    @patch("Medical_KG_rev.pipeline.features_stage.SPLADEAggregator")
    @patch("Medical_KG_rev.pipeline.features_stage.Qwen3Contextualizer")
    @patch("Medical_KG_rev.pipeline.features_stage.Qwen3Service")
    def test_generate_qwen3_embedding_success(
        self,
        mock_qwen3_service,
        mock_qwen3_contextualizer,
        mock_splade_aggregator,
        mock_splade_segmenter,
        mock_bm25_field_mapper,
        mock_manifest_manager,
        mock_chunk_store,
        stage,
        mock_chunk,
    ):
        """Test successful Qwen3 embedding generation."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_manifest_manager.return_value = Mock()
        mock_qwen3_contextualizer.return_value.get_contextualized_text.return_value = (
            "Contextualized text"
        )
        mock_qwen3_service.return_value.generate_embedding.return_value = [0.1, 0.2, 0.3]

        # Initialize
        stage.initialize()

        # Generate Qwen3 embedding
        result = stage._generate_qwen3_embedding(mock_chunk)

        # Verify result
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 3

    @patch("Medical_KG_rev.pipeline.features_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.features_stage.ManifestManager")
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

    @patch("Medical_KG_rev.pipeline.features_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.features_stage.ManifestManager")
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

    @patch("Medical_KG_rev.pipeline.features_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.features_stage.ManifestManager")
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

    @patch("Medical_KG_rev.pipeline.features_stage.ChunkStore")
    @patch("Medical_KG_rev.pipeline.features_stage.ManifestManager")
    def test_health_check_success(self, mock_manifest_manager, mock_chunk_store, stage):
        """Test successful health check."""
        # Setup mocks
        mock_chunk_store.return_value.initialize.return_value = None
        mock_chunk_store.return_value.health_check.return_value = {"status": "healthy"}
        mock_manifest_manager.return_value = Mock()

        # Initialize
        stage.initialize()

        # Health check
        health = stage.health_check()

        # Verify health status
        assert health["stage"] == "features"
        assert health["status"] == "healthy"
        assert "components" in health
        assert "chunk_store" in health["components"]

    def test_health_check_not_initialized(self, stage):
        """Test health check when stage is not initialized."""
        health = stage.health_check()

        # Verify health status
        assert health["stage"] == "features"
        assert health["status"] == "healthy"  # Default status
        assert "components" in health
