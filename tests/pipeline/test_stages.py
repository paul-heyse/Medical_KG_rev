"""Tests for pipeline stages."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from Medical_KG_rev.pipeline.stages import (
    ChunkStage,
    ConvertStage,
    FeaturesStage,
    IndexStage,
    PipelineStage,
    StageResult,
    StageStatus,
)


class TestStageResult:
    """Test StageResult model."""

    def test_stage_result_creation(self):
        """Test creating a StageResult."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.COMPLETED,
            start_time=1000.0,
            end_time=1001.0,
            duration_seconds=1.0,
            input_data={"test": "input"},
            output_data={"test": "output"},
        )

        assert result.stage_name == "test_stage"
        assert result.status == StageStatus.COMPLETED
        assert result.start_time == 1000.0
        assert result.end_time == 1001.0
        assert result.duration_seconds == 1.0
        assert result.input_data == {"test": "input"}
        assert result.output_data == {"test": "output"}
        assert result.error_message is None
        assert result.metadata == {}

    def test_stage_result_with_error(self):
        """Test creating a StageResult with error."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.FAILED,
            start_time=1000.0,
            end_time=1001.0,
            duration_seconds=1.0,
            input_data={"test": "input"},
            error_message="Test error",
            metadata={"error_code": 500},
        )

        assert result.status == StageStatus.FAILED
        assert result.error_message == "Test error"
        assert result.metadata == {"error_code": 500}
        assert result.output_data == {}


class TestPipelineStage:
    """Test PipelineStage abstract base class."""

    def test_pipeline_stage_initialization(self):
        """Test PipelineStage initialization."""

        class TestStage(PipelineStage):
            def execute(self, input_data):
                return StageResult(
                    stage_name=self.name,
                    status=StageStatus.COMPLETED,
                    start_time=1000.0,
                    input_data=input_data,
                )

        stage = TestStage("test_stage", {"test": "config"})

        assert stage.name == "test_stage"
        assert stage.config == {"test": "config"}
        assert stage.logger.name == "Medical_KG_rev.pipeline.stages.test_stage"

    def test_validate_input_default(self):
        """Test default input validation."""

        class TestStage(PipelineStage):
            def execute(self, input_data):
                return StageResult(
                    stage_name=self.name,
                    status=StageStatus.COMPLETED,
                    start_time=1000.0,
                    input_data=input_data,
                )

        stage = TestStage("test_stage")

        assert stage.validate_input({"test": "data"}) is True
        assert stage.validate_input({}) is True

    def test_cleanup(self):
        """Test stage cleanup."""

        class TestStage(PipelineStage):
            def execute(self, input_data):
                return StageResult(
                    stage_name=self.name,
                    status=StageStatus.COMPLETED,
                    start_time=1000.0,
                    input_data=input_data,
                )

        stage = TestStage("test_stage")
        stage.cleanup()  # Should not raise an exception


class TestConvertStage:
    """Test ConvertStage class."""

    def test_convert_stage_initialization(self):
        """Test ConvertStage initialization."""
        config = {
            "enable_docling": True,
            "output_format": "doctags",
        }

        stage = ConvertStage(config)

        assert stage.name == "convert"
        assert stage.config == config
        assert stage.enable_docling is True
        assert stage.output_format == "doctags"

    def test_convert_stage_default_config(self):
        """Test ConvertStage with default config."""
        stage = ConvertStage()

        assert stage.enable_docling is True
        assert stage.output_format == "doctags"

    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        stage = ConvertStage()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"test pdf content")
            temp_path = temp_file.name

        try:
            input_data = {
                "document_id": "doc_1",
                "pdf_path": temp_path,
                "docling_service_url": "http://localhost:8000",
            }

            assert stage.validate_input(input_data) is True
        finally:
            Path(temp_path).unlink()

    def test_validate_input_missing_fields(self):
        """Test input validation with missing fields."""
        stage = ConvertStage()

        # Missing document_id
        input_data = {"pdf_path": "/path/to/file.pdf"}
        assert stage.validate_input(input_data) is False

        # Missing pdf_path
        input_data = {"document_id": "doc_1"}
        assert stage.validate_input(input_data) is False

    def test_validate_input_nonexistent_file(self):
        """Test input validation with nonexistent file."""
        stage = ConvertStage()

        input_data = {
            "document_id": "doc_1",
            "pdf_path": "/nonexistent/file.pdf",
        }

        assert stage.validate_input(input_data) is False

    @patch("Medical_KG_rev.pipeline.stages.DoclingVLMService")
    def test_execute_with_docling(self, mock_docling_service):
        """Test execute with Docling service."""
        # Mock Docling service
        mock_service_instance = Mock()
        mock_result = Mock()
        mock_result.processing_time_seconds = 1.0
        mock_result.gpu_memory_used_mb = 1024
        mock_service_instance.process_pdf.return_value = mock_result
        mock_docling_service.return_value = mock_service_instance

        stage = ConvertStage({"enable_docling": True})

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"test pdf content")
            temp_path = temp_file.name

        try:
            input_data = {
                "document_id": "doc_1",
                "pdf_path": temp_path,
                "docling_service_url": "http://localhost:8000",
            }

            result = stage.execute(input_data)

            assert result.stage_name == "convert"
            assert result.status == StageStatus.COMPLETED
            assert result.input_data == input_data
            assert "document_id" in result.output_data
            assert "processing_method" in result.output_data
            assert result.output_data["processing_method"] == "docling"
            assert result.error_message is None

            mock_service_instance.process_pdf.assert_called_once_with(temp_path, "doc_1")
        finally:
            Path(temp_path).unlink()


    def test_execute_invalid_input(self):
        """Test execute with invalid input."""
        stage = ConvertStage()

        input_data = {"invalid": "data"}

        result = stage.execute(input_data)

        assert result.stage_name == "convert"
        assert result.status == StageStatus.FAILED
        assert result.input_data == input_data
        assert result.error_message is not None
        assert "Invalid input data" in result.error_message


class TestChunkStage:
    """Test ChunkStage class."""

    def test_chunk_stage_initialization(self):
        """Test ChunkStage initialization."""
        config = {
            "chunker_type": "hybrid",
            "max_tokens": 256,
            "overlap_tokens": 25,
            "enable_tokenizer_alignment": True,
        }

        stage = ChunkStage(config)

        assert stage.name == "chunk"
        assert stage.config == config
        assert stage.chunker_type == "hybrid"
        assert stage.max_tokens == 256
        assert stage.overlap_tokens == 25
        assert stage.enable_tokenizer_alignment is True

    def test_chunk_stage_default_config(self):
        """Test ChunkStage with default config."""
        stage = ChunkStage()

        assert stage.chunker_type == "hybrid"
        assert stage.max_tokens == 512
        assert stage.overlap_tokens == 50
        assert stage.enable_tokenizer_alignment is True

    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        stage = ChunkStage()

        input_data = {
            "document_id": "doc_1",
            "result": Mock(),
        }

        assert stage.validate_input(input_data) is True

    def test_validate_input_missing_fields(self):
        """Test input validation with missing fields."""
        stage = ChunkStage()

        # Missing document_id
        input_data = {"result": Mock()}
        assert stage.validate_input(input_data) is False

        # Missing result
        input_data = {"document_id": "doc_1"}
        assert stage.validate_input(input_data) is False

    @patch("Medical_KG_rev.pipeline.stages.get_chunker")
    def test_execute_success(self, mock_get_chunker):
        """Test successful chunk stage execution."""
        # Mock chunker
        mock_chunker = Mock()
        mock_chunks = [Mock(), Mock(), Mock()]
        mock_chunker.chunk.return_value = mock_chunks
        mock_get_chunker.return_value = mock_chunker

        stage = ChunkStage()

        input_data = {
            "document_id": "doc_1",
            "result": Mock(),
        }

        result = stage.execute(input_data)

        assert result.stage_name == "chunk"
        assert result.status == StageStatus.COMPLETED
        assert result.input_data == input_data
        assert "document_id" in result.output_data
        assert "chunks" in result.output_data
        assert "chunk_count" in result.output_data
        assert result.output_data["chunk_count"] == 3
        assert result.error_message is None

        mock_chunker.chunk.assert_called_once()

    def test_execute_invalid_input(self):
        """Test execute with invalid input."""
        stage = ChunkStage()

        input_data = {"invalid": "data"}

        result = stage.execute(input_data)

        assert result.stage_name == "chunk"
        assert result.status == StageStatus.FAILED
        assert result.input_data == input_data
        assert result.error_message is not None
        assert "Invalid input data" in result.error_message


class TestFeaturesStage:
    """Test FeaturesStage class."""

    def test_features_stage_initialization(self):
        """Test FeaturesStage initialization."""
        config = {
            "enable_bm25": True,
            "enable_splade": False,
            "enable_qwen3": True,
            "batch_size": 16,
        }

        stage = FeaturesStage(config)

        assert stage.name == "features"
        assert stage.config == config
        assert stage.enable_bm25 is True
        assert stage.enable_splade is False
        assert stage.enable_qwen3 is True
        assert stage.batch_size == 16

    def test_features_stage_default_config(self):
        """Test FeaturesStage with default config."""
        stage = FeaturesStage()

        assert stage.enable_bm25 is True
        assert stage.enable_splade is True
        assert stage.enable_qwen3 is True
        assert stage.batch_size == 32

    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        stage = FeaturesStage()

        input_data = {
            "document_id": "doc_1",
            "chunks": [Mock(), Mock()],
        }

        assert stage.validate_input(input_data) is True

    def test_validate_input_missing_fields(self):
        """Test input validation with missing fields."""
        stage = FeaturesStage()

        # Missing document_id
        input_data = {"chunks": [Mock()]}
        assert stage.validate_input(input_data) is False

        # Missing chunks
        input_data = {"document_id": "doc_1"}
        assert stage.validate_input(input_data) is False

    @patch("Medical_KG_rev.pipeline.stages.BM25Service")
    def test_execute_with_bm25(self, mock_bm25_service):
        """Test execute with BM25 features."""
        # Mock BM25 service
        mock_service_instance = Mock()
        mock_features = [Mock(), Mock()]
        mock_service_instance.process_batch.return_value = mock_features
        mock_bm25_service.return_value = mock_service_instance

        stage = FeaturesStage({"enable_bm25": True, "enable_splade": False, "enable_qwen3": False})

        input_data = {
            "document_id": "doc_1",
            "chunks": [Mock(), Mock()],
        }

        result = stage.execute(input_data)

        assert result.stage_name == "features"
        assert result.status == StageStatus.COMPLETED
        assert result.input_data == input_data
        assert "document_id" in result.output_data
        assert "chunks" in result.output_data
        assert "features" in result.output_data
        assert "bm25" in result.output_data["features"]
        assert result.error_message is None

        mock_service_instance.process_batch.assert_called_once()

    def test_execute_invalid_input(self):
        """Test execute with invalid input."""
        stage = FeaturesStage()

        input_data = {"invalid": "data"}

        result = stage.execute(input_data)

        assert result.stage_name == "features"
        assert result.status == StageStatus.FAILED
        assert result.input_data == input_data
        assert result.error_message is not None
        assert "Invalid input data" in result.error_message


class TestIndexStage:
    """Test IndexStage class."""

    def test_index_stage_initialization(self):
        """Test IndexStage initialization."""
        config = {
            "enable_chunk_store": True,
            "enable_bm25_index": False,
            "enable_splade_index": True,
            "enable_qwen3_index": False,
            "batch_size": 50,
        }

        stage = IndexStage(config)

        assert stage.name == "index"
        assert stage.config == config
        assert stage.enable_chunk_store is True
        assert stage.enable_bm25_index is False
        assert stage.enable_splade_index is True
        assert stage.enable_qwen3_index is False
        assert stage.batch_size == 50

    def test_index_stage_default_config(self):
        """Test IndexStage with default config."""
        stage = IndexStage()

        assert stage.enable_chunk_store is True
        assert stage.enable_bm25_index is True
        assert stage.enable_splade_index is True
        assert stage.enable_qwen3_index is True
        assert stage.batch_size == 100

    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        stage = IndexStage()

        input_data = {
            "document_id": "doc_1",
            "chunks": [Mock(), Mock()],
            "features": {"bm25": Mock()},
        }

        assert stage.validate_input(input_data) is True

    def test_validate_input_missing_fields(self):
        """Test input validation with missing fields."""
        stage = IndexStage()

        # Missing document_id
        input_data = {"chunks": [Mock()]}
        assert stage.validate_input(input_data) is False

        # Missing chunks
        input_data = {"document_id": "doc_1"}
        assert stage.validate_input(input_data) is False

    @patch("Medical_KG_rev.pipeline.stages.ChunkStore")
    def test_execute_with_chunk_store(self, mock_chunk_store):
        """Test execute with chunk store."""
        # Mock chunk store
        mock_store_instance = Mock()
        mock_chunk_store.return_value = mock_store_instance

        stage = IndexStage(
            {
                "enable_chunk_store": True,
                "enable_bm25_index": False,
                "enable_splade_index": False,
                "enable_qwen3_index": False,
            }
        )

        # Mock chunks
        mock_chunks = [Mock(), Mock()]
        for chunk in mock_chunks:
            chunk.chunk_id = f"chunk_{len(mock_chunks)}"
            chunk.doctags_sha = "sha123"
            chunk.page_no = 1
            chunk.bbox = {}
            chunk.element_label = "paragraph"
            chunk.section_path = ""
            chunk.char_start = 0
            chunk.char_end = 100
            chunk.contextualized_text = "Test text"
            chunk.content_only_text = "Test content"
            chunk.table_payload = {}

        input_data = {
            "document_id": "doc_1",
            "chunks": mock_chunks,
            "features": {},
        }

        result = stage.execute(input_data)

        assert result.stage_name == "index"
        assert result.status == StageStatus.COMPLETED
        assert result.input_data == input_data
        assert "document_id" in result.output_data
        assert "chunks" in result.output_data
        assert "indexing_results" in result.output_data
        assert "chunk_store" in result.output_data["indexing_results"]
        assert result.error_message is None

        mock_store_instance.add_chunk.assert_called()

    def test_execute_invalid_input(self):
        """Test execute with invalid input."""
        stage = IndexStage()

        input_data = {"invalid": "data"}

        result = stage.execute(input_data)

        assert result.stage_name == "index"
        assert result.status == StageStatus.FAILED
        assert result.input_data == input_data
        assert result.error_message is not None
        assert "Invalid input data" in result.error_message
