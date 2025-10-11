"""Tests for chunk stage implementation."""

from unittest.mock import Mock, patch

from Medical_KG_rev.pipeline.chunk_stage import ChunkStageConfig, ChunkStageImpl


class TestChunkStageConfig:
    """Test ChunkStageConfig model."""

    def test_chunk_stage_config_creation(self):
        """Test creating a ChunkStageConfig."""
        config = ChunkStageConfig(
            chunker_type="hybrid",
            max_tokens=256,
            overlap_tokens=25,
            enable_tokenizer_alignment=True,
            tokenizer_name="naver/splade-v3",
            enable_hierarchical_chunking=True,
            preserve_structure=True,
            enable_metadata_extraction=True,
            batch_size=16,
            max_chunk_size_bytes=5000,
        )

        assert config.chunker_type == "hybrid"
        assert config.max_tokens == 256
        assert config.overlap_tokens == 25
        assert config.enable_tokenizer_alignment is True
        assert config.tokenizer_name == "naver/splade-v3"
        assert config.enable_hierarchical_chunking is True
        assert config.preserve_structure is True
        assert config.enable_metadata_extraction is True
        assert config.batch_size == 16
        assert config.max_chunk_size_bytes == 5000

    def test_chunk_stage_config_defaults(self):
        """Test ChunkStageConfig defaults."""
        config = ChunkStageConfig()

        assert config.chunker_type == "hybrid"
        assert config.max_tokens == 512
        assert config.overlap_tokens == 50
        assert config.enable_tokenizer_alignment is True
        assert config.tokenizer_name == "naver/splade-v3"
        assert config.enable_hierarchical_chunking is True
        assert config.preserve_structure is True
        assert config.enable_metadata_extraction is True
        assert config.batch_size == 32
        assert config.max_chunk_size_bytes == 10000


@patch("Medical_KG_rev.pipeline.chunk_stage.ChunkerFactory")
class TestChunkStageImpl:
    """Test ChunkStageImpl class."""

    def test_chunk_stage_impl_initialization(self, mock_factory):
        """Test ChunkStageImpl initialization."""
        config = ChunkStageConfig(
            chunker_type="hybrid",
            max_tokens=256,
            enable_tokenizer_alignment=True,
        )

        stage = ChunkStageImpl(config)

        assert stage.name == "chunk"
        assert stage.chunker_type == "hybrid"
        assert stage.max_tokens == 256
        assert stage.enable_tokenizer_alignment is True
        assert stage.tokenizer_name == "naver/splade-v3"
        assert stage.enable_hierarchical_chunking is True
        assert stage.preserve_structure is True
        assert stage.enable_metadata_extraction is True
        assert stage.batch_size == 32
        assert stage.max_chunk_size_bytes == 10000

    def test_chunk_stage_impl_default_config(self, mock_factory):
        """Test ChunkStageImpl with default config."""
        stage = ChunkStageImpl()

        assert isinstance(stage.stage_config, ChunkStageConfig)
        assert stage.chunker_type == "hybrid"
        assert stage.max_tokens == 512
        assert stage.overlap_tokens == 50

    def test_validate_input_valid(self, mock_factory):
        """Test input validation with valid data."""
        stage = ChunkStageImpl()

        input_data = {
            "document_id": "doc_1",
            "result": Mock(),
        }

        assert stage.validate_input(input_data) is True

    def test_validate_input_missing_fields(self, mock_factory):
        """Test input validation with missing fields."""
        stage = ChunkStageImpl()

        # Missing document_id
        input_data = {"result": Mock()}
        assert stage.validate_input(input_data) is False

        # Missing result
        input_data = {"document_id": "doc_1"}
        assert stage.validate_input(input_data) is False

    def test_validate_input_invalid_chunker_type(self, mock_factory):
        """Test input validation with invalid chunker type."""
        config = ChunkStageConfig(chunker_type="invalid")
        stage = ChunkStageImpl(config)

        input_data = {
            "document_id": "doc_1",
            "result": Mock(),
        }

        assert stage.validate_input(input_data) is False

    def test_validate_input_invalid_tokens(self, mock_factory):
        """Test input validation with invalid token settings."""
        config = ChunkStageConfig(max_tokens=0)
        stage = ChunkStageImpl(config)

        input_data = {
            "document_id": "doc_1",
            "result": Mock(),
        }

        assert stage.validate_input(input_data) is False

    def test_validate_input_invalid_overlap(self, mock_factory):
        """Test input validation with invalid overlap tokens."""
        config = ChunkStageConfig(overlap_tokens=1000, max_tokens=500)
        stage = ChunkStageImpl(config)

        input_data = {
            "document_id": "doc_1",
            "result": Mock(),
        }

        assert stage.validate_input(input_data) is False

    @patch("Medical_KG_rev.pipeline.chunk_stage.ChunkerConfig")
    def test_execute_success(self, mock_config_class, mock_factory):
        """Test successful chunk stage execution."""
        # Mock chunker factory
        mock_factory_instance = Mock()
        mock_registered_chunker = Mock()
        mock_chunker = Mock()
        mock_chunks = [Mock(), Mock(), Mock()]
        mock_chunker.chunk.return_value = mock_chunks
        mock_registered_chunker.instance = mock_chunker
        mock_factory_instance.create.return_value = mock_registered_chunker
        mock_factory.return_value = mock_factory_instance

        # Mock config
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        stage = ChunkStageImpl()

        input_data = {
            "document_id": "doc_1",
            "result": Mock(),
        }

        result = stage.execute(input_data)

        assert result.stage_name == "chunk"
        assert result.status.value == "completed"
        assert result.input_data == input_data
        assert "document_id" in result.output_data
        assert "chunks" in result.output_data
        assert "chunk_count" in result.output_data
        assert result.output_data["chunk_count"] == 3
        assert result.error_message is None

        mock_chunker.chunk.assert_called_once()

    def test_execute_invalid_input(self, mock_factory):
        """Test execute with invalid input."""
        stage = ChunkStageImpl()

        input_data = {"invalid": "data"}

        result = stage.execute(input_data)

        assert result.stage_name == "chunk"
        assert result.status.value == "failed"
        assert result.input_data == input_data
        assert result.error_message is not None
        assert "Invalid input data" in result.error_message

    def test_extract_chunk_metadata(self, mock_factory):
        """Test chunk metadata extraction."""
        stage = ChunkStageImpl()

        # Mock chunks
        mock_chunks = [Mock(), Mock()]
        for i, chunk in enumerate(mock_chunks):
            chunk.metadata = {}
            chunk.contextualized_text = f"Test text {i}"
            chunk.page_no = i + 1
            chunk.char_start = i * 100
            chunk.char_end = (i + 1) * 100

        enhanced_chunks = stage._extract_chunk_metadata(mock_chunks, "doc_1")

        assert len(enhanced_chunks) == 2
        for i, chunk in enumerate(enhanced_chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["document_id"] == "doc_1"
            assert chunk.metadata["chunker_type"] == "hybrid"
            assert chunk.metadata["text_length"] == len(f"Test text {i}")
            assert chunk.metadata["word_count"] == 2
            assert chunk.metadata["page_number"] == i + 1
            assert chunk.metadata["char_span"]["start"] == i * 100
            assert chunk.metadata["char_span"]["end"] == (i + 1) * 100
            assert chunk.metadata["char_span"]["length"] == 100

    def test_validate_chunk_sizes(self, mock_factory):
        """Test chunk size validation."""
        config = ChunkStageConfig(max_chunk_size_bytes=100)
        stage = ChunkStageImpl(config)

        # Mock chunks
        mock_chunks = [Mock(), Mock()]
        mock_chunks[0].contextualized_text = "short text"  # Small
        mock_chunks[1].contextualized_text = "x" * 200  # Large

        validated_chunks = stage._validate_chunk_sizes(mock_chunks)

        assert len(validated_chunks) == 1
        assert validated_chunks[0] == mock_chunks[0]

    def test_sort_chunks(self, mock_factory):
        """Test chunk sorting."""
        stage = ChunkStageImpl()

        # Mock chunks with different page numbers and positions
        mock_chunks = [Mock(), Mock(), Mock()]
        mock_chunks[0].page_no = 2
        mock_chunks[0].char_start = 100
        mock_chunks[1].page_no = 1
        mock_chunks[1].char_start = 200
        mock_chunks[2].page_no = 1
        mock_chunks[2].char_start = 50

        sorted_chunks = stage._sort_chunks(mock_chunks)

        assert sorted_chunks[0] == mock_chunks[2]  # page 1, char 50
        assert sorted_chunks[1] == mock_chunks[1]  # page 1, char 200
        assert sorted_chunks[2] == mock_chunks[0]  # page 2, char 100

    def test_get_stage_info(self, mock_factory):
        """Test getting stage information."""
        stage = ChunkStageImpl()

        info = stage.get_stage_info()

        assert "stage_name" in info
        assert "config" in info
        assert "capabilities" in info
        assert "limits" in info
        assert info["stage_name"] == "chunk"
        assert "hybrid" in info["capabilities"]["chunker_types"]
        assert info["limits"]["max_tokens"] == 512

    @patch("Medical_KG_rev.pipeline.chunk_stage.AutoTokenizer")
    def test_health_check(self, mock_tokenizer, mock_factory):
        """Test health check."""
        stage = ChunkStageImpl()

        health = stage.health_check()

        assert "status" in health
        assert "stage_name" in health
        assert "config" in health
        assert "capabilities" in health
        assert "limits" in health
        assert health["stage_name"] == "chunk"
        assert health["status"] == "healthy"

    @patch("Medical_KG_rev.pipeline.chunk_stage.AutoTokenizer")
    def test_health_check_tokenizer_error(self, mock_tokenizer, mock_factory):
        """Test health check with tokenizer error."""
        mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer error")

        stage = ChunkStageImpl()

        health = stage.health_check()

        assert health["status"] == "healthy"  # Overall status is still healthy
        assert health["tokenizer_status"] == "unhealthy"
        assert "tokenizer_error" in health
