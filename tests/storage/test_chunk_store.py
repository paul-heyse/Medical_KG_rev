"""Tests for chunk store database."""

from unittest.mock import Mock, patch

import pytest

from Medical_KG_rev.storage.chunk_store import ChunkRecord, ChunkStore, ChunkStoreConfig


class TestChunkRecord:
    """Test ChunkRecord model."""

    def test_chunk_record_creation(self):
        """Test creating a ChunkRecord."""
        chunk = ChunkRecord(
            chunk_id="chunk_1",
            doc_id="doc_1",
            doctags_sha="sha123",
            page_no=1,
            element_label="paragraph",
            contextualized_text="Test text",
            content_only_text="Test content",
        )

        assert chunk.chunk_id == "chunk_1"
        assert chunk.doc_id == "doc_1"
        assert chunk.doctags_sha == "sha123"
        assert chunk.page_no == 1
        assert chunk.element_label == "paragraph"
        assert chunk.contextualized_text == "Test text"
        assert chunk.content_only_text == "Test content"
        assert chunk.bbox == {}
        assert chunk.section_path == ""
        assert chunk.char_start == 0
        assert chunk.char_end == 0
        assert chunk.table_payload == {}

    def test_chunk_record_with_optional_fields(self):
        """Test creating a ChunkRecord with optional fields."""
        chunk = ChunkRecord(
            chunk_id="chunk_2",
            doc_id="doc_2",
            doctags_sha="sha456",
            page_no=2,
            element_label="table",
            bbox={"x": 0, "y": 0, "width": 100, "height": 50},
            section_path="Introduction/Methods",
            char_start=100,
            char_end=200,
            contextualized_text="Table content",
            content_only_text="Table data",
            table_payload={"rows": 3, "cols": 2},
        )

        assert chunk.bbox == {"x": 0, "y": 0, "width": 100, "height": 50}
        assert chunk.section_path == "Introduction/Methods"
        assert chunk.char_start == 100
        assert chunk.char_end == 200
        assert chunk.table_payload == {"rows": 3, "cols": 2}


class TestChunkStoreConfig:
    """Test ChunkStoreConfig model."""

    def test_default_config(self):
        """Test default configuration."""
        config = ChunkStoreConfig()

        assert config.database_path == "chunk_store.duckdb"
        assert config.enable_analytics is True
        assert config.enable_validation is True
        assert config.max_text_length == 100000
        assert config.enable_compression is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkStoreConfig(
            database_path="custom_store.duckdb",
            enable_analytics=False,
            enable_validation=False,
            max_text_length=50000,
            enable_compression=False,
        )

        assert config.database_path == "custom_store.duckdb"
        assert config.enable_analytics is False
        assert config.enable_validation is False
        assert config.max_text_length == 50000
        assert config.enable_compression is False


@patch("Medical_KG_rev.storage.chunk_store.duckdb")
class TestChunkStore:
    """Test ChunkStore class."""

    def test_chunk_store_initialization(self, mock_duckdb):
        """Test ChunkStore initialization."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        config = ChunkStoreConfig(database_path="test_store.duckdb")
        store = ChunkStore(config)

        assert store.config == config
        assert store.conn == mock_conn
        mock_duckdb.connect.assert_called_once_with("test_store.duckdb")

    def test_chunk_store_default_config(self, mock_duckdb):
        """Test ChunkStore with default config."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        store = ChunkStore()

        assert isinstance(store.config, ChunkStoreConfig)
        assert store.config.database_path == "chunk_store.duckdb"

    def test_chunk_store_missing_duckdb(self, mock_duckdb):
        """Test ChunkStore when duckdb is not available."""
        mock_duckdb = None

        with patch("Medical_KG_rev.storage.chunk_store.duckdb", None):
            with pytest.raises(ImportError, match="duckdb is required but not installed"):
                ChunkStore()

    def test_add_chunk(self, mock_duckdb):
        """Test adding a chunk to the store."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        store = ChunkStore()

        chunk = ChunkRecord(
            chunk_id="chunk_1",
            doc_id="doc_1",
            doctags_sha="sha123",
            page_no=1,
            element_label="paragraph",
            contextualized_text="Test text",
        )

        store.add_chunk(chunk)

        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called_once()

    def test_get_chunk(self, mock_duckdb):
        """Test getting a chunk from the store."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        # Mock database result
        mock_result = (
            "chunk_1",
            "doc_1",
            "sha123",
            1,
            "{}",
            "paragraph",
            "",
            0,
            0,
            "Test text",
            "Test content",
            "{}",
            "2024-01-01 00:00:00",
        )
        mock_conn.execute.return_value.fetchone.return_value = mock_result

        store = ChunkStore()

        chunk = store.get_chunk("chunk_1")

        assert chunk is not None
        assert chunk.chunk_id == "chunk_1"
        assert chunk.doc_id == "doc_1"
        assert chunk.element_label == "paragraph"

    def test_get_chunk_not_found(self, mock_duckdb):
        """Test getting a non-existent chunk."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        mock_conn.execute.return_value.fetchone.return_value = None

        store = ChunkStore()

        chunk = store.get_chunk("nonexistent")

        assert chunk is None

    def test_get_chunks_by_doc(self, mock_duckdb):
        """Test getting chunks by document."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        # Mock database results
        mock_results = [
            (
                "chunk_1",
                "doc_1",
                "sha123",
                1,
                "{}",
                "paragraph",
                "",
                0,
                0,
                "Test text 1",
                "Test content 1",
                "{}",
                "2024-01-01 00:00:00",
            ),
            (
                "chunk_2",
                "doc_1",
                "sha123",
                2,
                "{}",
                "paragraph",
                "",
                0,
                0,
                "Test text 2",
                "Test content 2",
                "{}",
                "2024-01-01 00:00:00",
            ),
        ]
        mock_conn.execute.return_value.fetchall.return_value = mock_results

        store = ChunkStore()

        chunks = store.get_chunks_by_doc("doc_1")

        assert len(chunks) == 2
        assert chunks[0].chunk_id == "chunk_1"
        assert chunks[1].chunk_id == "chunk_2"

    def test_remove_chunk(self, mock_duckdb):
        """Test removing a chunk from the store."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        # Mock successful deletion
        mock_result = Mock()
        mock_result.rowcount = 1
        mock_conn.execute.return_value = mock_result

        store = ChunkStore()

        result = store.remove_chunk("chunk_1")

        assert result is True
        mock_conn.commit.assert_called_once()

    def test_remove_chunk_not_found(self, mock_duckdb):
        """Test removing a non-existent chunk."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        # Mock no deletion
        mock_result = Mock()
        mock_result.rowcount = 0
        mock_conn.execute.return_value = mock_result

        store = ChunkStore()

        result = store.remove_chunk("nonexistent")

        assert result is False

    def test_get_store_stats(self, mock_duckdb):
        """Test getting store statistics."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        # Mock database result
        mock_result = (100, 10, 5, 500.0, 400.0, "2024-01-01", "2024-01-02")
        mock_conn.execute.return_value.fetchone.return_value = mock_result

        store = ChunkStore()

        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 1024000

            stats = store.get_store_stats()

            assert stats["total_chunks"] == 100
            assert stats["total_documents"] == 10
            assert stats["element_types"] == 5
            assert stats["avg_contextualized_length"] == 500.0
            assert stats["avg_content_length"] == 400.0
            assert stats["database_size_bytes"] == 1024000

    def test_health_check(self, mock_duckdb):
        """Test chunk store health check."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        store = ChunkStore()

        with patch.object(store, "get_store_stats") as mock_stats:
            mock_stats.return_value = {
                "total_chunks": 100,
                "total_documents": 10,
                "database_size_bytes": 1024000,
                "first_created": "2024-01-01",
                "last_created": "2024-01-02",
                "config": {},
            }

            health = store.health_check()

            assert health["status"] == "healthy"
            assert health["total_chunks"] == 100
            assert health["total_documents"] == 10
            assert health["database_size_bytes"] == 1024000

    def test_close(self, mock_duckdb):
        """Test closing the database connection."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        store = ChunkStore()
        store.close()

        mock_conn.close.assert_called_once()

    def test_context_manager(self, mock_duckdb):
        """Test context manager functionality."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        with ChunkStore() as store:
            assert store.conn == mock_conn

        mock_conn.close.assert_called_once()


class TestChunkStoreValidation:
    """Test chunk validation functionality."""

    @patch("Medical_KG_rev.storage.chunk_store.duckdb")
    def test_validate_chunk_missing_fields(self, mock_duckdb):
        """Test validation with missing required fields."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        store = ChunkStore()

        # Test missing chunk_id
        chunk = ChunkRecord(
            chunk_id="",  # Empty chunk_id
            doc_id="doc_1",
            doctags_sha="sha123",
            page_no=1,
            element_label="paragraph",
        )

        with pytest.raises(ValueError, match="chunk_id is required"):
            store.add_chunk(chunk)

    @patch("Medical_KG_rev.storage.chunk_store.duckdb")
    def test_validate_chunk_text_length(self, mock_duckdb):
        """Test validation with text length limits."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        config = ChunkStoreConfig(max_text_length=100)
        store = ChunkStore(config)

        # Test text too long
        chunk = ChunkRecord(
            chunk_id="chunk_1",
            doc_id="doc_1",
            doctags_sha="sha123",
            page_no=1,
            element_label="paragraph",
            contextualized_text="x" * 200,  # Too long
        )

        with pytest.raises(ValueError, match="contextualized_text too long"):
            store.add_chunk(chunk)

    @patch("Medical_KG_rev.storage.chunk_store.duckdb")
    def test_validate_chunk_char_positions(self, mock_duckdb):
        """Test validation with invalid character positions."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        store = ChunkStore()

        # Test invalid character positions
        chunk = ChunkRecord(
            chunk_id="chunk_1",
            doc_id="doc_1",
            doctags_sha="sha123",
            page_no=1,
            element_label="paragraph",
            char_start=100,
            char_end=50,  # char_end < char_start
        )

        with pytest.raises(ValueError, match="char_start must be <= char_end"):
            store.add_chunk(chunk)
