"""Chunk store database using DuckDB.

This module implements a chunk store database for storing processed document
chunks with comprehensive metadata and analytics capabilities.
"""

from pathlib import Path
from typing import Any
import logging
import time

from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field
import duckdb


logger = logging.getLogger(__name__)

# Prometheus metrics
CHUNK_STORE_SECONDS = Histogram(
    "chunk_store_seconds",
    "Time spent on chunk store operations",
    ["operation", "status"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

CHUNK_STORE_OPERATIONS = Counter(
    "chunk_store_operations_total",
    "Total number of chunk store operations",
    ["operation", "status"],
)

CHUNK_STORE_SIZE_BYTES = Histogram(
    "chunk_store_size_bytes",
    "Size of chunk store database in bytes",
    ["operation"],
    buckets=[1024, 10240, 102400, 1024000, 10240000, 102400000],
)


class ChunkRecord(BaseModel):
    """Chunk record for database storage."""

    chunk_id: str = Field(..., description="Chunk identifier")
    doc_id: str = Field(..., description="Document identifier")
    doctags_sha: str = Field(..., description="DocTags SHA hash")
    page_no: int = Field(..., description="Page number")
    bbox: dict[str, Any] = Field(default_factory=dict, description="Bounding box coordinates")
    element_label: str = Field(..., description="Element label")
    section_path: str = Field(default="", description="Section path")
    char_start: int = Field(default=0, description="Character start position")
    char_end: int = Field(default=0, description="Character end position")
    contextualized_text: str = Field(default="", description="Contextualized text")
    content_only_text: str = Field(default="", description="Content-only text")
    table_payload: dict[str, Any] = Field(default_factory=dict, description="Table payload")
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        description="Creation timestamp",
    )


class ChunkStoreConfig(BaseModel):
    """Configuration for chunk store database."""

    database_path: str = Field(default="chunk_store.duckdb", description="Database file path")
    enable_analytics: bool = Field(default=True, description="Enable analytics views")
    enable_validation: bool = Field(default=True, description="Enable data validation")
    max_text_length: int = Field(default=100000, description="Maximum text length")
    enable_compression: bool = Field(default=True, description="Enable database compression")


class ChunkStore:
    """Chunk store database using DuckDB.

    This class implements efficient storage and retrieval of document chunks
    with comprehensive metadata and analytics capabilities.
    """

    def __init__(self, config: ChunkStoreConfig | None = None):
        """Initialize chunk store database.

        Args:
        ----
            config: Configuration for chunk store

        """
        self.config = config or ChunkStoreConfig()

        # Create database path
        self.db_path = Path(self.config.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize DuckDB connection
        if duckdb is None:
            raise ImportError("duckdb is required but not installed")
        self.conn = duckdb.connect(str(self.db_path))

        # Initialize database schema
        self._initialize_schema()

        logger.info(
            "Initialized chunk store database",
            extra={
                "database_path": str(self.db_path),
                "enable_analytics": self.config.enable_analytics,
                "enable_validation": self.config.enable_validation,
                "max_text_length": self.config.max_text_length,
                "enable_compression": self.config.enable_compression,
            },
        )

    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        try:
            # Create chunks table
            create_chunks_table = """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR PRIMARY KEY,
                doc_id VARCHAR NOT NULL,
                doctags_sha VARCHAR NOT NULL,
                page_no INTEGER NOT NULL,
                bbox JSON,
                element_label VARCHAR NOT NULL,
                section_path VARCHAR,
                char_start INTEGER DEFAULT 0,
                char_end INTEGER DEFAULT 0,
                contextualized_text TEXT,
                content_only_text TEXT,
                table_payload JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """

            self.conn.execute(create_chunks_table)

            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_element_label ON chunks(element_label)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_page_no ON chunks(page_no)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_section_path ON chunks(section_path)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks(created_at)",
            ]

            for index_sql in indexes:
                self.conn.execute(index_sql)

            # Create analytics views if enabled
            if self.config.enable_analytics:
                self._create_analytics_views()

            logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise

    def _create_analytics_views(self) -> None:
        """Create analytics views for chunk analysis."""
        try:
            # Chunks by label view
            chunks_by_label_view = """
            CREATE OR REPLACE VIEW chunks_by_label AS
            SELECT
                element_label,
                COUNT(*) as chunk_count,
                AVG(LENGTH(contextualized_text)) as avg_contextualized_length,
                AVG(LENGTH(content_only_text)) as avg_content_length,
                MIN(created_at) as first_created,
                MAX(created_at) as last_created
            FROM chunks
            GROUP BY element_label
            ORDER BY chunk_count DESC
            """

            self.conn.execute(chunks_by_label_view)

            # Token length distribution view
            token_length_view = """
            CREATE OR REPLACE VIEW token_length_distribution AS
            SELECT
                CASE
                    WHEN LENGTH(contextualized_text) < 100 THEN '0-100'
                    WHEN LENGTH(contextualized_text) < 500 THEN '100-500'
                    WHEN LENGTH(contextualized_text) < 1000 THEN '500-1000'
                    WHEN LENGTH(contextualized_text) < 2000 THEN '1000-2000'
                    ELSE '2000+'
                END as length_range,
                COUNT(*) as chunk_count,
                AVG(LENGTH(contextualized_text)) as avg_length
            FROM chunks
            GROUP BY length_range
            ORDER BY chunk_count DESC
            """

            self.conn.execute(token_length_view)

            # Document statistics view
            document_stats_view = """
            CREATE OR REPLACE VIEW document_stats AS
            SELECT
                doc_id,
                COUNT(*) as chunk_count,
                COUNT(DISTINCT page_no) as page_count,
                COUNT(DISTINCT element_label) as element_types,
                AVG(LENGTH(contextualized_text)) as avg_chunk_length,
                MIN(created_at) as first_created,
                MAX(created_at) as last_created
            FROM chunks
            GROUP BY doc_id
            ORDER BY chunk_count DESC
            """

            self.conn.execute(document_stats_view)

            logger.info("Analytics views created successfully")

        except Exception as e:
            logger.error(f"Failed to create analytics views: {e}")
            raise

    def add_chunk(self, chunk: ChunkRecord) -> None:
        """Add a chunk to the store.

        Args:
        ----
            chunk: Chunk record to add

        """
        start_time = time.perf_counter()

        try:
            # Validate chunk data
            if self.config.enable_validation:
                self._validate_chunk(chunk)

            # Insert chunk
            insert_sql = """
            INSERT INTO chunks (
                chunk_id, doc_id, doctags_sha, page_no, bbox, element_label,
                section_path, char_start, char_end, contextualized_text,
                content_only_text, table_payload, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            self.conn.execute(
                insert_sql,
                [
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.doctags_sha,
                    chunk.page_no,
                    str(chunk.bbox) if chunk.bbox else None,
                    chunk.element_label,
                    chunk.section_path,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.contextualized_text,
                    chunk.content_only_text,
                    str(chunk.table_payload) if chunk.table_payload else None,
                    chunk.created_at,
                ],
            )

            # Commit transaction
            self.conn.commit()

            processing_time = time.perf_counter() - start_time

            CHUNK_STORE_SECONDS.labels(operation="add_chunk", status="ok").observe(processing_time)
            CHUNK_STORE_OPERATIONS.labels(operation="add_chunk", status="ok").inc()
            CHUNK_STORE_SIZE_BYTES.labels(operation="add_chunk").observe(
                self.db_path.stat().st_size
            )

            logger.info(
                "Chunk added to store",
                extra={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "element_label": chunk.element_label,
                    "processing_time_seconds": processing_time,
                },
            )

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            CHUNK_STORE_SECONDS.labels(operation="add_chunk", status="error").observe(
                processing_time
            )
            CHUNK_STORE_OPERATIONS.labels(operation="add_chunk", status="error").inc()

            logger.error(
                "Failed to add chunk to store",
                extra={
                    "chunk_id": chunk.chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        """Get a chunk from the store.

        Args:
        ----
            chunk_id: Chunk identifier

        Returns:
        -------
            Chunk record if found, None otherwise

        """
        start_time = time.perf_counter()

        try:
            select_sql = "SELECT * FROM chunks WHERE chunk_id = ?"
            result = self.conn.execute(select_sql, [chunk_id]).fetchone()

            if result is None:
                processing_time = time.perf_counter() - start_time
                CHUNK_STORE_SECONDS.labels(operation="get_chunk", status="not_found").observe(
                    processing_time
                )
                CHUNK_STORE_OPERATIONS.labels(operation="get_chunk", status="not_found").inc()
                return None

            # Convert result to ChunkRecord
            chunk = ChunkRecord(
                chunk_id=result[0],
                doc_id=result[1],
                doctags_sha=result[2],
                page_no=result[3],
                bbox=eval(result[4]) if result[4] else {},
                element_label=result[5],
                section_path=result[6] or "",
                char_start=result[7] or 0,
                char_end=result[8] or 0,
                contextualized_text=result[9] or "",
                content_only_text=result[10] or "",
                table_payload=eval(result[11]) if result[11] else {},
                created_at=result[12],
            )

            processing_time = time.perf_counter() - start_time

            CHUNK_STORE_SECONDS.labels(operation="get_chunk", status="ok").observe(processing_time)
            CHUNK_STORE_OPERATIONS.labels(operation="get_chunk", status="ok").inc()

            return chunk

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            CHUNK_STORE_SECONDS.labels(operation="get_chunk", status="error").observe(
                processing_time
            )
            CHUNK_STORE_OPERATIONS.labels(operation="get_chunk", status="error").inc()

            logger.error(
                "Failed to get chunk from store",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def get_chunks_by_doc(self, doc_id: str) -> list[ChunkRecord]:
        """Get all chunks for a document.

        Args:
        ----
            doc_id: Document identifier

        Returns:
        -------
            List of chunk records

        """
        start_time = time.perf_counter()

        try:
            select_sql = "SELECT * FROM chunks WHERE doc_id = ? ORDER BY page_no, char_start"
            results = self.conn.execute(select_sql, [doc_id]).fetchall()

            chunks = []
            for result in results:
                chunk = ChunkRecord(
                    chunk_id=result[0],
                    doc_id=result[1],
                    doctags_sha=result[2],
                    page_no=result[3],
                    bbox=eval(result[4]) if result[4] else {},
                    element_label=result[5],
                    section_path=result[6] or "",
                    char_start=result[7] or 0,
                    char_end=result[8] or 0,
                    contextualized_text=result[9] or "",
                    content_only_text=result[10] or "",
                    table_payload=eval(result[11]) if result[11] else {},
                    created_at=result[12],
                )
                chunks.append(chunk)

            processing_time = time.perf_counter() - start_time

            CHUNK_STORE_SECONDS.labels(operation="get_chunks_by_doc", status="ok").observe(
                processing_time
            )
            CHUNK_STORE_OPERATIONS.labels(operation="get_chunks_by_doc", status="ok").inc()

            logger.info(
                "Retrieved chunks for document",
                extra={
                    "doc_id": doc_id,
                    "chunk_count": len(chunks),
                    "processing_time_seconds": processing_time,
                },
            )

            return chunks

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            CHUNK_STORE_SECONDS.labels(operation="get_chunks_by_doc", status="error").observe(
                processing_time
            )
            CHUNK_STORE_OPERATIONS.labels(operation="get_chunks_by_doc", status="error").inc()

            logger.error(
                "Failed to get chunks for document",
                extra={
                    "doc_id": doc_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from the store.

        Args:
        ----
            chunk_id: Chunk identifier

        Returns:
        -------
            True if chunk was removed, False if not found

        """
        start_time = time.perf_counter()

        try:
            delete_sql = "DELETE FROM chunks WHERE chunk_id = ?"
            result = self.conn.execute(delete_sql, [chunk_id])

            if result.rowcount == 0:
                processing_time = time.perf_counter() - start_time
                CHUNK_STORE_SECONDS.labels(operation="remove_chunk", status="not_found").observe(
                    processing_time
                )
                CHUNK_STORE_OPERATIONS.labels(operation="remove_chunk", status="not_found").inc()
                return False

            # Commit transaction
            self.conn.commit()

            processing_time = time.perf_counter() - start_time

            CHUNK_STORE_SECONDS.labels(operation="remove_chunk", status="ok").observe(
                processing_time
            )
            CHUNK_STORE_OPERATIONS.labels(operation="remove_chunk", status="ok").inc()

            logger.info(
                "Chunk removed from store",
                extra={
                    "chunk_id": chunk_id,
                    "processing_time_seconds": processing_time,
                },
            )

            return True

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            CHUNK_STORE_SECONDS.labels(operation="remove_chunk", status="error").observe(
                processing_time
            )
            CHUNK_STORE_OPERATIONS.labels(operation="remove_chunk", status="error").inc()

            logger.error(
                "Failed to remove chunk from store",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def get_store_stats(self) -> dict[str, Any]:
        """Get store statistics.

        Returns
        -------
            Dictionary with store statistics

        """
        try:
            # Get basic statistics
            stats_sql = """
            SELECT
                COUNT(*) as total_chunks,
                COUNT(DISTINCT doc_id) as total_documents,
                COUNT(DISTINCT element_label) as element_types,
                AVG(LENGTH(contextualized_text)) as avg_contextualized_length,
                AVG(LENGTH(content_only_text)) as avg_content_length,
                MIN(created_at) as first_created,
                MAX(created_at) as last_created
            FROM chunks
            """

            result = self.conn.execute(stats_sql).fetchone()

            stats = {
                "total_chunks": result[0] if result[0] else 0,
                "total_documents": result[1] if result[1] else 0,
                "element_types": result[2] if result[2] else 0,
                "avg_contextualized_length": float(result[3]) if result[3] else 0.0,
                "avg_content_length": float(result[4]) if result[4] else 0.0,
                "first_created": result[5],
                "last_created": result[6],
                "database_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
                "config": self.config.dict(),
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get store statistics: {e}")
            return {
                "error": str(e),
                "database_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            }

    def _validate_chunk(self, chunk: ChunkRecord) -> None:
        """Validate chunk data.

        Args:
        ----
            chunk: Chunk record to validate

        """
        errors = []

        # Check required fields
        if not chunk.chunk_id:
            errors.append("chunk_id is required")
        if not chunk.doc_id:
            errors.append("doc_id is required")
        if not chunk.doctags_sha:
            errors.append("doctags_sha is required")
        if not chunk.element_label:
            errors.append("element_label is required")

        # Check text length
        if len(chunk.contextualized_text) > self.config.max_text_length:
            errors.append(
                f"contextualized_text too long: {len(chunk.contextualized_text)} > {self.config.max_text_length}"
            )
        if len(chunk.content_only_text) > self.config.max_text_length:
            errors.append(
                f"content_only_text too long: {len(chunk.content_only_text)} > {self.config.max_text_length}"
            )

        # Check character positions
        if chunk.char_start < 0:
            errors.append("char_start must be non-negative")
        if chunk.char_end < 0:
            errors.append("char_end must be non-negative")
        if chunk.char_start > chunk.char_end:
            errors.append("char_start must be <= char_end")

        if errors:
            raise ValueError(f"Chunk validation failed: {', '.join(errors)}")

    def health_check(self) -> dict[str, Any]:
        """Check chunk store health.

        Returns
        -------
            Health status information

        """
        try:
            stats = self.get_store_stats()

            return {
                "status": "healthy",
                "total_chunks": stats.get("total_chunks", 0),
                "total_documents": stats.get("total_documents", 0),
                "database_size_bytes": stats.get("database_size_bytes", 0),
                "first_created": stats.get("first_created"),
                "last_created": stats.get("last_created"),
                "config": stats.get("config", {}),
            }

        except Exception as e:
            logger.error("Chunk store health check failed", extra={"error": str(e)})
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Chunk store database connection closed")

    def __enter__(self) -> "ChunkStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
