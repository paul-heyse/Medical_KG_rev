"""Pipeline stages for hybrid retrieval system.

This module implements discrete, restartable pipeline stages for processing
documents through the hybrid retrieval system.
"""

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Pipeline stage status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageResult(BaseModel):
    """Result of a pipeline stage execution."""

    stage_name: str = Field(..., description="Name of the stage")
    status: StageStatus = Field(..., description="Stage status")
    start_time: float = Field(..., description="Stage start time")
    end_time: float | None = Field(default=None, description="Stage end time")
    duration_seconds: float | None = Field(default=None, description="Stage duration in seconds")
    input_data: dict[str, Any] = Field(default_factory=dict, description="Input data for the stage")
    output_data: dict[str, Any] = Field(
        default_factory=dict, description="Output data from the stage"
    )
    error_message: str | None = Field(default=None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize pipeline stage.

        Args:
            name: Name of the stage
            config: Configuration for the stage

        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")

        self.logger.info(f"Initialized pipeline stage: {name}")

    @abstractmethod
    def execute(self, input_data: dict[str, Any]) -> StageResult:
        """Execute the pipeline stage.

        Args:
            input_data: Input data for the stage

        Returns:
            Stage execution result

        """
        pass

    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for the stage.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise

        """
        return True

    def cleanup(self) -> None:
        """Clean up stage resources."""
        self.logger.info(f"Cleaning up pipeline stage: {self.name}")


class ConvertStage(PipelineStage):
    """Convert stage for document processing."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize convert stage."""
        super().__init__("convert", config)

        # Stage-specific configuration
        self.enable_docling = self.config.get("enable_docling", True)
        self.output_format = self.config.get("output_format", "doctags")

        self.logger.info(
            "Initialized convert stage",
            extra={
                "enable_docling": self.enable_docling,
                "output_format": self.output_format,
            },
        )

    def execute(self, input_data: dict[str, Any]) -> StageResult:
        """Execute convert stage.

        Args:
            input_data: Input data containing document information

        Returns:
            Stage execution result

        """
        start_time = time.perf_counter()

        try:
            self.logger.info("Starting convert stage execution")

            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data for convert stage")

            # Extract input parameters
            document_id = input_data.get("document_id")
            pdf_path = input_data.get("pdf_path")
            docling_service_url = input_data.get("docling_service_url")

            if not document_id or not pdf_path:
                raise ValueError("Missing required input: document_id or pdf_path")

            # Process document
            result_data = self._process_document(document_id, pdf_path, docling_service_url)

            # Calculate duration
            end_time = time.perf_counter()
            duration = end_time - start_time

            self.logger.info(
                "Convert stage completed successfully",
                extra={
                    "document_id": document_id,
                    "duration_seconds": duration,
                    "output_keys": list(result_data.keys()),
                },
            )

            return StageResult(
                stage_name=self.name,
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_data=input_data,
                output_data=result_data,
                metadata={
                    "enable_docling": self.enable_docling,
                    "output_format": self.output_format,
                },
            )

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time

            self.logger.error(
                "Convert stage failed",
                extra={
                    "error": str(e),
                    "duration_seconds": duration,
                },
            )

            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_data=input_data,
                error_message=str(e),
            )

    def _process_document(
        self, document_id: str, pdf_path: str, docling_service_url: str | None = None
    ) -> dict[str, Any]:
        """Process document using Docling VLM.

        Args:
            document_id: Document identifier
            pdf_path: Path to PDF file
            docling_service_url: URL to Docling service

        Returns:
            Processing result data

        """
        try:
            if self.enable_docling and docling_service_url:
                # Use Docling VLM service
                result = self._process_with_docling(document_id, pdf_path, docling_service_url)
            else:
                raise ValueError("Docling VLM service URL required for processing")

            return result

        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            raise

    def _process_with_docling(
        self, document_id: str, pdf_path: str, service_url: str
    ) -> dict[str, Any]:
        """Process document with Docling VLM.

        Args:
            document_id: Document identifier
            pdf_path: Path to PDF file
            service_url: Docling service URL

        Returns:
            Processing result

        """
        # Import here to avoid circular imports
        from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMService

        # Initialize Docling service
        docling_service = DoclingVLMService(docker_service_url=service_url)

        # Process document
        result = docling_service.process_pdf(pdf_path, document_id)

        return {
            "document_id": document_id,
            "processing_method": "docling",
            "service_url": service_url,
            "result": result,
            "metadata": {
                "model_name": "google/gemma-3-12b-it",
                "processing_time": result.processing_time_seconds,
                "gpu_memory_used": result.gpu_memory_used_mb,
            },
        }


    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for convert stage.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise

        """
        required_fields = ["document_id", "pdf_path"]

        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"Missing required field: {field}")
                return False

        # Check if PDF file exists
        pdf_path = input_data.get("pdf_path")
        if pdf_path and not Path(pdf_path).exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return False

        return True


class ChunkStage(PipelineStage):
    """Chunk stage for document segmentation."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize chunk stage."""
        super().__init__("chunk", config)

        # Stage-specific configuration
        self.chunker_type = self.config.get("chunker_type", "hybrid")
        self.max_tokens = self.config.get("max_tokens", 512)
        self.overlap_tokens = self.config.get("overlap_tokens", 50)
        self.enable_tokenizer_alignment = self.config.get("enable_tokenizer_alignment", True)

        self.logger.info(
            "Initialized chunk stage",
            extra={
                "chunker_type": self.chunker_type,
                "max_tokens": self.max_tokens,
                "overlap_tokens": self.overlap_tokens,
                "enable_tokenizer_alignment": self.enable_tokenizer_alignment,
            },
        )

    def execute(self, input_data: dict[str, Any]) -> StageResult:
        """Execute chunk stage.

        Args:
            input_data: Input data from convert stage

        Returns:
            Stage execution result

        """
        start_time = time.perf_counter()

        try:
            self.logger.info("Starting chunk stage execution")

            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data for chunk stage")

            # Extract input parameters
            document_id = input_data.get("document_id")
            processing_result = input_data.get("result")

            if not document_id or not processing_result:
                raise ValueError("Missing required input: document_id or result")

            # Process chunks
            result_data = self._process_chunks(document_id, processing_result)

            # Calculate duration
            end_time = time.perf_counter()
            duration = end_time - start_time

            self.logger.info(
                "Chunk stage completed successfully",
                extra={
                    "document_id": document_id,
                    "duration_seconds": duration,
                    "chunk_count": len(result_data.get("chunks", [])),
                },
            )

            return StageResult(
                stage_name=self.name,
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_data=input_data,
                output_data=result_data,
                metadata={
                    "chunker_type": self.chunker_type,
                    "max_tokens": self.max_tokens,
                    "overlap_tokens": self.overlap_tokens,
                    "enable_tokenizer_alignment": self.enable_tokenizer_alignment,
                },
            )

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time

            self.logger.error(
                "Chunk stage failed",
                extra={
                    "error": str(e),
                    "duration_seconds": duration,
                },
            )

            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_data=input_data,
                error_message=str(e),
            )

    def _process_chunks(self, document_id: str, processing_result: Any) -> dict[str, Any]:
        """Process document chunks.

        Args:
            document_id: Document identifier
            processing_result: Result from convert stage

        Returns:
            Chunk processing result

        """
        try:
            # Import here to avoid circular imports
            from Medical_KG_rev.chunking.registry import get_chunker

            # Get chunker
            chunker = get_chunker(self.chunker_type)

            # Extract document from processing result
            if hasattr(processing_result, "document"):
                document = processing_result.document
            else:
                document = processing_result

            # Create chunks
            chunks = chunker.chunk(document)

            # Process chunks for tokenizer alignment if enabled
            if self.enable_tokenizer_alignment and self.chunker_type == "hybrid":
                chunks = self._align_chunks_with_tokenizer(chunks)

            return {
                "document_id": document_id,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "chunker_type": self.chunker_type,
                "metadata": {
                    "max_tokens": self.max_tokens,
                    "overlap_tokens": self.overlap_tokens,
                    "tokenizer_alignment": self.enable_tokenizer_alignment,
                },
            }

        except Exception as e:
            self.logger.error(f"Chunk processing failed: {e}")
            raise

    def _align_chunks_with_tokenizer(self, chunks: list[Any]) -> list[Any]:
        """Align chunks with tokenizer for SPLADE compatibility.

        Args:
            chunks: List of chunks to align

        Returns:
            Aligned chunks

        """
        try:
            # Import here to avoid circular imports
            from Medical_KG_rev.chunking.hybrid_chunker import HybridChunker

            # Use hybrid chunker for tokenizer alignment
            hybrid_chunker = HybridChunker()

            aligned_chunks = []
            for chunk in chunks:
                aligned_chunk = hybrid_chunker._segment_chunk_for_splade(chunk, self.max_tokens)
                aligned_chunks.extend(aligned_chunk)

            return aligned_chunks

        except Exception as e:
            self.logger.warning(f"Tokenizer alignment failed: {e}")
            return chunks

    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for chunk stage.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise

        """
        required_fields = ["document_id", "result"]

        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"Missing required field: {field}")
                return False

        return True


class FeaturesStage(PipelineStage):
    """Features stage for extracting retrieval features."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize features stage."""
        super().__init__("features", config)

        # Stage-specific configuration
        self.enable_bm25 = self.config.get("enable_bm25", True)
        self.enable_splade = self.config.get("enable_splade", True)
        self.enable_qwen3 = self.config.get("enable_qwen3", True)
        self.batch_size = self.config.get("batch_size", 32)

        self.logger.info(
            "Initialized features stage",
            extra={
                "enable_bm25": self.enable_bm25,
                "enable_splade": self.enable_splade,
                "enable_qwen3": self.enable_qwen3,
                "batch_size": self.batch_size,
            },
        )

    def execute(self, input_data: dict[str, Any]) -> StageResult:
        """Execute features stage.

        Args:
            input_data: Input data from chunk stage

        Returns:
            Stage execution result

        """
        start_time = time.perf_counter()

        try:
            self.logger.info("Starting features stage execution")

            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data for features stage")

            # Extract input parameters
            document_id = input_data.get("document_id")
            chunks = input_data.get("chunks", [])

            if not document_id or not chunks:
                raise ValueError("Missing required input: document_id or chunks")

            # Process features
            result_data = self._process_features(document_id, chunks)

            # Calculate duration
            end_time = time.perf_counter()
            duration = end_time - start_time

            self.logger.info(
                "Features stage completed successfully",
                extra={
                    "document_id": document_id,
                    "duration_seconds": duration,
                    "chunk_count": len(chunks),
                },
            )

            return StageResult(
                stage_name=self.name,
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_data=input_data,
                output_data=result_data,
                metadata={
                    "enable_bm25": self.enable_bm25,
                    "enable_splade": self.enable_splade,
                    "enable_qwen3": self.enable_qwen3,
                    "batch_size": self.batch_size,
                },
            )

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time

            self.logger.error(
                "Features stage failed",
                extra={
                    "error": str(e),
                    "duration_seconds": duration,
                },
            )

            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_data=input_data,
                error_message=str(e),
            )

    def _process_features(self, document_id: str, chunks: list[Any]) -> dict[str, Any]:
        """Process features for chunks.

        Args:
            document_id: Document identifier
            chunks: List of chunks to process

        Returns:
            Features processing result

        """
        try:
            result_data = {
                "document_id": document_id,
                "chunks": chunks,
                "features": {},
                "metadata": {},
            }

            # Process BM25 features
            if self.enable_bm25:
                bm25_features = self._process_bm25_features(chunks)
                result_data["features"]["bm25"] = bm25_features
                result_data["metadata"]["bm25_enabled"] = True

            # Process SPLADE features
            if self.enable_splade:
                splade_features = self._process_splade_features(chunks)
                result_data["features"]["splade"] = splade_features
                result_data["metadata"]["splade_enabled"] = True

            # Process Qwen3 features
            if self.enable_qwen3:
                qwen3_features = self._process_qwen3_features(chunks)
                result_data["features"]["qwen3"] = qwen3_features
                result_data["metadata"]["qwen3_enabled"] = True

            return result_data

        except Exception as e:
            self.logger.error(f"Features processing failed: {e}")
            raise

    def _process_bm25_features(self, chunks: list[Any]) -> dict[str, Any]:
        """Process BM25 features for chunks.

        Args:
            chunks: List of chunks to process

        Returns:
            BM25 features

        """
        try:
            # Import here to avoid circular imports
            from Medical_KG_rev.services.retrieval.bm25_service import BM25Service

            # Initialize BM25 service
            bm25_service = BM25Service()

            # Process chunks in batches
            features = []
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]
                batch_features = bm25_service.process_batch(batch)
                features.extend(batch_features)

            return {
                "features": features,
                "chunk_count": len(chunks),
                "batch_size": self.batch_size,
            }

        except Exception as e:
            self.logger.error(f"BM25 features processing failed: {e}")
            raise

    def _process_splade_features(self, chunks: list[Any]) -> dict[str, Any]:
        """Process SPLADE features for chunks.

        Args:
            chunks: List of chunks to process

        Returns:
            SPLADE features

        """
        try:
            # Import here to avoid circular imports
            from Medical_KG_rev.services.retrieval.splade_service import SPLADEService

            # Initialize SPLADE service
            splade_service = SPLADEService()

            # Process chunks in batches
            features = []
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]
                batch_features = splade_service.process_batch(batch)
                features.extend(batch_features)

            return {
                "features": features,
                "chunk_count": len(chunks),
                "batch_size": self.batch_size,
            }

        except Exception as e:
            self.logger.error(f"SPLADE features processing failed: {e}")
            raise

    def _process_qwen3_features(self, chunks: list[Any]) -> dict[str, Any]:
        """Process Qwen3 features for chunks.

        Args:
            chunks: List of chunks to process

        Returns:
            Qwen3 features

        """
        try:
            # Import here to avoid circular imports
            from Medical_KG_rev.services.retrieval.qwen3_service import Qwen3Service

            # Initialize Qwen3 service
            qwen3_service = Qwen3Service()

            # Process chunks in batches
            features = []
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]
                batch_features = qwen3_service.process_batch(batch)
                features.extend(batch_features)

            return {
                "features": features,
                "chunk_count": len(chunks),
                "batch_size": self.batch_size,
            }

        except Exception as e:
            self.logger.error(f"Qwen3 features processing failed: {e}")
            raise

    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for features stage.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise

        """
        required_fields = ["document_id", "chunks"]

        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"Missing required field: {field}")
                return False

        return True


class IndexStage(PipelineStage):
    """Index stage for storing processed data."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize index stage."""
        super().__init__("index", config)

        # Stage-specific configuration
        self.enable_chunk_store = self.config.get("enable_chunk_store", True)
        self.enable_bm25_index = self.config.get("enable_bm25_index", True)
        self.enable_splade_index = self.config.get("enable_splade_index", True)
        self.enable_qwen3_index = self.config.get("enable_qwen3_index", True)
        self.batch_size = self.config.get("batch_size", 100)

        self.logger.info(
            "Initialized index stage",
            extra={
                "enable_chunk_store": self.enable_chunk_store,
                "enable_bm25_index": self.enable_bm25_index,
                "enable_splade_index": self.enable_splade_index,
                "enable_qwen3_index": self.enable_qwen3_index,
                "batch_size": self.batch_size,
            },
        )

    def execute(self, input_data: dict[str, Any]) -> StageResult:
        """Execute index stage.

        Args:
            input_data: Input data from features stage

        Returns:
            Stage execution result

        """
        start_time = time.perf_counter()

        try:
            self.logger.info("Starting index stage execution")

            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data for index stage")

            # Extract input parameters
            document_id = input_data.get("document_id")
            chunks = input_data.get("chunks", [])
            features = input_data.get("features", {})

            if not document_id or not chunks:
                raise ValueError("Missing required input: document_id or chunks")

            # Process indexing
            result_data = self._process_indexing(document_id, chunks, features)

            # Calculate duration
            end_time = time.perf_counter()
            duration = end_time - start_time

            self.logger.info(
                "Index stage completed successfully",
                extra={
                    "document_id": document_id,
                    "duration_seconds": duration,
                    "chunk_count": len(chunks),
                },
            )

            return StageResult(
                stage_name=self.name,
                status=StageStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_data=input_data,
                output_data=result_data,
                metadata={
                    "enable_chunk_store": self.enable_chunk_store,
                    "enable_bm25_index": self.enable_bm25_index,
                    "enable_splade_index": self.enable_splade_index,
                    "enable_qwen3_index": self.enable_qwen3_index,
                    "batch_size": self.batch_size,
                },
            )

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time

            self.logger.error(
                "Index stage failed",
                extra={
                    "error": str(e),
                    "duration_seconds": duration,
                },
            )

            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_data=input_data,
                error_message=str(e),
            )

    def _process_indexing(
        self, document_id: str, chunks: list[Any], features: dict[str, Any]
    ) -> dict[str, Any]:
        """Process indexing for chunks and features.

        Args:
            document_id: Document identifier
            chunks: List of chunks to index
            features: Features to index

        Returns:
            Indexing result

        """
        try:
            result_data = {
                "document_id": document_id,
                "chunks": chunks,
                "features": features,
                "indexing_results": {},
                "metadata": {},
            }

            # Index chunks
            if self.enable_chunk_store:
                chunk_store_result = self._index_chunks(document_id, chunks)
                result_data["indexing_results"]["chunk_store"] = chunk_store_result
                result_data["metadata"]["chunk_store_enabled"] = True

            # Index BM25 features
            if self.enable_bm25_index and "bm25" in features:
                bm25_index_result = self._index_bm25_features(document_id, chunks, features["bm25"])
                result_data["indexing_results"]["bm25_index"] = bm25_index_result
                result_data["metadata"]["bm25_index_enabled"] = True

            # Index SPLADE features
            if self.enable_splade_index and "splade" in features:
                splade_index_result = self._index_splade_features(
                    document_id, chunks, features["splade"]
                )
                result_data["indexing_results"]["splade_index"] = splade_index_result
                result_data["metadata"]["splade_index_enabled"] = True

            # Index Qwen3 features
            if self.enable_qwen3_index and "qwen3" in features:
                qwen3_index_result = self._index_qwen3_features(
                    document_id, chunks, features["qwen3"]
                )
                result_data["indexing_results"]["qwen3_index"] = qwen3_index_result
                result_data["metadata"]["qwen3_index_enabled"] = True

            return result_data

        except Exception as e:
            self.logger.error(f"Indexing processing failed: {e}")
            raise

    def _index_chunks(self, document_id: str, chunks: list[Any]) -> dict[str, Any]:
        """Index chunks in chunk store.

        Args:
            document_id: Document identifier
            chunks: List of chunks to index

        Returns:
            Chunk indexing result

        """
        try:
            # Import here to avoid circular imports
            from Medical_KG_rev.storage.chunk_store import ChunkRecord, ChunkStore

            # Initialize chunk store
            chunk_store = ChunkStore()

            # Process chunks in batches
            indexed_count = 0
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]

                for chunk in batch:
                    # Create chunk record
                    chunk_record = ChunkRecord(
                        chunk_id=chunk.chunk_id,
                        doc_id=document_id,
                        doctags_sha=chunk.doctags_sha,
                        page_no=chunk.page_no,
                        bbox=chunk.bbox,
                        element_label=chunk.element_label,
                        section_path=chunk.section_path,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        contextualized_text=chunk.contextualized_text,
                        content_only_text=chunk.content_only_text,
                        table_payload=chunk.table_payload,
                    )

                    # Add to chunk store
                    chunk_store.add_chunk(chunk_record)
                    indexed_count += 1

            return {
                "indexed_count": indexed_count,
                "total_chunks": len(chunks),
                "batch_size": self.batch_size,
            }

        except Exception as e:
            self.logger.error(f"Chunk indexing failed: {e}")
            raise

    def _index_bm25_features(
        self, document_id: str, chunks: list[Any], bm25_features: dict[str, Any]
    ) -> dict[str, Any]:
        """Index BM25 features.

        Args:
            document_id: Document identifier
            chunks: List of chunks
            bm25_features: BM25 features to index

        Returns:
            BM25 indexing result

        """
        try:
            # Import here to avoid circular imports
            from Medical_KG_rev.services.vector_store.stores.bm25_index import BM25Index

            # Initialize BM25 index
            bm25_index = BM25Index()

            # Process chunks in batches
            indexed_count = 0
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]

                for chunk in batch:
                    # Add document to BM25 index
                    bm25_index.add_document(chunk.chunk_id, chunk.contextualized_text)
                    indexed_count += 1

            return {
                "indexed_count": indexed_count,
                "total_chunks": len(chunks),
                "batch_size": self.batch_size,
            }

        except Exception as e:
            self.logger.error(f"BM25 indexing failed: {e}")
            raise

    def _index_splade_features(
        self, document_id: str, chunks: list[Any], splade_features: dict[str, Any]
    ) -> dict[str, Any]:
        """Index SPLADE features.

        Args:
            document_id: Document identifier
            chunks: List of chunks
            splade_features: SPLADE features to index

        Returns:
            SPLADE indexing result

        """
        try:
            # Import here to avoid circular imports
            from Medical_KG_rev.services.vector_store.stores.splade_index import SPLADEImpactIndex

            # Initialize SPLADE index
            splade_index = SPLADEImpactIndex()

            # Process chunks in batches
            indexed_count = 0
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]

                for chunk in batch:
                    # Add vector to SPLADE index
                    splade_index.add_vector(
                        chunk.chunk_id, splade_features["features"][indexed_count]
                    )
                    indexed_count += 1

            return {
                "indexed_count": indexed_count,
                "total_chunks": len(chunks),
                "batch_size": self.batch_size,
            }

        except Exception as e:
            self.logger.error(f"SPLADE indexing failed: {e}")
            raise

    def _index_qwen3_features(
        self, document_id: str, chunks: list[Any], qwen3_features: dict[str, Any]
    ) -> dict[str, Any]:
        """Index Qwen3 features.

        Args:
            document_id: Document identifier
            chunks: List of chunks
            qwen3_features: Qwen3 features to index

        Returns:
            Qwen3 indexing result

        """
        try:
            # Import here to avoid circular imports
            from Medical_KG_rev.services.vector_store.stores.qwen3_index import Qwen3Index

            # Initialize Qwen3 index
            qwen3_index = Qwen3Index()

            # Process chunks in batches
            indexed_count = 0
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]

                for chunk in batch:
                    # Add vector to Qwen3 index
                    qwen3_index.add_vector(
                        chunk.chunk_id, qwen3_features["features"][indexed_count]
                    )
                    indexed_count += 1

            return {
                "indexed_count": indexed_count,
                "total_chunks": len(chunks),
                "batch_size": self.batch_size,
            }

        except Exception as e:
            self.logger.error(f"Qwen3 indexing failed: {e}")
            raise

    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for index stage.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise

        """
        required_fields = ["document_id", "chunks"]

        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"Missing required field: {field}")
                return False

        return True
