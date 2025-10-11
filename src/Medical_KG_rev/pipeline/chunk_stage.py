"""Chunk stage implementation for document segmentation.

This module implements the Chunk stage for segmenting documents into
retrieval-optimized chunks using hybrid chunking strategies.
"""

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from Medical_KG_rev.pipeline.stages import ChunkStage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class ChunkStageConfig(BaseModel):
    """Configuration for Chunk stage."""

    chunker_type: str = Field(default="hybrid", description="Type of chunker to use")
    max_tokens: int = Field(default=512, description="Maximum tokens per chunk")
    overlap_tokens: int = Field(default=50, description="Overlap tokens between chunks")
    enable_tokenizer_alignment: bool = Field(default=True, description="Enable tokenizer alignment")
    tokenizer_name: str = Field(
        default="naver/splade-v3", description="Tokenizer name for alignment"
    )
    enable_hierarchical_chunking: bool = Field(
        default=True, description="Enable hierarchical chunking"
    )
    preserve_structure: bool = Field(default=True, description="Preserve document structure")
    enable_metadata_extraction: bool = Field(default=True, description="Enable metadata extraction")
    batch_size: int = Field(default=32, description="Batch size for processing")
    max_chunk_size_bytes: int = Field(default=10000, description="Maximum chunk size in bytes")


class ChunkStageImpl(ChunkStage):
    """Enhanced Chunk stage implementation."""

    def __init__(self, config: ChunkStageConfig | None = None):
        """Initialize Chunk stage implementation.

        Args:
            config: Configuration for the stage

        """
        self.stage_config = config or ChunkStageConfig()

        # Convert to dict for base class
        base_config = self.stage_config.dict()
        super().__init__(base_config)

        # Stage-specific attributes
        self.tokenizer_name = self.stage_config.tokenizer_name
        self.enable_hierarchical_chunking = self.stage_config.enable_hierarchical_chunking
        self.preserve_structure = self.stage_config.preserve_structure
        self.enable_metadata_extraction = self.stage_config.enable_metadata_extraction
        self.batch_size = self.stage_config.batch_size
        self.max_chunk_size_bytes = self.stage_config.max_chunk_size_bytes

        self.logger.info(
            "Initialized Chunk stage implementation",
            extra={
                "chunker_type": self.chunker_type,
                "max_tokens": self.max_tokens,
                "overlap_tokens": self.overlap_tokens,
                "enable_tokenizer_alignment": self.enable_tokenizer_alignment,
                "tokenizer_name": self.tokenizer_name,
                "enable_hierarchical_chunking": self.enable_hierarchical_chunking,
                "preserve_structure": self.preserve_structure,
                "enable_metadata_extraction": self.enable_metadata_extraction,
                "batch_size": self.batch_size,
                "max_chunk_size_bytes": self.max_chunk_size_bytes,
            },
        )

    def execute(self, input_data: dict[str, Any]) -> StageResult:
        """Execute Chunk stage with enhanced functionality.

        Args:
            input_data: Input data from convert stage

        Returns:
            Stage execution result

        """
        start_time = time.perf_counter()

        try:
            self.logger.info("Starting Chunk stage execution")

            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data for Chunk stage")

            # Extract input parameters
            document_id = input_data.get("document_id")
            processing_result = input_data.get("result")

            if not document_id or not processing_result:
                raise ValueError("Missing required input: document_id or result")

            # Process chunks
            result_data = self._process_chunks_enhanced(document_id, processing_result)

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
                    "tokenizer_name": self.tokenizer_name,
                    "enable_hierarchical_chunking": self.enable_hierarchical_chunking,
                    "preserve_structure": self.preserve_structure,
                    "enable_metadata_extraction": self.enable_metadata_extraction,
                    "batch_size": self.batch_size,
                    "max_chunk_size_bytes": self.max_chunk_size_bytes,
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

    def _process_chunks_enhanced(self, document_id: str, processing_result: Any) -> dict[str, Any]:
        """Process document chunks with enhanced functionality.

        Args:
            document_id: Document identifier
            processing_result: Result from convert stage

        Returns:
            Chunk processing result

        """
        try:
            # Import here to avoid circular imports
            from Medical_KG_rev.chunking.factory import ChunkerFactory
            from Medical_KG_rev.chunking.models import ChunkerConfig

            # Get chunker
            factory = ChunkerFactory()
            config = ChunkerConfig(name=self.chunker_type, params={})
            registered_chunker = factory.create(config)
            chunker = registered_chunker.instance

            # Extract document from processing result
            if hasattr(processing_result, "document"):
                document = processing_result.document
            else:
                document = processing_result

            # Create chunks
            chunks = chunker.chunk(document, tenant_id="default")

            # Enhance chunks with metadata
            if self.enable_metadata_extraction:
                chunks = self._extract_chunk_metadata(chunks, document_id)

            # Process chunks for tokenizer alignment if enabled
            # Note: Tokenizer alignment is handled internally by the hybrid chunker
            # if self.enable_tokenizer_alignment and self.chunker_type == "hybrid":
            #     chunks = self._align_chunks_with_tokenizer(chunks)

            # Validate chunk sizes
            chunks = self._validate_chunk_sizes(chunks)

            # Sort chunks by page and position
            chunks = self._sort_chunks(chunks)

            return {
                "document_id": document_id,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "chunker_type": self.chunker_type,
                "metadata": {
                    "max_tokens": self.max_tokens,
                    "overlap_tokens": self.overlap_tokens,
                    "tokenizer_alignment": self.enable_tokenizer_alignment,
                    "tokenizer_name": self.tokenizer_name,
                    "hierarchical_chunking": self.enable_hierarchical_chunking,
                    "structure_preserved": self.preserve_structure,
                    "metadata_extracted": self.enable_metadata_extraction,
                    "batch_size": self.batch_size,
                    "max_chunk_size_bytes": self.max_chunk_size_bytes,
                },
            }

        except Exception as e:
            self.logger.error(f"Enhanced chunk processing failed: {e}")
            raise

    def _extract_chunk_metadata(self, chunks: list[Any], document_id: str) -> list[Any]:
        """Extract metadata for chunks.

        Args:
            chunks: List of chunks to enhance
            document_id: Document identifier

        Returns:
            Enhanced chunks with metadata

        """
        try:
            enhanced_chunks = []

            for i, chunk in enumerate(chunks):
                # Add chunk metadata
                if hasattr(chunk, "metadata"):
                    chunk.metadata = chunk.metadata or {}
                else:
                    chunk.metadata = {}

                # Add chunk-specific metadata
                chunk.metadata.update(
                    {
                        "chunk_index": i,
                        "document_id": document_id,
                        "chunker_type": self.chunker_type,
                        "max_tokens": self.max_tokens,
                        "overlap_tokens": self.overlap_tokens,
                        "tokenizer_alignment": self.enable_tokenizer_alignment,
                        "hierarchical_chunking": self.enable_hierarchical_chunking,
                        "structure_preserved": self.preserve_structure,
                    }
                )

                # Add text statistics
                if hasattr(chunk, "contextualized_text"):
                    chunk.metadata["text_length"] = len(chunk.contextualized_text)
                    chunk.metadata["word_count"] = len(chunk.contextualized_text.split())

                # Add position metadata
                if hasattr(chunk, "page_no"):
                    chunk.metadata["page_number"] = chunk.page_no

                if hasattr(chunk, "char_start") and hasattr(chunk, "char_end"):
                    chunk.metadata["char_span"] = {
                        "start": chunk.char_start,
                        "end": chunk.char_end,
                        "length": chunk.char_end - chunk.char_start,
                    }

                enhanced_chunks.append(chunk)

            return enhanced_chunks

        except Exception as e:
            self.logger.warning(f"Metadata extraction failed: {e}")
            return chunks

    def _validate_chunk_sizes(self, chunks: list[Any]) -> list[Any]:
        """Validate and filter chunks by size.

        Args:
            chunks: List of chunks to validate

        Returns:
            Validated chunks

        """
        try:
            validated_chunks = []

            for chunk in chunks:
                # Check chunk size
                if hasattr(chunk, "contextualized_text"):
                    text_size = len(chunk.contextualized_text.encode("utf-8"))

                    if text_size > self.max_chunk_size_bytes:
                        self.logger.warning(
                            f"Chunk size exceeds limit: {text_size} > {self.max_chunk_size_bytes} bytes",
                            extra={"chunk_id": getattr(chunk, "chunk_id", "unknown")},
                        )
                        continue

                validated_chunks.append(chunk)

            return validated_chunks

        except Exception as e:
            self.logger.warning(f"Chunk size validation failed: {e}")
            return chunks

    def _sort_chunks(self, chunks: list[Any]) -> list[Any]:
        """Sort chunks by page and position.

        Args:
            chunks: List of chunks to sort

        Returns:
            Sorted chunks

        """
        try:

            def chunk_sort_key(chunk: Any) -> tuple[int, int]:
                # Sort by page number first, then by character position
                page_no = getattr(chunk, "page_no", 0)
                char_start = getattr(chunk, "char_start", 0)
                return (page_no, char_start)

            return sorted(chunks, key=chunk_sort_key)

        except Exception as e:
            self.logger.warning(f"Chunk sorting failed: {e}")
            return chunks

    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for Chunk stage.

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

        # Validate chunker type
        if self.chunker_type not in ["hybrid", "semantic", "fixed", "sliding"]:
            self.logger.error(f"Invalid chunker type: {self.chunker_type}")
            return False

        # Validate token limits
        if self.max_tokens <= 0:
            self.logger.error(f"Invalid max_tokens: {self.max_tokens}")
            return False

        if self.overlap_tokens < 0 or self.overlap_tokens >= self.max_tokens:
            self.logger.error(f"Invalid overlap_tokens: {self.overlap_tokens}")
            return False

        return True

    def get_stage_info(self) -> dict[str, Any]:
        """Get stage information.

        Returns:
            Stage information

        """
        return {
            "stage_name": self.name,
            "config": self.stage_config.dict(),
            "capabilities": {
                "chunker_types": ["hybrid", "semantic", "fixed", "sliding"],
                "tokenizer_alignment": self.enable_tokenizer_alignment,
                "hierarchical_chunking": self.enable_hierarchical_chunking,
                "structure_preservation": self.preserve_structure,
                "metadata_extraction": self.enable_metadata_extraction,
            },
            "limits": {
                "max_tokens": self.max_tokens,
                "overlap_tokens": self.overlap_tokens,
                "max_chunk_size_bytes": self.max_chunk_size_bytes,
                "batch_size": self.batch_size,
            },
        }

    def health_check(self) -> dict[str, Any]:
        """Check stage health.

        Returns:
            Health status information

        """
        try:
            health = {
                "status": "healthy",
                "stage_name": self.name,
                "config": self.stage_config.dict(),
                "capabilities": {
                    "chunker_types": ["hybrid", "semantic", "fixed", "sliding"],
                    "tokenizer_alignment": self.enable_tokenizer_alignment,
                    "hierarchical_chunking": self.enable_hierarchical_chunking,
                    "structure_preservation": self.preserve_structure,
                    "metadata_extraction": self.enable_metadata_extraction,
                },
                "limits": {
                    "max_tokens": self.max_tokens,
                    "overlap_tokens": self.overlap_tokens,
                    "max_chunk_size_bytes": self.max_chunk_size_bytes,
                    "batch_size": self.batch_size,
                },
            }

            # Check if tokenizer is available
            if self.enable_tokenizer_alignment:
                try:
                    from transformers import AutoTokenizer

                    AutoTokenizer.from_pretrained(self.tokenizer_name)  # type: ignore
                    health["tokenizer_status"] = "healthy"
                    health["tokenizer_name"] = self.tokenizer_name
                except Exception as e:
                    health["tokenizer_status"] = "unhealthy"
                    health["tokenizer_error"] = str(e)

            return health

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }
