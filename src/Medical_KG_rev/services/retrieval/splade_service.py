"""SPLADE-v3 service for learned sparse retrieval with Rep-Max aggregation.

This module implements SPLADE-v3 (Sparse Lexical and Expansion) retrieval using
the naver/splade-v3 model with Rep-Max aggregation for efficient chunk-level
sparse vector representation.
"""

import logging
import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from Medical_KG_rev.config.settings import get_settings
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
SPLADE_PROCESSING_SECONDS = Histogram(
    "splade_processing_seconds",
    "Time spent processing SPLADE operations",
    ["operation", "status"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

SPLADE_MODEL_LOAD_SECONDS = Histogram(
    "splade_model_load_seconds",
    "Time spent loading SPLADE model",
    ["status"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

SPLADE_GPU_MEMORY_MB = Histogram(
    "splade_gpu_memory_mb",
    "GPU memory usage for SPLADE operations",
    ["operation"],
    buckets=[100, 500, 1000, 2000, 4000, 8000, 16000]
)

SPLADE_RETRIES_TOTAL = Counter(
    "splade_retries_total",
    "Total number of SPLADE operation retries",
    ["operation", "reason"]
)

SPLADE_OPERATIONS_TOTAL = Counter(
    "splade_operations_total",
    "Total number of SPLADE operations",
    ["operation", "status"]
)


class SPLADESegment(BaseModel):
    """A segment of text for SPLADE processing."""

    text: str = Field(..., description="Segment text content")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    token_count: int = Field(..., description="Number of tokens in segment")
    segment_id: str = Field(..., description="Unique segment identifier")


class SPLADEVector(BaseModel):
    """SPLADE sparse vector representation."""

    terms: dict[int, float] = Field(..., description="Term ID to weight mapping")
    tokenizer_name: str = Field(..., description="Tokenizer used for encoding")
    model_name: str = Field(..., description="SPLADE model name")
    sparsity_threshold: float = Field(default=0.01, description="Minimum weight threshold")
    quantization_scale: int = Field(default=1000, description="Quantization scale factor")


class SPLADEResult(BaseModel):
    """Result from SPLADE processing."""

    chunk_id: str = Field(..., description="Chunk identifier")
    segments: list[SPLADESegment] = Field(..., description="Processed segments")
    sparse_vector: SPLADEVector = Field(..., description="Aggregated sparse vector")
    processing_time_seconds: float = Field(..., description="Processing time")
    gpu_memory_used_mb: int = Field(..., description="GPU memory usage")


class SPLADEProcessingError(Exception):
    """Exception raised during SPLADE processing."""
    pass


class SPLADEModelLoadError(Exception):
    """Exception raised when SPLADE model fails to load."""
    pass


class SPLADEService:
    """SPLADE-v3 service for learned sparse retrieval.

    This service implements SPLADE-v3 with Rep-Max aggregation for efficient
    chunk-level sparse vector representation. It uses the naver/splade-v3 model
    with transformers pipeline for consistent tokenization and encoding.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-v3",
        max_tokens: int = 512,
        sparsity_threshold: float = 0.01,
        quantization_scale: int = 1000,
        gpu_manager: Any = None,
    ):
        """Initialize SPLADE service.

        Args:
            model_name: Name of the SPLADE model to use
            max_tokens: Maximum tokens per segment
            sparsity_threshold: Minimum weight threshold for sparsity control
            quantization_scale: Scale factor for weight quantization
            gpu_manager: GPU manager for resource allocation
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.sparsity_threshold = sparsity_threshold
        self.quantization_scale = quantization_scale
        self.gpu_manager = gpu_manager

        # Model components (loaded lazily)
        self._tokenizer = None
        self._model = None
        self._pipeline = None
        self._is_loaded = False

        # Settings
        self._settings = get_settings()

        logger.info(
            "Initialized SPLADE service",
            extra={
                "model_name": self.model_name,
                "max_tokens": self.max_tokens,
                "sparsity_threshold": self.sparsity_threshold,
                "quantization_scale": self.quantization_scale,
            }
        )

    def _load_model(self) -> None:
        """Load SPLADE model and tokenizer."""
        if self._is_loaded:
            return

        start_time = time.perf_counter()

        try:
            # Check GPU availability if gpu_manager is provided
            if self.gpu_manager and hasattr(self.gpu_manager, 'is_gpu_available'):
                if not self.gpu_manager.is_gpu_available():
                    raise SPLADEModelLoadError("GPU not available for SPLADE model")

            # Import transformers components
            try:
# import torch  # type: ignore  # Removed for torch isolation
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ImportError as e:
                raise SPLADEModelLoadError(f"Transformers not available: {e}")

            # Load tokenizer
            logger.info("Loading SPLADE tokenizer", extra={"model_name": self.model_name})
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model
            logger.info("Loading SPLADE model", extra={"model_name": self.model_name})
            self._model = AutoModel.from_pretrained(self.model_name)

            # Move to GPU if available
            # GPU functionality moved to gRPC services
            logger.info("SPLADE model loaded (GPU operations handled by gRPC services)")

            # Set model to evaluation mode
            self._model.eval()

            # Create pipeline for consistent processing
            self._pipeline = {
                "tokenizer": self._tokenizer,
                "model": self._model,
            }

            self._is_loaded = True

            load_time = time.perf_counter() - start_time
            SPLADE_MODEL_LOAD_SECONDS.labels(status="ok").observe(load_time)

            logger.info(
                "SPLADE model loaded successfully",
                extra={
                    "model_name": self.model_name,
                    "load_time_seconds": load_time,
                    "gpu_available": False  # GPU functionality moved to gRPC services,
                }
            )

        except Exception as e:
            load_time = time.perf_counter() - start_time
            SPLADE_MODEL_LOAD_SECONDS.labels(status="error").observe(load_time)

            logger.error(
                "Failed to load SPLADE model",
                extra={
                    "model_name": self.model_name,
                    "load_time_seconds": load_time,
                    "error": str(e),
                }
            )
            raise SPLADEModelLoadError(f"Failed to load SPLADE model: {e}")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using SPLADE tokenizer."""
        if not self._is_loaded:
            self._load_model()

        try:
            tokens = self._tokenizer.encode(text, add_special_tokens=True)  # type: ignore
            return len(tokens)
        except Exception as e:
            logger.warning(
                "Failed to count tokens with SPLADE tokenizer, using word count",
                extra={"text_length": len(text), "error": str(e)}
            )
            # Fallback to word count
            return len(text.split())

    def segment_chunk_for_splade(self, chunk_text: str, chunk_id: str) -> list[SPLADESegment]:
        """Segment chunk text into ≤512-token segments using SPLADE tokenizer.

        Args:
            chunk_text: Text content to segment
            chunk_id: Identifier for the chunk

        Returns:
            List of segments with token counts and boundaries
        """
        if not self._is_loaded:
            self._load_model()

        start_time = time.perf_counter()

        try:
            segments: list[SPLADESegment] = []
            current_segment: list[str] = []
            current_tokens = 0
            char_position = 0

            # Split text into sentences/paragraphs for better segmentation
            sentences = chunk_text.split('. ')
            if len(sentences) == 1:
                sentences = chunk_text.split('\n')

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Count tokens for this sentence
                sentence_tokens = self._count_tokens(sentence)

                # If adding this sentence would exceed max_tokens, finalize current segment
                if current_tokens + sentence_tokens > self.max_tokens and current_segment:
                    # Create segment from current content
                    segment_text = '. '.join(current_segment)
                    if segment_text.endswith('.'):
                        segment_text = segment_text[:-1]  # Remove trailing period

                    segment = SPLADESegment(
                        text=segment_text,
                        start_char=char_position - len(segment_text),
                        end_char=char_position,
                        token_count=current_tokens,
                        segment_id=f"{chunk_id}_seg_{len(segments)}"
                    )
                    segments.append(segment)

                    # Reset for next segment
                    current_segment = []
                    current_tokens = 0

                # Add sentence to current segment
                current_segment.append(sentence)
                current_tokens += sentence_tokens
                char_position += len(sentence) + 2  # +2 for '. ' separator

            # Add final segment if there's remaining content
            if current_segment:
                segment_text = '. '.join(current_segment)
                if segment_text.endswith('.'):
                    segment_text = segment_text[:-1]

                segment = SPLADESegment(
                    text=segment_text,
                    start_char=char_position - len(segment_text),
                    end_char=char_position,
                    token_count=current_tokens,
                    segment_id=f"{chunk_id}_seg_{len(segments)}"
                )
                segments.append(segment)

            processing_time = time.perf_counter() - start_time
            SPLADE_PROCESSING_SECONDS.labels(operation="segment", status="ok").observe(processing_time)

            logger.info(
                "Chunk segmented for SPLADE",
                extra={
                    "chunk_id": chunk_id,
                    "original_length": len(chunk_text),
                    "segments_count": len(segments),
                    "processing_time_seconds": processing_time,
                }
            )

            return segments

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            SPLADE_PROCESSING_SECONDS.labels(operation="segment", status="error").observe(processing_time)

            logger.error(
                "Failed to segment chunk for SPLADE",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                }
            )
            raise SPLADEProcessingError(f"Failed to segment chunk: {e}")

    def _encode_segment(self, segment: SPLADESegment) -> SPLADEVector:
        """Encode a single segment using SPLADE model.

        Args:
            segment: Segment to encode

        Returns:
            SPLADE sparse vector for the segment
        """
        if not self._is_loaded:
            self._load_model()

        try:
# import torch  # Removed for torch isolation

            # Tokenize segment
            inputs = self._tokenizer(  # type: ignore
                segment.text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_tokens
            )

            # Move to GPU if available
            # GPU functionality moved to gRPC services
            logger.info("SPLADE processing (GPU operations handled by gRPC services)")

            # Get model outputs
            # Torch functionality moved to gRPC services
            raise NotImplementedError("Torch functionality moved to gRPC services")

        except Exception as e:
            logger.error(
                "Failed to encode segment with SPLADE",
                extra={
                    "segment_id": segment.segment_id,
                    "error": str(e),
                }
            )
            raise SPLADEProcessingError(f"Failed to encode segment: {e}")

    def aggregate_splade_segments(self, segments: list[SPLADESegment], chunk_id: str) -> SPLADEVector:
        """Aggregate SPLADE segments using Rep-Max aggregation.

        Rep-Max aggregation merges segment vectors by taking the maximum weight
        per term, creating one learned-sparse vector per chunk.

        Args:
            segments: List of segments to aggregate
            chunk_id: Identifier for the chunk

        Returns:
            Aggregated SPLADE vector
        """
        if not self._is_loaded:
            self._load_model()

        start_time = time.perf_counter()

        try:
            # Encode each segment
            segment_vectors = []
            for segment in segments:
                vector = self._encode_segment(segment)
                segment_vectors.append(vector)

            # Rep-Max aggregation: take maximum weight per term
            aggregated_terms = {}
            for vector in segment_vectors:
                for term_id, weight in vector.terms.items():
                    if term_id not in aggregated_terms:
                        aggregated_terms[term_id] = weight
                    else:
                        aggregated_terms[term_id] = max(aggregated_terms[term_id], weight)

            # Apply sparsity threshold
            filtered_terms = {
                term_id: weight
                for term_id, weight in aggregated_terms.items()
                if weight >= self.sparsity_threshold
            }

            # Quantize weights
            quantized_terms: dict[int, float] = {
                term_id: float(int(weight * self.quantization_scale))
                for term_id, weight in filtered_terms.items()
            }

            aggregated_vector = SPLADEVector(
                terms=quantized_terms,
                tokenizer_name=self.model_name,
                model_name=self.model_name,
                sparsity_threshold=self.sparsity_threshold,
                quantization_scale=self.quantization_scale
            )

            processing_time = time.perf_counter() - start_time
            SPLADE_PROCESSING_SECONDS.labels(operation="aggregate", status="ok").observe(processing_time)

            logger.info(
                "SPLADE segments aggregated",
                extra={
                    "chunk_id": chunk_id,
                    "segments_count": len(segments),
                    "terms_count": len(quantized_terms),
                    "processing_time_seconds": processing_time,
                }
            )

            return aggregated_vector

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            SPLADE_PROCESSING_SECONDS.labels(operation="aggregate", status="error").observe(processing_time)

            logger.error(
                "Failed to aggregate SPLADE segments",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                }
            )
            raise SPLADEProcessingError(f"Failed to aggregate segments: {e}")

    def process_chunk(self, chunk_text: str, chunk_id: str) -> SPLADEResult:
        """Process a chunk through the complete SPLADE pipeline.

        Args:
            chunk_text: Text content of the chunk
            chunk_id: Unique identifier for the chunk

        Returns:
            SPLADE processing result with segments and aggregated vector
        """
        if not self._is_loaded:
            self._load_model()

        start_time = time.perf_counter()

        try:
            # Segment chunk
            segments = self.segment_chunk_for_splade(chunk_text, chunk_id)

            # Aggregate segments
            sparse_vector = self.aggregate_splade_segments(segments, chunk_id)

            # Calculate GPU memory usage
            gpu_memory_mb = 0
            # GPU functionality moved to gRPC services
            logger.info("GPU memory calculation handled by gRPC services")

            processing_time = time.perf_counter() - start_time

            result = SPLADEResult(
                chunk_id=chunk_id,
                segments=segments,
                sparse_vector=sparse_vector,
                processing_time_seconds=processing_time,
                gpu_memory_used_mb=gpu_memory_mb
            )

            SPLADE_OPERATIONS_TOTAL.labels(operation="process_chunk", status="ok").inc()
            SPLADE_GPU_MEMORY_MB.labels(operation="process_chunk").observe(gpu_memory_mb)

            logger.info(
                "Chunk processed with SPLADE",
                extra={
                    "chunk_id": chunk_id,
                    "segments_count": len(segments),
                    "terms_count": len(sparse_vector.terms),
                    "processing_time_seconds": processing_time,
                    "gpu_memory_mb": gpu_memory_mb,
                }
            )

            return result

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            SPLADE_OPERATIONS_TOTAL.labels(operation="process_chunk", status="error").inc()

            logger.error(
                "Failed to process chunk with SPLADE",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                }
            )
            raise SPLADEProcessingError(f"Failed to process chunk: {e}")

    def process_batch(self, chunks: list[tuple[str, str]]) -> list[SPLADEResult]:
        """Process multiple chunks in batch.

        Args:
            chunks: List of (chunk_text, chunk_id) tuples

        Returns:
            List of SPLADE processing results
        """
        if not self._is_loaded:
            self._load_model()

        start_time = time.perf_counter()
        results = []

        try:
            for chunk_text, chunk_id in chunks:
                try:
                    result = self.process_chunk(chunk_text, chunk_id)
                    results.append(result)
                except Exception as e:
                    logger.error(
                        "Failed to process chunk in batch",
                        extra={"chunk_id": chunk_id, "error": str(e)}
                    )
                    # Continue with other chunks
                    continue

            processing_time = time.perf_counter() - start_time

            logger.info(
                "Batch processed with SPLADE",
                extra={
                    "total_chunks": len(chunks),
                    "successful_chunks": len(results),
                    "processing_time_seconds": processing_time,
                }
            )

            return results

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            logger.error(
                "Failed to process batch with SPLADE",
                extra={
                    "total_chunks": len(chunks),
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                }
            )
            raise SPLADEProcessingError(f"Failed to process batch: {e}")

    def health_check(self) -> dict[str, Any]:
        """Check SPLADE service health.

        Returns:
            Health status information
        """
        try:
            # Check GPU availability
            gpu_available = False
            if self.gpu_manager and hasattr(self.gpu_manager, 'is_gpu_available'):
                gpu_available = self.gpu_manager.is_gpu_available()

            # Check model loading
            model_loaded = self._is_loaded

            # Try to load model if not loaded
            if not model_loaded:
                try:
                    self._load_model()
                    model_loaded = True
                except Exception as e:
                    logger.warning("SPLADE model failed to load during health check", extra={"error": str(e)})  # noqa: BLE001

            health_status = {
                "status": "healthy" if gpu_available and model_loaded else "unhealthy",
                "gpu_available": gpu_available,
                "model_loaded": model_loaded,
                "model_name": self.model_name,
                "max_tokens": self.max_tokens,
                "sparsity_threshold": self.sparsity_threshold,
                "quantization_scale": self.quantization_scale,
            }

            return health_status

        except Exception as e:
            logger.error("SPLADE health check failed", extra={"error": str(e)})  # noqa: BLE001
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": self.model_name,
            }
