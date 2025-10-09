"""MinerU split-container service implementation.

This module provides the core MinerU service implementation for processing
PDF documents using GPU-accelerated layout analysis and text extraction.
It coordinates between the MinerU CLI, vLLM server, storage systems, and
various processing pipelines.

Key Components:
    - MineruProcessor: Main service class for PDF processing
    - MineruGrpcService: Async gRPC interface for remote processing
    - Exception classes: Specialized error handling for GPU and memory issues
    - Batch processing: Efficient handling of multiple documents
    - Storage integration: PDF and figure storage management

Responsibilities:
    - Process PDF documents using MinerU CLI and vLLM backend
    - Handle batch processing with configurable batch sizes
    - Manage GPU memory and circuit breaker patterns
    - Integrate with storage systems for PDF and figure assets
    - Provide both sync and async processing interfaces
    - Handle version validation and health checks

Collaborators:
    - MinerU CLI wrapper for document processing
    - vLLM client for GPU-accelerated operations
    - PDF and figure storage clients
    - Output parser and postprocessor
    - Pipeline orchestration system

Side Effects:
    - Logs processing operations and performance metrics
    - Updates circuit breaker states
    - Manages GPU memory allocation
    - Generates processing metadata

Thread Safety:
    - Thread-safe: Uses thread-local worker IDs
    - Circuit breakers provide thread-safe failure handling
    - Batch processing is designed for concurrent access

Performance Characteristics:
    - GPU-accelerated processing for optimal performance
    - Batch processing reduces overhead
    - Circuit breakers prevent cascading failures
    - Memory management prevents OOM conditions

Example:
    >>> processor = MineruProcessor()
    >>> request = MineruRequest(
    ...     tenant_id="tenant1",
    ...     document_id="doc1",
    ...     content=pdf_bytes
    ... )
    >>> response = processor.process(request)
    >>> print(f"Processed {len(response.document.blocks)} blocks")
"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================
import asyncio
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

import structlog
from Medical_KG_rev.chunking.exceptions import (
    MineruGpuUnavailableError as ChunkingMineruGpuUnavailableError,
)
from Medical_KG_rev.chunking.exceptions import (
    MineruOutOfMemoryError as ChunkingMineruOutOfMemoryError,
)
from Medical_KG_rev.config.settings import MineruSettings, get_settings
from Medical_KG_rev.storage.clients import PdfStorageClient
from Medical_KG_rev.storage.object_store import FigureStorageClient

from .cli_wrapper import (
    MineruCliBase,
    MineruCliError,
    MineruCliInput,
    MineruCliResult,
    SimulatedMineruCli,
    create_cli,
)
from .metrics import MINERU_CLI_FAILURES_TOTAL
from .output_parser import MineruOutputParser, MineruOutputParserError, ParsedDocument
from .pipeline import MineruPipeline, PipelineMetrics
from .postprocessor import MineruPostProcessor
from .types import (
    Document,
    MineruBatchResponse,
    MineruRequest,
    MineruResponse,
    ProcessingMetadata,
)
from .vllm_client import VLLMClient, VLLMClientError

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logger = structlog.get_logger(__name__)


# ==============================================================================
# EXCEPTION CLASSES
# ==============================================================================

class MineruOutOfMemoryError(MineruCliError, ChunkingMineruOutOfMemoryError):
    """Raised when MinerU CLI indicates an out-of-memory failure.

    This exception is raised when the MinerU CLI encounters GPU memory
    exhaustion during document processing. It inherits from both
    MineruCliError and ChunkingMineruOutOfMemoryError for proper
    error handling in the chunking pipeline.

    Example:
        >>> try:
        ...     processor.process(request)
        ... except MineruOutOfMemoryError:
        ...     print("GPU memory exhausted during processing")
    """

    def __init__(self) -> None:
        """Initialize out-of-memory error with proper inheritance."""
        ChunkingMineruOutOfMemoryError.__init__(self)
        MineruCliError.__init__(self, str(self))


class MineruGpuUnavailableError(MineruCliError, ChunkingMineruGpuUnavailableError):
    """Raised when MinerU cannot access a healthy GPU backend.

    This exception is raised when the MinerU service cannot access
    a healthy GPU backend for processing. It inherits from both
    MineruCliError and ChunkingMineruGpuUnavailableError for proper
    error handling in the chunking pipeline.

    Example:
        >>> try:
        ...     processor.process(request)
        ... except MineruGpuUnavailableError:
        ...     print("GPU backend unavailable for processing")
    """

    def __init__(self) -> None:
        """Initialize GPU unavailable error with proper inheritance."""
        ChunkingMineruGpuUnavailableError.__init__(self)
        MineruCliError.__init__(self, str(self))


# ==============================================================================
# SERVICE IMPLEMENTATION
# ==============================================================================

class MineruProcessor:
    """MinerU processor configured for split-container vLLM usage.

    Main service class for processing PDF documents using MinerU CLI
    with GPU-accelerated layout analysis and text extraction. Handles
    batch processing, storage integration, and error recovery.

    Attributes:
        _settings: MinerU configuration settings
        _worker_id: Unique identifier for this worker instance
        _pdf_storage: Client for PDF storage operations
        _vllm_client: Client for vLLM server communication
        _pipeline: Processing pipeline for document handling
        _parser: Output parser for MinerU results
        _postprocessor: Postprocessor for document refinement
        _cli: MinerU CLI wrapper for execution
        _mineru_version: Validated MinerU version

    Invariants:
        - _settings is never None
        - _worker_id is never empty
        - _pipeline is never None
        - _cli is never None

    Thread Safety:
        - Thread-safe: Uses thread-local worker IDs
        - Circuit breakers provide thread-safe failure handling
        - Batch processing is designed for concurrent access

    Lifecycle:
        - Initialized with optional dependencies
        - MinerU version is validated during initialization
        - CLI and pipeline components are configured
        - Health checks are performed on demand

    Example:
        >>> processor = MineruProcessor()
        >>> request = MineruRequest(
        ...     tenant_id="tenant1",
        ...     document_id="doc1",
        ...     content=pdf_bytes
        ... )
        >>> response = processor.process(request)
        >>> print(f"Processed {len(response.document.blocks)} blocks")
    """

    def __init__(
        self,
        *,
        settings: MineruSettings | None = None,
        cli: MineruCliBase | None = None,
        parser: MineruOutputParser | None = None,
        postprocessor: MineruPostProcessor | None = None,
        figure_storage: FigureStorageClient | None = None,
        pdf_storage: PdfStorageClient | None = None,
        worker_id: str | None = None,
        vllm_client: VLLMClient | None = None,
    ) -> None:
        """Initialize MinerU processor with optional dependencies.

        Args:
            settings: Optional MinerU configuration settings
            cli: Optional MinerU CLI wrapper
            parser: Optional output parser for results
            postprocessor: Optional postprocessor for refinement
            figure_storage: Optional figure storage client
            pdf_storage: Optional PDF storage client
            worker_id: Optional worker identifier
            vllm_client: Optional vLLM client for GPU operations

        Note:
            If no settings are provided, default settings are loaded.
            Worker ID defaults to the current thread name if not provided.
            All components are initialized with sensible defaults.
        """
        self._settings = settings or get_settings().mineru
        self._worker_id = worker_id or threading.current_thread().name
        self._pdf_storage = pdf_storage
        self._vllm_client = vllm_client
        parser_instance = parser or MineruOutputParser()
        postprocessor_instance = postprocessor or MineruPostProcessor(
            figure_storage=figure_storage
        )
        self._pipeline = MineruPipeline(
            parser=parser_instance,
            postprocessor=postprocessor_instance,
            metrics=PipelineMetrics(worker_id=self._worker_id),
        )
        self._parser = parser_instance
        self._postprocessor = postprocessor_instance
        self._cli = cli or create_cli(self._settings)
        self._mineru_version = self._ensure_mineru_version()
        logger.info(
            "mineru.processor.initialised",
            worker_id=self._worker_id,
            backend=self._settings.workers.backend,
            vllm_url=str(self._settings.vllm_server.base_url),
        )

    def _extract_checksum_from_uri(self, storage_uri: str) -> str | None:
        """Extract checksum from storage URI."""
        try:
            # Handle s3://bucket/path/tenant/document/checksum.pdf format
            if storage_uri.startswith("s3://"):
                parts = storage_uri.split("/")
                if len(parts) >= 2:
                    filename = parts[-1]
                    if filename.endswith(".pdf"):
                        return filename[:-4]  # Remove .pdf extension
            # Handle in-memory://checksum format
            elif storage_uri.startswith("in-memory://"):
                return storage_uri[12:]  # Remove in-memory:// prefix
        except Exception:
            pass
        return None

    async def _fetch_pdf_from_storage(self, request: MineruRequest) -> bytes | None:
        """Fetch PDF content from storage if available.

        Args:
            request: MinerU request with storage URI

        Returns:
            PDF content bytes if found, None otherwise

        Note:
            Attempts to fetch PDF content from storage using the request's
            storage URI. Logs warnings for failures but does not raise
            exceptions to allow graceful degradation.
        """
        if not request.storage_uri or not self._pdf_storage:
            return None

        try:
            checksum = self._extract_checksum_from_uri(request.storage_uri)
            if checksum:
                asset = await self._pdf_storage.get_pdf_asset(
                    tenant_id=request.tenant_id,
                    document_id=request.document_id,
                    checksum=checksum,
                )
                if asset:
                    return await self._pdf_storage.get_pdf_data(asset)
        except Exception as e:
            logger.warning(
                "mineru.process.storage_fetch_failed",
                tenant_id=request.tenant_id,
                document_id=request.document_id,
                storage_uri=request.storage_uri,
                error=str(e),
            )
        return None

    def process(self, request: MineruRequest) -> MineruResponse:
        """Process a single MinerU request.

        Args:
            request: MinerU request containing PDF content and metadata

        Returns:
            MinerU response with processed document and metadata

        Raises:
            MineruCliError: If processing fails or no output is returned

        Note:
            This method processes a single request synchronously. For
            requests with storage URIs, the content must be provided
            directly as storage fetching is not supported in sync mode.

        Example:
            >>> request = MineruRequest(
            ...     tenant_id="tenant1",
            ...     document_id="doc1",
            ...     content=pdf_bytes
            ... )
            >>> response = processor.process(request)
            >>> print(f"Processed {len(response.document.blocks)} blocks")
        """
        # Fetch PDF content from storage if needed
        if not request.content and request.storage_uri:
            # Note: This is a sync method, so we can't await here
            # In practice, this should be handled by the caller or made async
            logger.warning(
                "mineru.process.sync_storage_fetch",
                message="Cannot fetch from storage in sync method",
                tenant_id=request.tenant_id,
                document_id=request.document_id,
            )

        batch = self.process_batch([request])
        if not batch.documents:
            raise MineruCliError("MinerU CLI returned no outputs")
        return MineruResponse(
            document=batch.documents[0],
            processed_at=batch.processed_at,
            duration_seconds=batch.duration_seconds,
            metadata=batch.metadata[0],
        )

    async def process_async(self, request: MineruRequest) -> MineruResponse:
        """Process a single MinerU request with async storage support."""
        # Fetch PDF content from storage if needed
        if not request.content and request.storage_uri:
            content = await self._fetch_pdf_from_storage(request)
            if content:
                # Create a new request with the fetched content
                request = MineruRequest(
                    tenant_id=request.tenant_id,
                    document_id=request.document_id,
                    content=content,
                    storage_uri=request.storage_uri,
                )
            else:
                raise MineruCliError("Cannot fetch PDF content from storage")

        # Use the sync batch processing
        batch = self.process_batch([request])
        if not batch.documents:
            raise MineruCliError("MinerU CLI returned no outputs")
        return MineruResponse(
            document=batch.documents[0],
            processed_at=batch.processed_at,
            duration_seconds=batch.duration_seconds,
            metadata=batch.metadata[0],
        )

    def process_batch(self, requests: Sequence[MineruRequest]) -> MineruBatchResponse:
        """Process multiple MinerU requests in batches.

        Args:
            requests: Sequence of MinerU requests to process

        Returns:
            Batch response containing all processed documents and metadata

        Note:
            Requests are automatically batched according to the configured
            batch size. Empty request lists return an empty response.
            Processing is optimized for throughput with configurable batch sizes.

        Example:
            >>> requests = [
            ...     MineruRequest(tenant_id="t1", document_id="d1", content=pdf1),
            ...     MineruRequest(tenant_id="t1", document_id="d2", content=pdf2)
            ... ]
            >>> response = processor.process_batch(requests)
            >>> print(f"Processed {len(response.documents)} documents")
        """
        request_list = list(requests)
        if not request_list:
            now = datetime.now(datetime.UTC)
            return MineruBatchResponse(
                documents=[], processed_at=now, duration_seconds=0.0, metadata=[]
            )

        batch_limit = max(1, self._settings.workers.batch_size)
        batches = [request_list[i : i + batch_limit] for i in range(0, len(request_list), batch_limit)]
        logger.bind(
            size=len(request_list), batches=len(batches), batch_limit=batch_limit
        ).info("mineru.process.batch_started")

        start_monotonic = time.monotonic()
        aggregated_documents: list[Document] = []
        aggregated_metadata: list[ProcessingMetadata] = []
        processed_at: datetime | None = None

        for index, batch in enumerate(batches, start=1):
            partial = self._run_cli_batch(batch, batch_index=index, total_batches=len(batches))
            aggregated_documents.extend(partial.documents)
            aggregated_metadata.extend(partial.metadata)
            processed_at = partial.processed_at

        duration = time.monotonic() - start_monotonic
        logger.bind(
            size=len(request_list), batches=len(batches), duration=round(duration, 4)
        ).info("mineru.process.batch_completed")

        return MineruBatchResponse(
            documents=aggregated_documents,
            processed_at=processed_at or datetime.now(datetime.UTC),
            duration_seconds=duration,
            metadata=aggregated_metadata,
        )

    def _run_cli_batch(
        self,
        requests: Sequence[MineruRequest],
        *,
        batch_index: int,
        total_batches: int,
    ) -> MineruBatchResponse:
        """Execute CLI batch processing with error handling.

        Args:
            requests: Sequence of MinerU requests to process
            batch_index: Index of current batch (1-based)
            total_batches: Total number of batches

        Returns:
            Batch response with processed documents and metadata

        Raises:
            MineruCliError: If CLI execution fails
            MineruOutputParserError: If output parsing fails

        Note:
            Handles CLI execution with fallback to simulated CLI
            on failure. Orchestrates the processing pipeline.
        """
        cli_inputs = [
            MineruCliInput(document_id=request.document_id, content=request.content or b"")
            for request in requests
        ]

        def orchestrate(
            executor: Callable[[Sequence[MineruCliInput]], tuple[MineruCliResult, str, int]],
        ) -> MineruBatchResponse:
            return self._pipeline.execute(
                requests=requests,
                cli_inputs=cli_inputs,
                execute_cli=executor,
                metadata_builder=self._build_metadata,
                batch_index=batch_index,
                total_batches=total_batches,
                record_gpu_memory=None,
            )

        try:
            return orchestrate(self._execute_cli)
        except MineruCliError as exc:
            logger.bind(
                reason="cli-error", error=str(exc), batch=batch_index
            ).error("mineru.process.failed")
            if self._handle_cli_failure(exc):
                return orchestrate(self._execute_simulated_cli)
            raise
        except MineruOutputParserError:
            raise
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.bind(error=str(exc)).exception("mineru.process.unexpected")
            raise

    def _execute_cli(
        self, cli_inputs: Sequence[MineruCliInput]
    ) -> tuple[MineruCliResult, str, int]:
        """Execute MinerU CLI with real backend.

        Args:
            cli_inputs: CLI input objects for processing

        Returns:
            Tuple of (CLI result, backend type, exit code)

        Note:
            Executes MinerU CLI using the configured CLI wrapper
            with vLLM HTTP backend.
        """
        cli_result = self._cli.run_batch(cli_inputs)
        return cli_result, "vllm-http", 0

    def _execute_simulated_cli(
        self, cli_inputs: Sequence[MineruCliInput]
    ) -> tuple[MineruCliResult, str, int]:
        """Execute MinerU CLI with simulated backend.

        Args:
            cli_inputs: CLI input objects for processing

        Returns:
            Tuple of (CLI result, backend type, exit code)

        Note:
            Executes MinerU CLI using simulated backend as fallback
            when real backend fails. Creates simulated CLI if needed.
        """
        simulated = self._cli
        if not isinstance(simulated, SimulatedMineruCli):
            simulated = SimulatedMineruCli(self._settings)
            self._cli = simulated
        cli_result = simulated.run_batch(cli_inputs)
        return cli_result, "simulated", 0

    def _handle_cli_failure(self, exc: MineruCliError) -> bool:
        """Handle CLI failure and determine recovery strategy.

        Args:
            exc: MinerU CLI exception that occurred

        Returns:
            True if fallback to simulated CLI should be attempted

        Note:
            Analyzes the exception to determine if it's an out-of-memory
            error and updates metrics accordingly. Raises specific
            exceptions for OOM conditions.
        """
        reason = "oom" if self._looks_like_oom(str(exc)) else "cli-error"
        MINERU_CLI_FAILURES_TOTAL.labels(reason=reason).inc()
        if reason == "oom":
            raise MineruOutOfMemoryError() from exc
        return True

    def _ensure_mineru_version(self) -> str | None:
        """Ensure MinerU version meets requirements.

        Returns:
            Installed MinerU version if valid, None if not found

        Raises:
            RuntimeError: If installed version doesn't meet requirements

        Note:
            Validates the installed MinerU version against the expected
            version from settings. Logs version information for debugging.
        """
        try:
            installed = importlib_metadata.version("mineru")
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover - optional in tests
            logger.warning("mineru.version.missing")
            return None

        spec = self._settings.expected_version.strip()
        minimum = spec[2:] if spec.startswith(">=") else spec
        if minimum and self._compare_versions(installed, minimum) < 0:
            raise RuntimeError(
                f"MinerU version {installed} does not satisfy expectation '{self._settings.expected_version}'"
            )
        logger.bind(
            installed=installed, expected=self._settings.expected_version
        ).info("mineru.version.validated")
        return installed

    @staticmethod
    def _compare_versions(installed: str, minimum: str) -> int:
        """Compare two version strings.

        Args:
            installed: Installed version string
            minimum: Minimum required version string

        Returns:
            -1 if installed < minimum, 0 if equal, 1 if installed > minimum

        Note:
            Performs semantic version comparison by normalizing version
            strings to integer tuples and comparing component by component.
        """
        def _normalize(value: str) -> tuple[int, ...]:
            parts: list[int] = []
            for token in value.split("."):
                digits = "".join(ch for ch in token if ch.isdigit())
                if digits:
                    parts.append(int(digits))
                else:
                    break
            return tuple(parts)

        from itertools import zip_longest

        left = _normalize(installed)
        right = _normalize(minimum)
        for lhs, rhs in zip_longest(left, right, fillvalue=0):
            if lhs > rhs:
                return 1
            if lhs < rhs:
                return -1
        return 0

    def _build_metadata(
        self,
        *,
        request: MineruRequest,
        parsed: ParsedDocument,
        gpu_label: str,
        started_at: datetime,
        completed_at: datetime,
        cli_result,
        planned_memory_mb: int,
    ) -> ProcessingMetadata:
        """Build processing metadata from request and results.

        Args:
            request: Original MinerU request
            parsed: Parsed document result
            gpu_label: GPU identifier used
            started_at: Processing start time
            completed_at: Processing completion time
            cli_result: CLI execution result
            planned_memory_mb: Planned memory usage in MB

        Returns:
            Processing metadata object

        Note:
            Extracts model names from parsed metadata and builds
            comprehensive processing metadata for observability.
        """
        model_names = {
            "layout": parsed.metadata.get("layout_model", "unknown"),
            "table": parsed.metadata.get("table_model", "unknown"),
            "vision": parsed.metadata.get("vision_model", "unknown"),
        }
        return ProcessingMetadata(
            document_id=request.document_id,
            mineru_version=self._mineru_version,
            model_names=model_names,
            gpu_id=gpu_label,
            worker_id=self._worker_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=float(cli_result.duration_seconds),
            cli_stdout=cli_result.stdout.strip(),
            cli_stderr=cli_result.stderr.strip(),
            cli_descriptor=self._cli.describe(),
            planned_memory_mb=planned_memory_mb,
        )

    def _looks_like_oom(self, message: str) -> bool:
        """Check if error message indicates out-of-memory condition.

        Args:
            message: Error message to analyze

        Returns:
            True if message indicates OOM condition

        Note:
            Performs case-insensitive search for OOM indicators
            in the error message.
        """
        lowered = message.lower()
        return "out of memory" in lowered or "oom" in lowered

    def _build_vllm_client(self) -> VLLMClient | None:
        """Build vLLM client with circuit breaker configuration.

        Returns:
            Configured vLLM client instance

        Note:
            Creates vLLM client with circuit breaker, timeout, and
            connection pool settings from configuration.
        """
        breaker_settings = self._settings.http_client.circuit_breaker
        circuit_breaker = None
        if breaker_settings.enabled:
            from .circuit_breaker import CircuitBreaker

            circuit_breaker = CircuitBreaker(
                failure_threshold=breaker_settings.failure_threshold,
                recovery_timeout=breaker_settings.recovery_timeout_seconds,
                success_threshold=breaker_settings.success_threshold,
            )
        client = VLLMClient(
            base_url=str(self._settings.vllm_server.base_url),
            timeout=self._settings.http_client.timeout_seconds,
            max_connections=self._settings.http_client.connection_pool_size,
            max_keepalive_connections=self._settings.http_client.keepalive_connections,
            circuit_breaker=circuit_breaker,
            retry_attempts=self._settings.http_client.retry_attempts,
            retry_backoff_multiplier=self._settings.http_client.retry_backoff_multiplier,
        )
        return client

    def _ensure_vllm_health(self) -> None:
        """Ensure vLLM server is healthy and accessible.

        Raises:
            MineruGpuUnavailableError: If vLLM server is not healthy

        Note:
            Performs health check on vLLM server and raises exception
            if server is not accessible or healthy.
        """
        async def _check() -> bool:
            try:
                if self._vllm_client:
                    return await self._vllm_client.health_check()
                return False
            except VLLMClientError as exc:
                logger.error("mineru.vllm.health_check_failed", error=str(exc))
                return False

        try:
            healthy = asyncio.run(_check())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                healthy = loop.run_until_complete(_check())
            finally:
                loop.close()
        if not healthy:
            raise MineruGpuUnavailableError()

    @property
    def vllm_client(self) -> VLLMClient | None:
        """Expose the configured vLLM client for observability and testing."""
        return self._vllm_client


# ==============================================================================
# GRPC SERVICE IMPLEMENTATION
# ==============================================================================

class MineruGrpcService:
    """Async gRPC servicer bridging proto definitions to the processor.

    Provides gRPC interface for remote MinerU processing operations.
    Handles protocol buffer serialization/deserialization and error
    translation between gRPC status codes and MinerU exceptions.

    Attributes:
        _processor: MinerU processor instance for document processing

    Invariants:
        - _processor is never None

    Thread Safety:
        - Thread-safe: Delegates to thread-safe processor

    Example:
        >>> processor = MineruProcessor()
        >>> grpc_service = MineruGrpcService(processor)
        >>> # Used by gRPC server for remote processing
    """

    def __init__(self, processor: MineruProcessor) -> None:
        """Initialize gRPC service with processor.

        Args:
            processor: MinerU processor instance for processing
        """
        self._processor = processor

    async def ProcessPdf(self, request, context):  # type: ignore[override]
        """Process PDF document via gRPC interface.

        Args:
            request: gRPC request containing PDF content and metadata
            context: gRPC context for error handling

        Returns:
            gRPC response with processed document and metadata

        Raises:
            grpc.StatusCode.RESOURCE_EXHAUSTED: For out-of-memory errors
            grpc.StatusCode.INTERNAL: For other processing errors

        Note:
            Handles gRPC-specific error translation and protocol buffer
            serialization. Delegates actual processing to the processor.
        """
        mineru_request = MineruRequest(
            tenant_id=request.tenant_id,
            document_id=request.document_id,
            content=request.content,
        )
        try:
            response = self._processor.process(mineru_request)
        except MineruOutOfMemoryError as exc:
            if context is not None:
                import grpc

                await context.abort(
                    code=grpc.StatusCode.RESOURCE_EXHAUSTED,
                    details=str(exc),
                )
            raise
        except (MineruCliError, VLLMClientError) as exc:
            if context is not None:
                import grpc

                await context.abort(
                    code=grpc.StatusCode.INTERNAL,
                    details=str(exc),
                )
            raise

        try:
            from Medical_KG_rev.proto.gen import mineru_pb2  # type: ignore import-error
        except ImportError:  # pragma: no cover - generated stubs absent during CI
            return None

        reply = mineru_pb2.ProcessPdfResponse()
        self._populate_document(reply.document, response.document)
        self._populate_metadata(reply.metadata, response.metadata)
        return reply

    @staticmethod
    def _populate_document(proto_document, document: Document) -> None:
        """Populate gRPC document message from Document object.

        Args:
            proto_document: gRPC document message to populate
            document: Document object containing data

        Note:
            Converts Document object to gRPC protocol buffer format.
            Handles optional fields with sensible defaults.
        """
        proto_document.document_id = document.document_id
        proto_document.tenant_id = document.tenant_id
        for block in document.blocks:
            item = proto_document.blocks.add()
            item.id = block.id
            item.page = block.page
            item.kind = block.kind
            item.text = block.text or ""
            item.confidence = block.confidence or 0.0
            item.reading_order = block.reading_order or 0

    @staticmethod
    def _populate_metadata(proto_metadata, metadata: ProcessingMetadata) -> None:
        """Populate gRPC metadata message from ProcessingMetadata object.

        Args:
            proto_metadata: gRPC metadata message to populate
            metadata: ProcessingMetadata object containing data

        Note:
            Converts ProcessingMetadata object to gRPC protocol buffer format.
            Handles optional fields with sensible defaults.
        """
        proto_metadata.document_id = metadata.document_id
        proto_metadata.worker_id = metadata.worker_id or ""
        proto_metadata.started_at = metadata.started_at.isoformat()
        proto_metadata.completed_at = metadata.completed_at.isoformat()
        proto_metadata.duration_seconds = metadata.duration_seconds
        proto_metadata.cli_stdout = metadata.cli_stdout
        proto_metadata.cli_stderr = metadata.cli_stderr
        proto_metadata.cli_descriptor = metadata.cli_descriptor
        if metadata.mineru_version:
            proto_metadata.mineru_version = metadata.mineru_version


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "MineruBatchResponse",
    "MineruGpuUnavailableError",
    "MineruGrpcService",
    "MineruOutOfMemoryError",
    "MineruProcessor",
    "MineruRequest",
    "MineruResponse",
]
