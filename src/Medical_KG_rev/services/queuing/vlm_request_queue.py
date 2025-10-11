"""Request queuing system for VLM processing under load.

This module provides a comprehensive request queuing system for VLM processing,
ensuring efficient handling of requests under high load conditions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import asyncio
import time
import uuid

import structlog


logger = structlog.get_logger(__name__)


class RequestPriority(Enum):
    """Request priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class RequestStatus(Enum):
    """Request status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class QueueStrategy(Enum):
    """Queue processing strategy."""

    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based
    ROUND_ROBIN = "round_robin"  # Round-robin
    WEIGHTED = "weighted"  # Weighted round-robin


@dataclass
class VLMRequest:
    """VLM processing request."""

    request_id: str
    pdf_content: bytes
    config: dict[str, Any]
    options: dict[str, Any]
    priority: RequestPriority = RequestPriority.NORMAL
    status: RequestStatus = RequestStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    timeout: float = 300.0  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None


@dataclass
class QueueConfig:
    """Configuration for request queue."""

    max_queue_size: int = 1000
    max_concurrent_requests: int = 10
    strategy: QueueStrategy = QueueStrategy.PRIORITY
    default_timeout: float = 300.0
    default_priority: RequestPriority = RequestPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 30.0
    cleanup_interval: float = 300.0
    enable_metrics: bool = True
    enable_tracing: bool = True


@dataclass
class QueueMetrics:
    """Queue performance metrics."""

    total_requests: int = 0
    pending_requests: int = 0
    processing_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    cancelled_requests: int = 0
    timeout_requests: int = 0
    average_processing_time: float = 0.0
    average_wait_time: float = 0.0
    queue_utilization: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    timestamp: float = field(default_factory=time.time)


class VLMRequestQueue:
    """Request queue for VLM processing."""

    def __init__(self, config: QueueConfig):
        self.config = config
        self.queue: asyncio.Queue[VLMRequest] = asyncio.Queue(maxsize=config.max_queue_size)
        self.active_requests: dict[str, VLMRequest] = {}
        self.completed_requests: dict[str, VLMRequest] = {}
        self.failed_requests: dict[str, VLMRequest] = {}
        self.request_history: list[VLMRequest] = []

        # Processing control
        self.processing_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.processing_tasks: dict[str, asyncio.Task[None]] = {}
        self.is_running = False
        self.stop_event = asyncio.Event()

        # Metrics
        self.metrics = QueueMetrics()
        self.metrics_history: list[QueueMetrics] = []

        # Background tasks
        self.health_check_task: asyncio.Task[None] | None = None
        self.cleanup_task: asyncio.Task[None] | None = None
        self.metrics_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the request queue."""
        if self.is_running:
            logger.warning("Request queue is already running")
            return

        self.is_running = True
        self.stop_event.clear()

        # Start background tasks
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.metrics_task = asyncio.create_task(self._metrics_loop())

        logger.info("VLM request queue started")

    async def stop(self) -> None:
        """Stop the request queue."""
        if not self.is_running:
            logger.warning("Request queue is not running")
            return

        self.is_running = False
        self.stop_event.set()

        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)

        # Stop background tasks
        if self.health_check_task:
            self.health_check_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.metrics_task:
            self.metrics_task.cancel()

        # Wait for background tasks
        tasks = [self.health_check_task, self.cleanup_task, self.metrics_task]
        await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)

        logger.info("VLM request queue stopped")

    async def submit_request(
        self,
        pdf_content: bytes,
        config: dict[str, Any],
        options: dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float | None = None,
        max_retries: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Submit a request to the queue."""
        if not self.is_running:
            raise RuntimeError("Request queue is not running")

        if self.queue.full():
            raise RuntimeError("Request queue is full")

        request_id = str(uuid.uuid4())
        request = VLMRequest(
            request_id=request_id,
            pdf_content=pdf_content,
            config=config,
            options=options,
            priority=priority,
            timeout=timeout or self.config.default_timeout,
            max_retries=max_retries or self.config.max_retries,
            metadata=metadata or {},
        )

        # Add to queue based on strategy
        await self._add_to_queue(request)

        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.pending_requests += 1

        logger.info(
            "Request submitted",
            request_id=request_id,
            priority=priority.value,
            queue_size=self.queue.qsize(),
        )

        return request_id

    async def _add_to_queue(self, request: VLMRequest) -> None:
        """Add request to queue based on strategy."""
        if self.config.strategy == QueueStrategy.FIFO:
            await self.queue.put(request)
        elif self.config.strategy == QueueStrategy.PRIORITY:
            # For priority queue, we need to implement priority ordering
            # This is a simplified implementation
            await self.queue.put(request)
        elif self.config.strategy == QueueStrategy.ROUND_ROBIN:
            await self.queue.put(request)
        elif self.config.strategy == QueueStrategy.WEIGHTED:
            await self.queue.put(request)
        # All strategies use the same implementation for now

    async def get_request(self) -> VLMRequest | None:
        """Get next request from queue."""
        try:
            request = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            return request
        except asyncio.TimeoutError:
            return None

    async def process_request(self, request: VLMRequest, processor_func: Any) -> None:
        """Process a single request."""
        async with self.processing_semaphore:
            try:
                # Update request status
                request.status = RequestStatus.PROCESSING
                request.started_at = time.time()
                self.active_requests[request.request_id] = request
                self.metrics.processing_requests += 1
                self.metrics.pending_requests -= 1

                logger.info(
                    "Processing request",
                    request_id=request.request_id,
                    priority=request.priority.value,
                )

                # Process the request
                result = await processor_func(request.pdf_content, request.config, request.options)

                # Update request with result
                request.result = result
                request.status = RequestStatus.COMPLETED
                request.completed_at = time.time()

                # Move to completed requests
                self.completed_requests[request.request_id] = request
                del self.active_requests[request.request_id]

                # Update metrics
                self.metrics.completed_requests += 1
                self.metrics.processing_requests -= 1

                # Calculate processing time
                processing_time = request.completed_at - request.started_at
                self._update_average_processing_time(processing_time)

                logger.info(
                    "Request completed",
                    request_id=request.request_id,
                    processing_time=processing_time,
                )

            except asyncio.CancelledError:
                # Request was cancelled
                request.status = RequestStatus.CANCELLED
                request.completed_at = time.time()
                self.failed_requests[request.request_id] = request
                if request.request_id in self.active_requests:
                    del self.active_requests[request.request_id]
                self.metrics.cancelled_requests += 1
                self.metrics.processing_requests -= 1

                logger.info("Request cancelled", request_id=request.request_id)
                raise

            except Exception as e:
                # Request failed
                request.status = RequestStatus.FAILED
                request.error = str(e)
                request.completed_at = time.time()

                # Handle retries
                if request.retry_count < request.max_retries:
                    request.retry_count += 1
                    request.status = RequestStatus.PENDING

                    # Wait before retry
                    await asyncio.sleep(self.config.retry_delay * request.retry_count)

                    # Re-queue the request
                    await self._add_to_queue(request)
                    self.metrics.pending_requests += 1

                    logger.warning(
                        "Request failed, retrying",
                        request_id=request.request_id,
                        error=str(e),
                        retry_count=request.retry_count,
                    )
                else:
                    # Max retries exceeded
                    self.failed_requests[request.request_id] = request
                    if request.request_id in self.active_requests:
                        del self.active_requests[request.request_id]
                    self.metrics.failed_requests += 1
                    self.metrics.processing_requests -= 1

                    logger.error(
                        "Request failed after max retries",
                        request_id=request.request_id,
                        error=str(e),
                        retry_count=request.retry_count,
                    )

    def _update_average_processing_time(self, processing_time: float) -> None:
        """Update average processing time."""
        if self.metrics.completed_requests == 1:
            self.metrics.average_processing_time = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_processing_time = (
                alpha * processing_time + (1 - alpha) * self.metrics.average_processing_time
            )

    async def _health_check_loop(self) -> None:
        """Health check loop."""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Check for timeout requests
                current_time = time.time()
                timeout_requests = []

                for request in self.active_requests.values():
                    if request.started_at and current_time - request.started_at > request.timeout:
                        timeout_requests.append(request)

                # Handle timeout requests
                for request in timeout_requests:
                    request.status = RequestStatus.TIMEOUT
                    request.completed_at = current_time
                    self.failed_requests[request.request_id] = request
                    del self.active_requests[request.request_id]
                    self.metrics.timeout_requests += 1
                    self.metrics.processing_requests -= 1

                    logger.warning(
                        "Request timed out", request_id=request.request_id, timeout=request.timeout
                    )

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _cleanup_loop(self) -> None:
        """Cleanup loop for old requests."""
        while self.is_running and not self.stop_event.is_set():
            try:
                current_time = time.time()
                cutoff_time = current_time - 3600  # 1 hour

                # Clean up old completed requests
                old_completed = [
                    req_id
                    for req_id, req in self.completed_requests.items()
                    if req.completed_at and req.completed_at < cutoff_time
                ]
                for req_id in old_completed:
                    del self.completed_requests[req_id]

                # Clean up old failed requests
                old_failed = [
                    req_id
                    for req_id, req in self.failed_requests.items()
                    if req.completed_at and req.completed_at < cutoff_time
                ]
                for req_id in old_failed:
                    del self.failed_requests[req_id]

                # Clean up old request history
                self.request_history = [
                    req
                    for req in self.request_history
                    if req.completed_at and req.completed_at > cutoff_time
                ]

                if old_completed or old_failed:
                    logger.info(
                        "Cleaned up old requests",
                        completed_removed=len(old_completed),
                        failed_removed=len(old_failed),
                    )

                await asyncio.sleep(self.config.cleanup_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(self.config.cleanup_interval)

    async def _metrics_loop(self) -> None:
        """Metrics collection loop."""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Update metrics
                self.metrics.pending_requests = self.queue.qsize()
                self.metrics.processing_requests = len(self.active_requests)
                self.metrics.queue_utilization = (
                    self.metrics.processing_requests / self.config.max_concurrent_requests
                )

                # Calculate error rate
                total_processed = (
                    self.metrics.completed_requests
                    + self.metrics.failed_requests
                    + self.metrics.cancelled_requests
                    + self.metrics.timeout_requests
                )
                if total_processed > 0:
                    self.metrics.error_rate = (
                        self.metrics.failed_requests
                        + self.metrics.cancelled_requests
                        + self.metrics.timeout_requests
                    ) / total_processed

                # Calculate throughput (requests per minute)
                if self.metrics.completed_requests > 0:
                    elapsed_time = time.time() - self.metrics.timestamp
                    if elapsed_time > 0:
                        self.metrics.throughput = self.metrics.completed_requests / (
                            elapsed_time / 60
                        )

                # Update timestamp
                self.metrics.timestamp = time.time()

                # Store metrics history
                self.metrics_history.append(
                    QueueMetrics(
                        total_requests=self.metrics.total_requests,
                        pending_requests=self.metrics.pending_requests,
                        processing_requests=self.metrics.processing_requests,
                        completed_requests=self.metrics.completed_requests,
                        failed_requests=self.metrics.failed_requests,
                        cancelled_requests=self.metrics.cancelled_requests,
                        timeout_requests=self.metrics.timeout_requests,
                        average_processing_time=self.metrics.average_processing_time,
                        average_wait_time=self.metrics.average_wait_time,
                        queue_utilization=self.metrics.queue_utilization,
                        error_rate=self.metrics.error_rate,
                        throughput=self.metrics.throughput,
                        timestamp=self.metrics.timestamp,
                    )
                )

                # Keep only recent metrics
                cutoff_time = time.time() - 3600  # 1 hour
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp > cutoff_time
                ]

                await asyncio.sleep(30)  # Update every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics error: {e}")
                await asyncio.sleep(30)

    async def get_request_status(self, request_id: str) -> dict[str, Any]:
        """Get status of a specific request."""
        # Check active requests
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            return {
                "request_id": request_id,
                "status": request.status.value,
                "priority": request.priority.value,
                "created_at": request.created_at,
                "started_at": request.started_at,
                "processing_time": time.time() - request.started_at if request.started_at else None,
                "retry_count": request.retry_count,
                "metadata": request.metadata,
            }

        # Check completed requests
        if request_id in self.completed_requests:
            request = self.completed_requests[request_id]
            return {
                "request_id": request_id,
                "status": request.status.value,
                "priority": request.priority.value,
                "created_at": request.created_at,
                "started_at": request.started_at,
                "completed_at": request.completed_at,
                "processing_time": request.completed_at - request.started_at
                if request.started_at and request.completed_at
                else None,
                "retry_count": request.retry_count,
                "metadata": request.metadata,
                "result": request.result,
                "error": request.error,
            }

        # Check failed requests
        if request_id in self.failed_requests:
            request = self.failed_requests[request_id]
            return {
                "request_id": request_id,
                "status": request.status.value,
                "priority": request.priority.value,
                "created_at": request.created_at,
                "started_at": request.started_at,
                "completed_at": request.completed_at,
                "processing_time": request.completed_at - request.started_at
                if request.started_at and request.completed_at
                else None,
                "retry_count": request.retry_count,
                "metadata": request.metadata,
                "error": request.error,
            }

        return {"request_id": request_id, "status": "not_found"}

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending request."""
        # Check if request is in queue
        # Note: This is a simplified implementation
        # In a real implementation, you'd need to remove the request from the queue

        # Check if request is being processed
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            request.status = RequestStatus.CANCELLED
            request.completed_at = time.time()

            # Cancel the processing task
            if request_id in self.processing_tasks:
                self.processing_tasks[request_id].cancel()
                del self.processing_tasks[request_id]

            # Move to failed requests
            self.failed_requests[request_id] = request
            del self.active_requests[request_id]

            # Update metrics
            self.metrics.cancelled_requests += 1
            self.metrics.processing_requests -= 1

            logger.info("Request cancelled", request_id=request_id)
            return True

        return False

    def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status."""
        return {
            "is_running": self.is_running,
            "queue_size": self.queue.qsize(),
            "max_queue_size": self.config.max_queue_size,
            "active_requests": len(self.active_requests),
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "completed_requests": len(self.completed_requests),
            "failed_requests": len(self.failed_requests),
            "strategy": self.config.strategy.value,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "pending_requests": self.metrics.pending_requests,
                "processing_requests": self.metrics.processing_requests,
                "completed_requests": self.metrics.completed_requests,
                "failed_requests": self.metrics.failed_requests,
                "cancelled_requests": self.metrics.cancelled_requests,
                "timeout_requests": self.metrics.timeout_requests,
                "average_processing_time": self.metrics.average_processing_time,
                "queue_utilization": self.metrics.queue_utilization,
                "error_rate": self.metrics.error_rate,
                "throughput": self.metrics.throughput,
            },
        }

    def get_metrics_history(self) -> list[dict[str, Any]]:
        """Get metrics history."""
        return [
            {
                "timestamp": m.timestamp,
                "total_requests": m.total_requests,
                "pending_requests": m.pending_requests,
                "processing_requests": m.processing_requests,
                "completed_requests": m.completed_requests,
                "failed_requests": m.failed_requests,
                "cancelled_requests": m.cancelled_requests,
                "timeout_requests": m.timeout_requests,
                "average_processing_time": m.average_processing_time,
                "queue_utilization": m.queue_utilization,
                "error_rate": m.error_rate,
                "throughput": m.throughput,
            }
            for m in self.metrics_history
        ]


# Global queue instance
_vlm_request_queue: VLMRequestQueue | None = None


def get_vlm_request_queue() -> VLMRequestQueue:
    """Get global VLM request queue instance."""
    global _vlm_request_queue
    if _vlm_request_queue is None:
        config = QueueConfig()
        _vlm_request_queue = VLMRequestQueue(config)
    return _vlm_request_queue


def create_vlm_request_queue(config: QueueConfig) -> VLMRequestQueue:
    """Create a new VLM request queue instance."""
    return VLMRequestQueue(config)
