"""Model warm-up procedures for consistent performance.

This module provides comprehensive model warm-up procedures for VLM models,
ensuring consistent performance and reducing cold start latency.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import structlog

try:
    import pynvml
except ImportError:
    pynvml = None

logger = structlog.get_logger(__name__)


class WarmupStatus(Enum):
    """Warm-up status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class WarmupStrategy(Enum):
    """Warm-up strategy."""
    MINIMAL = "minimal"  # Basic warm-up with minimal requests
    STANDARD = "standard"  # Standard warm-up with diverse requests
    COMPREHENSIVE = "comprehensive"  # Comprehensive warm-up with full coverage
    CUSTOM = "custom"  # Custom warm-up based on specific requirements


@dataclass
class WarmupConfig:
    """Configuration for model warm-up."""
    enabled: bool = True
    strategy: WarmupStrategy = WarmupStrategy.STANDARD
    warmup_requests: int = 10
    warmup_timeout: int = 300  # 5 minutes
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    request_types: List[str] = field(default_factory=lambda: ["text", "table", "figure", "mixed"])
    gpu_memory_threshold: float = 0.8  # 80% GPU memory usage threshold
    temperature_threshold: float = 70.0  # 70°C temperature threshold
    performance_threshold: float = 0.9  # 90% of expected performance
    retry_attempts: int = 3
    retry_delay: int = 30  # 30 seconds between retries
    monitoring_interval: int = 5  # 5 seconds monitoring interval


@dataclass
class WarmupRequest:
    """Warm-up request definition."""
    request_id: str
    request_type: str
    batch_size: int
    content: bytes
    config: Dict[str, Any]
    options: Dict[str, Any]
    expected_duration: float
    priority: int = 1  # 1 = highest, 5 = lowest


@dataclass
class WarmupResult:
    """Result of warm-up request."""
    request_id: str
    success: bool
    duration: float
    gpu_memory_usage: float
    gpu_utilization: float
    gpu_temperature: float
    error_message: Optional[str] = None
    performance_score: float = 0.0


@dataclass
class WarmupMetrics:
    """Warm-up metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    gpu_memory_peak: float = 0.0
    gpu_utilization_peak: float = 0.0
    gpu_temperature_peak: float = 0.0
    performance_score: float = 0.0


class ModelWarmupManager:
    """Manager for model warm-up procedures."""

    def __init__(self, config: WarmupConfig):
        self.config = config
        self.status = WarmupStatus.NOT_STARTED
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.results: List[WarmupResult] = []
        self.metrics = WarmupMetrics()
        self.warmup_requests: List[WarmupRequest] = []

        # GPU monitoring
        self.gpu_available = False
        if pynvml:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                logger.info("GPU monitoring initialized for warm-up")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")

    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics."""
        if not self.gpu_available:
            return {
                "memory_usage": 0.0,
                "utilization": 0.0,
                "temperature": 0.0,
                "power_usage": 0.0
            }

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Memory usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_usage = memory_info.used / memory_info.total

            # Utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu / 100.0

            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            # Power usage
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts

            return {
                "memory_usage": memory_usage,
                "utilization": gpu_utilization,
                "temperature": temperature,
                "power_usage": power_usage
            }
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {e}")
            return {
                "memory_usage": 0.0,
                "utilization": 0.0,
                "temperature": 0.0,
                "power_usage": 0.0
            }

    def _create_warmup_requests(self) -> List[WarmupRequest]:
        """Create warm-up requests based on strategy."""
        requests = []

        if self.config.strategy == WarmupStrategy.MINIMAL:
            # Minimal warm-up with basic requests
            for i in range(min(3, self.config.warmup_requests)):
                requests.append(WarmupRequest(
                    request_id=f"minimal_{i}",
                    request_type="text",
                    batch_size=1,
                    content=b"Sample PDF content for warm-up",
                    config={"model_name": "gemma3-12b", "temperature": 0.1},
                    options={"enable_text_extraction": True},
                    expected_duration=5.0,
                    priority=1
                ))

        elif self.config.strategy == WarmupStrategy.STANDARD:
            # Standard warm-up with diverse requests
            request_types = ["text", "table", "figure", "mixed"]
            batch_sizes = [1, 2, 4]

            for i, request_type in enumerate(request_types):
                for j, batch_size in enumerate(batch_sizes):
                    if len(requests) >= self.config.warmup_requests:
                        break

                    requests.append(WarmupRequest(
                        request_id=f"standard_{request_type}_{batch_size}",
                        request_type=request_type,
                        batch_size=batch_size,
                        content=b"Sample PDF content for warm-up",
                        config={
                            "model_name": "gemma3-12b",
                            "temperature": 0.1,
                            "max_model_len": 4096
                        },
                        options={
                            "enable_text_extraction": True,
                            "enable_table_extraction": request_type in ["table", "mixed"],
                            "enable_figure_extraction": request_type in ["figure", "mixed"]
                        },
                        expected_duration=10.0 / batch_size,
                        priority=2
                    ))

        elif self.config.strategy == WarmupStrategy.COMPREHENSIVE:
            # Comprehensive warm-up with full coverage
            request_types = ["text", "table", "figure", "mixed", "complex"]
            batch_sizes = [1, 2, 4, 8]

            for i, request_type in enumerate(request_types):
                for j, batch_size in enumerate(batch_sizes):
                    if len(requests) >= self.config.warmup_requests:
                        break

                    requests.append(WarmupRequest(
                        request_id=f"comprehensive_{request_type}_{batch_size}",
                        request_type=request_type,
                        batch_size=batch_size,
                        content=b"Sample PDF content for warm-up",
                        config={
                            "model_name": "gemma3-12b",
                            "temperature": 0.1,
                            "max_model_len": 4096
                        },
                        options={
                            "enable_text_extraction": True,
                            "enable_table_extraction": True,
                            "enable_figure_extraction": True,
                            "enable_medical_normalization": True,
                            "enable_table_fidelity": True,
                            "enable_terminology_support": True
                        },
                        expected_duration=15.0 / batch_size,
                        priority=3
                    ))

        else:  # CUSTOM
            # Custom warm-up based on specific requirements
            for i in range(self.config.warmup_requests):
                requests.append(WarmupRequest(
                    request_id=f"custom_{i}",
                    request_type="mixed",
                    batch_size=self.config.batch_sizes[i % len(self.config.batch_sizes)],
                    content=b"Sample PDF content for warm-up",
                    config={"model_name": "gemma3-12b", "temperature": 0.1},
                    options={"enable_text_extraction": True},
                    expected_duration=10.0,
                    priority=4
                ))

        return requests[:self.config.warmup_requests]

    async def _execute_warmup_request(
        self,
        request: WarmupRequest,
        vlm_client: Any
    ) -> WarmupResult:
        """Execute a single warm-up request."""
        start_time = time.time()
        gpu_metrics_start = self._get_gpu_metrics()

        try:
            # Simulate VLM processing
            # In a real implementation, this would call the actual VLM service
            await asyncio.sleep(request.expected_duration)

            # Get GPU metrics after processing
            gpu_metrics_end = self._get_gpu_metrics()

            duration = time.time() - start_time

            # Calculate performance score based on duration vs expected
            performance_score = min(1.0, request.expected_duration / duration)

            result = WarmupResult(
                request_id=request.request_id,
                success=True,
                duration=duration,
                gpu_memory_usage=gpu_metrics_end["memory_usage"],
                gpu_utilization=gpu_metrics_end["utilization"],
                gpu_temperature=gpu_metrics_end["temperature"],
                performance_score=performance_score
            )

            logger.info(
                "Warm-up request completed",
                request_id=request.request_id,
                duration=duration,
                performance_score=performance_score
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            gpu_metrics_end = self._get_gpu_metrics()

            result = WarmupResult(
                request_id=request.request_id,
                success=False,
                duration=duration,
                gpu_memory_usage=gpu_metrics_end["memory_usage"],
                gpu_utilization=gpu_metrics_end["utilization"],
                gpu_temperature=gpu_metrics_end["temperature"],
                error_message=str(e),
                performance_score=0.0
            )

            logger.error(
                "Warm-up request failed",
                request_id=request.request_id,
                error=str(e)
            )

            return result

    async def _monitor_warmup(self) -> None:
        """Monitor warm-up progress."""
        while self.status == WarmupStatus.IN_PROGRESS:
            gpu_metrics = self._get_gpu_metrics()

            # Check for GPU memory threshold
            if gpu_metrics["memory_usage"] > self.config.gpu_memory_threshold:
                logger.warning(
                    "GPU memory usage exceeds threshold",
                    memory_usage=gpu_metrics["memory_usage"],
                    threshold=self.config.gpu_memory_threshold
                )

            # Check for GPU temperature threshold
            if gpu_metrics["temperature"] > self.config.temperature_threshold:
                logger.warning(
                    "GPU temperature exceeds threshold",
                    temperature=gpu_metrics["temperature"],
                    threshold=self.config.temperature_threshold
                )

            await asyncio.sleep(self.config.monitoring_interval)

    def _update_metrics(self) -> None:
        """Update warm-up metrics."""
        if not self.results:
            return

        self.metrics.total_requests = len(self.results)
        self.metrics.successful_requests = len([r for r in self.results if r.success])
        self.metrics.failed_requests = len([r for r in self.results if not r.success])

        if self.results:
            durations = [r.duration for r in self.results]
            self.metrics.total_duration = sum(durations)
            self.metrics.average_duration = self.metrics.total_duration / len(durations)
            self.metrics.min_duration = min(durations)
            self.metrics.max_duration = max(durations)

            # GPU metrics
            self.metrics.gpu_memory_peak = max(r.gpu_memory_usage for r in self.results)
            self.metrics.gpu_utilization_peak = max(r.gpu_utilization for r in self.results)
            self.metrics.gpu_temperature_peak = max(r.gpu_temperature for r in self.results)

            # Performance score
            performance_scores = [r.performance_score for r in self.results if r.success]
            if performance_scores:
                self.metrics.performance_score = sum(performance_scores) / len(performance_scores)

    async def warmup(self, vlm_client: Any) -> bool:
        """Execute model warm-up procedure."""
        if self.status == WarmupStatus.IN_PROGRESS:
            logger.warning("Warm-up already in progress")
            return False

        if not self.config.enabled:
            logger.info("Model warm-up disabled")
            return True

        self.status = WarmupStatus.IN_PROGRESS
        self.start_time = time.time()
        self.results.clear()
        self.metrics = WarmupMetrics()

        logger.info(
            "Starting model warm-up",
            strategy=self.config.strategy.value,
            warmup_requests=self.config.warmup_requests,
            timeout=self.config.warmup_timeout
        )

        try:
            # Create warm-up requests
            self.warmup_requests = self._create_warmup_requests()

            # Start monitoring
            monitor_task = asyncio.create_task(self._monitor_warmup())

            # Execute warm-up requests
            for attempt in range(self.config.retry_attempts):
                try:
                    # Execute requests in parallel
                    tasks = [
                        self._execute_warmup_request(request, vlm_client)
                        for request in self.warmup_requests
                    ]

                    # Wait for all requests to complete
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.config.warmup_timeout
                    )

                    # Process results
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"Warm-up request failed with exception: {result}")
                            continue
                        self.results.append(result)

                    # Check if warm-up was successful
                    successful_results = [r for r in self.results if r.success]
                    if len(successful_results) >= self.config.warmup_requests * 0.8:  # 80% success rate
                        self.status = WarmupStatus.COMPLETED
                        break
                    else:
                        logger.warning(f"Warm-up attempt {attempt + 1} failed, retrying...")
                        if attempt < self.config.retry_attempts - 1:
                            await asyncio.sleep(self.config.retry_delay)

                except asyncio.TimeoutError:
                    logger.error("Warm-up timeout")
                    self.status = WarmupStatus.TIMEOUT
                    break
                except Exception as e:
                    logger.error(f"Warm-up error: {e}")
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay)
                    else:
                        self.status = WarmupStatus.FAILED
                        break

            # Cancel monitoring task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Update metrics
            self._update_metrics()

            # Set end time
            self.end_time = time.time()

            # Log final status
            if self.status == WarmupStatus.COMPLETED:
                logger.info(
                    "Model warm-up completed successfully",
                    duration=self.end_time - self.start_time,
                    successful_requests=self.metrics.successful_requests,
                    total_requests=self.metrics.total_requests,
                    performance_score=self.metrics.performance_score
                )
            else:
                logger.error(
                    "Model warm-up failed",
                    status=self.status.value,
                    duration=self.end_time - self.start_time,
                    successful_requests=self.metrics.successful_requests,
                    total_requests=self.metrics.total_requests
                )

            return self.status == WarmupStatus.COMPLETED

        except Exception as e:
            logger.error(f"Warm-up procedure failed: {e}")
            self.status = WarmupStatus.FAILED
            self.end_time = time.time()
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current warm-up status."""
        return {
            "status": self.status.value,
            "enabled": self.config.enabled,
            "strategy": self.config.strategy.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time - self.start_time) if self.end_time and self.start_time else None,
            "metrics": self.metrics.__dict__,
            "gpu_available": self.gpu_available
        }

    def is_ready(self) -> bool:
        """Check if model is ready (warm-up completed successfully)."""
        return self.status == WarmupStatus.COMPLETED

    def reset(self) -> None:
        """Reset warm-up state."""
        self.status = WarmupStatus.NOT_STARTED
        self.start_time = None
        self.end_time = None
        self.results.clear()
        self.metrics = WarmupMetrics()
        self.warmup_requests.clear()

        logger.info("Model warm-up state reset")


# Global warm-up manager instance
_warmup_manager: Optional[ModelWarmupManager] = None


def get_warmup_manager() -> ModelWarmupManager:
    """Get global warm-up manager instance."""
    global _warmup_manager
    if _warmup_manager is None:
        config = WarmupConfig()
        _warmup_manager = ModelWarmupManager(config)
    return _warmup_manager


def create_warmup_manager(config: WarmupConfig) -> ModelWarmupManager:
    """Create a new warm-up manager instance."""
    return ModelWarmupManager(config)
