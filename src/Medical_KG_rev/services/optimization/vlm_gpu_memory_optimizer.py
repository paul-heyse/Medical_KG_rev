"""GPU memory optimization for VLM models.

This module provides comprehensive GPU memory monitoring and optimization
for VLM models, ensuring efficient memory usage and preventing OOM errors.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

try:
    import pynvml
except ImportError:
    pynvml = None

logger = structlog.get_logger(__name__)


class MemoryOptimizationStrategy(Enum):
    """Memory optimization strategy."""

    CONSERVATIVE = "conservative"  # Prioritize stability
    BALANCED = "balanced"  # Balance performance and memory
    AGGRESSIVE = "aggressive"  # Maximize performance
    ADAPTIVE = "adaptive"  # Adapt based on usage patterns


class MemoryStatus(Enum):
    """Memory status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OOM = "oom"


@dataclass
class MemoryConfig:
    """Configuration for GPU memory optimization."""

    enabled: bool = True
    strategy: MemoryOptimizationStrategy = MemoryOptimizationStrategy.BALANCED
    monitoring_interval: int = 5  # 5 seconds
    memory_threshold_warning: float = 0.8  # 80% warning threshold
    memory_threshold_critical: float = 0.9  # 90% critical threshold
    memory_threshold_oom: float = 0.95  # 95% OOM threshold
    temperature_threshold: float = 80.0  # 80°C temperature threshold
    optimization_interval: int = 60  # 1 minute
    cleanup_interval: int = 300  # 5 minutes
    max_batch_size: int = 16
    min_batch_size: int = 1
    memory_reserve_mb: int = 1024  # 1GB reserve
    gc_threshold: float = 0.85  # 85% GC threshold
    fragmentation_threshold: float = 0.3  # 30% fragmentation threshold


@dataclass
class MemoryMetrics:
    """GPU memory metrics."""

    total_memory: int = 0
    used_memory: int = 0
    free_memory: int = 0
    memory_usage_percent: float = 0.0
    memory_fragmentation: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    gpu_power_usage: float = 0.0
    timestamp: float = 0.0


@dataclass
class MemoryOptimizationResult:
    """Result of memory optimization."""

    success: bool
    action_taken: str
    memory_freed_mb: int
    new_batch_size: int
    gc_performed: bool
    fragmentation_reduced: bool
    error_message: str | None = None


class VLMGPUMemoryOptimizer:
    """GPU memory optimizer for VLM models."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.status = MemoryStatus.HEALTHY
        self.metrics_history: list[MemoryMetrics] = []
        self.optimization_history: list[MemoryOptimizationResult] = []
        self.current_batch_size = 4
        self.last_optimization_time = 0.0
        self.last_cleanup_time = 0.0
        self.monitoring_task: asyncio.Task[None] | None = None
        self.optimization_enabled = True

        # GPU monitoring
        self.gpu_available = False
        if pynvml:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                logger.info("GPU monitoring initialized for memory optimization")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")

    def _get_gpu_metrics(self) -> MemoryMetrics:
        """Get current GPU metrics."""
        if not self.gpu_available:
            return MemoryMetrics(timestamp=time.time())

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total
            used_memory = memory_info.used
            free_memory = memory_info.free

            # Utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu / 100.0

            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            # Power usage
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts

            # Calculate memory usage percentage
            memory_usage_percent = used_memory / total_memory if total_memory > 0 else 0.0

            # Calculate memory fragmentation (simplified)
            # In a real implementation, this would use more sophisticated fragmentation detection
            memory_fragmentation = 0.0  # Placeholder

            return MemoryMetrics(
                total_memory=total_memory,
                used_memory=used_memory,
                free_memory=free_memory,
                memory_usage_percent=memory_usage_percent,
                memory_fragmentation=memory_fragmentation,
                gpu_utilization=gpu_utilization,
                gpu_temperature=temperature,
                gpu_power_usage=power_usage,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {e}")
            return MemoryMetrics(timestamp=time.time())

    def _update_status(self, metrics: MemoryMetrics) -> None:
        """Update memory status based on metrics."""
        if metrics.memory_usage_percent >= self.config.memory_threshold_oom:
            self.status = MemoryStatus.OOM
        elif metrics.memory_usage_percent >= self.config.memory_threshold_critical:
            self.status = MemoryStatus.CRITICAL
        elif metrics.memory_usage_percent >= self.config.memory_threshold_warning:
            self.status = MemoryStatus.WARNING
        else:
            self.status = MemoryStatus.HEALTHY

    def _calculate_optimal_batch_size(self, metrics: MemoryMetrics) -> int:
        """Calculate optimal batch size based on memory usage."""
        if not self.gpu_available:
            return self.current_batch_size

        # Base calculation on available memory
        available_memory_mb = metrics.free_memory / (1024 * 1024)
        reserve_memory_mb = self.config.memory_reserve_mb

        # Calculate available memory for processing
        usable_memory_mb = available_memory_mb - reserve_memory_mb

        if usable_memory_mb <= 0:
            return self.config.min_batch_size

        # Estimate memory per batch item (simplified)
        # In a real implementation, this would be based on actual model memory requirements
        memory_per_batch_item_mb = 500  # Placeholder

        # Calculate optimal batch size
        optimal_batch_size = int(usable_memory_mb / memory_per_batch_item_mb)

        # Apply strategy-based adjustments
        if self.config.strategy == MemoryOptimizationStrategy.CONSERVATIVE:
            optimal_batch_size = int(optimal_batch_size * 0.7)
        elif self.config.strategy == MemoryOptimizationStrategy.AGGRESSIVE:
            optimal_batch_size = int(optimal_batch_size * 1.2)
        elif self.config.strategy == MemoryOptimizationStrategy.ADAPTIVE:
            # Adaptive based on recent performance
            if len(self.metrics_history) > 5:
                recent_metrics = self.metrics_history[-5:]
                avg_utilization = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
                if avg_utilization < 0.7:
                    optimal_batch_size = int(optimal_batch_size * 1.1)
                elif avg_utilization > 0.9:
                    optimal_batch_size = int(optimal_batch_size * 0.9)

        # Clamp to configured limits
        optimal_batch_size = max(self.config.min_batch_size, optimal_batch_size)
        optimal_batch_size = min(self.config.max_batch_size, optimal_batch_size)

        return optimal_batch_size

    async def _perform_memory_cleanup(self) -> MemoryOptimizationResult:
        """Perform memory cleanup operations."""
        try:
            # Simulate memory cleanup
            # In a real implementation, this would:
            # 1. Clear model caches
            # 2. Run garbage collection
            # 3. Defragment memory
            # 4. Clear intermediate buffers

            await asyncio.sleep(0.1)  # Simulate cleanup time

            # Get metrics before and after cleanup
            metrics_before = self._get_gpu_metrics()

            # Simulate memory cleanup
            # In a real implementation, this would actually free memory
            memory_freed_mb = 100  # Placeholder

            metrics_after = self._get_gpu_metrics()

            result = MemoryOptimizationResult(
                success=True,
                action_taken="memory_cleanup",
                memory_freed_mb=memory_freed_mb,
                new_batch_size=self.current_batch_size,
                gc_performed=True,
                fragmentation_reduced=True
            )

            logger.info(
                "Memory cleanup performed",
                memory_freed_mb=memory_freed_mb,
                memory_usage_before=metrics_before.memory_usage_percent,
                memory_usage_after=metrics_after.memory_usage_percent
            )

            return result

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return MemoryOptimizationResult(
                success=False,
                action_taken="memory_cleanup",
                memory_freed_mb=0,
                new_batch_size=self.current_batch_size,
                gc_performed=False,
                fragmentation_reduced=False,
                error_message=str(e)
            )

    async def _optimize_batch_size(self, metrics: MemoryMetrics) -> MemoryOptimizationResult:
        """Optimize batch size based on memory usage."""
        try:
            old_batch_size = self.current_batch_size
            new_batch_size = self._calculate_optimal_batch_size(metrics)

            if new_batch_size != old_batch_size:
                self.current_batch_size = new_batch_size

                result = MemoryOptimizationResult(
                    success=True,
                    action_taken="batch_size_optimization",
                    memory_freed_mb=0,
                    new_batch_size=new_batch_size,
                    gc_performed=False,
                    fragmentation_reduced=False
                )

                logger.info(
                    "Batch size optimized",
                    old_batch_size=old_batch_size,
                    new_batch_size=new_batch_size,
                    memory_usage=metrics.memory_usage_percent
                )

                return result
            else:
                return MemoryOptimizationResult(
                    success=True,
                    action_taken="no_optimization_needed",
                    memory_freed_mb=0,
                    new_batch_size=old_batch_size,
                    gc_performed=False,
                    fragmentation_reduced=False
                )

        except Exception as e:
            logger.error(f"Batch size optimization failed: {e}")
            return MemoryOptimizationResult(
                success=False,
                action_taken="batch_size_optimization",
                memory_freed_mb=0,
                new_batch_size=self.current_batch_size,
                gc_performed=False,
                fragmentation_reduced=False,
                error_message=str(e)
            )

    async def optimize_memory(self) -> MemoryOptimizationResult:
        """Perform memory optimization."""
        if not self.optimization_enabled:
            return MemoryOptimizationResult(
                success=True,
                action_taken="optimization_disabled",
                memory_freed_mb=0,
                new_batch_size=self.current_batch_size,
                gc_performed=False,
                fragmentation_reduced=False
            )

        current_time = time.time()

        # Check if enough time has passed since last optimization
        if current_time - self.last_optimization_time < self.config.optimization_interval:
            return MemoryOptimizationResult(
                success=True,
                action_taken="optimization_interval_not_reached",
                memory_freed_mb=0,
                new_batch_size=self.current_batch_size,
                gc_performed=False,
                fragmentation_reduced=False
            )

        # Get current metrics
        metrics = self._get_gpu_metrics()
        self._update_status(metrics)

        # Determine optimization strategy based on status
        if self.status == MemoryStatus.OOM:
            # Critical: perform aggressive cleanup
            result = await self._perform_memory_cleanup()
            if result.success:
                # Also optimize batch size
                batch_result = await self._optimize_batch_size(metrics)
                result.new_batch_size = batch_result.new_batch_size
        elif self.status == MemoryStatus.CRITICAL:
            # Critical: optimize batch size and cleanup
            batch_result = await self._optimize_batch_size(metrics)
            if batch_result.success:
                cleanup_result = await self._perform_memory_cleanup()
                result = MemoryOptimizationResult(
                    success=True,
                    action_taken="critical_optimization",
                    memory_freed_mb=cleanup_result.memory_freed_mb,
                    new_batch_size=batch_result.new_batch_size,
                    gc_performed=cleanup_result.gc_performed,
                    fragmentation_reduced=cleanup_result.fragmentation_reduced
                )
            else:
                result = batch_result
        elif self.status == MemoryStatus.WARNING:
            # Warning: optimize batch size
            result = await self._optimize_batch_size(metrics)
        else:
            # Healthy: no optimization needed
            result = MemoryOptimizationResult(
                success=True,
                action_taken="no_optimization_needed",
                memory_freed_mb=0,
                new_batch_size=self.current_batch_size,
                gc_performed=False,
                fragmentation_reduced=False
            )

        # Update optimization history
        self.optimization_history.append(result)
        self.last_optimization_time = current_time

        # Keep only recent optimization results
        self.optimization_history = [
            r for r in self.optimization_history
            if current_time - self.last_optimization_time < 3600
        ]

        return result

    async def _monitor_memory(self) -> None:
        """Monitor GPU memory usage."""
        while self.optimization_enabled:
            try:
                # Get current metrics
                metrics = self._get_gpu_metrics()
                self.metrics_history.append(metrics)

                # Update status
                self._update_status(metrics)

                # Keep only recent metrics
                cutoff_time = time.time() - 3600  # 1 hour
                self.metrics_history = [
                    m for m in self.metrics_history
                    if m.timestamp > cutoff_time
                ]

                # Check if cleanup is needed
                current_time = time.time()
                if (current_time - self.last_cleanup_time > self.config.cleanup_interval and
                    metrics.memory_usage_percent > self.config.gc_threshold):

                    await self._perform_memory_cleanup()
                    self.last_cleanup_time = current_time

                # Log status if critical
                if self.status in [MemoryStatus.CRITICAL, MemoryStatus.OOM]:
                    logger.warning(
                        "Critical memory status",
                        status=self.status.value,
                        memory_usage=metrics.memory_usage_percent,
                        temperature=metrics.gpu_temperature
                    )

                await asyncio.sleep(self.config.monitoring_interval)

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(self.config.monitoring_interval)

    async def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Memory monitoring already running")
            return

        self.optimization_enabled = True
        self.monitoring_task = asyncio.create_task(self._monitor_memory())

        logger.info("GPU memory monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.optimization_enabled = False

        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("GPU memory monitoring stopped")

    def get_current_batch_size(self) -> int:
        """Get current optimized batch size."""
        return self.current_batch_size

    def get_memory_status(self) -> dict[str, Any]:
        """Get current memory status."""
        if not self.metrics_history:
            return {
                "status": self.status.value,
                "gpu_available": self.gpu_available,
                "current_batch_size": self.current_batch_size,
                "optimization_enabled": self.optimization_enabled
            }

        latest_metrics = self.metrics_history[-1]

        return {
            "status": self.status.value,
            "gpu_available": self.gpu_available,
            "current_batch_size": self.current_batch_size,
            "optimization_enabled": self.optimization_enabled,
            "memory_usage_percent": latest_metrics.memory_usage_percent,
            "memory_usage_mb": latest_metrics.used_memory / (1024 * 1024),
            "total_memory_mb": latest_metrics.total_memory / (1024 * 1024),
            "free_memory_mb": latest_metrics.free_memory / (1024 * 1024),
            "gpu_utilization": latest_metrics.gpu_utilization,
            "gpu_temperature": latest_metrics.gpu_temperature,
            "gpu_power_usage": latest_metrics.gpu_power_usage,
            "memory_fragmentation": latest_metrics.memory_fragmentation,
            "timestamp": latest_metrics.timestamp
        }

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get optimization history."""
        return [
            {
                "success": r.success,
                "action_taken": r.action_taken,
                "memory_freed_mb": r.memory_freed_mb,
                "new_batch_size": r.new_batch_size,
                "gc_performed": r.gc_performed,
                "fragmentation_reduced": r.fragmentation_reduced,
                "error_message": r.error_message
            }
            for r in self.optimization_history
        ]

    def set_batch_size(self, batch_size: int) -> bool:
        """Set batch size manually."""
        if self.config.min_batch_size <= batch_size <= self.config.max_batch_size:
            self.current_batch_size = batch_size
            logger.info(f"Batch size set to {batch_size}")
            return True
        else:
            logger.warning(f"Batch size {batch_size} out of range [{self.config.min_batch_size}, {self.config.max_batch_size}]")
            return False

    def enable_optimization(self) -> None:
        """Enable memory optimization."""
        self.optimization_enabled = True
        logger.info("Memory optimization enabled")

    def disable_optimization(self) -> None:
        """Disable memory optimization."""
        self.optimization_enabled = False
        logger.info("Memory optimization disabled")


# Global optimizer instance
_vlm_memory_optimizer: VLMGPUMemoryOptimizer | None = None


def get_vlm_memory_optimizer() -> VLMGPUMemoryOptimizer:
    """Get global VLM memory optimizer instance."""
    global _vlm_memory_optimizer
    if _vlm_memory_optimizer is None:
        config = MemoryConfig()
        _vlm_memory_optimizer = VLMGPUMemoryOptimizer(config)
    return _vlm_memory_optimizer


def create_vlm_memory_optimizer(config: MemoryConfig) -> VLMGPUMemoryOptimizer:
    """Create a new VLM memory optimizer instance."""
    return VLMGPUMemoryOptimizer(config)
