"""Batch size optimization for Gemma3 12B model processing.

This module provides dynamic batch size optimization for the Gemma3 12B model
based on GPU memory usage, processing latency, and throughput metrics.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

try:
    import pynvml
except ImportError:
    pynvml = None

logger = structlog.get_logger(__name__)


class OptimizationStrategy(Enum):
    """Batch size optimization strategy."""
    CONSERVATIVE = "conservative"  # Prioritize stability
    BALANCED = "balanced"  # Balance throughput and memory
    AGGRESSIVE = "aggressive"  # Maximize throughput


class BatchSizeState(Enum):
    """Batch size optimization state."""
    INITIALIZING = "initializing"
    OPTIMIZING = "optimizing"
    STABLE = "stable"
    DEGRADED = "degraded"


@dataclass
class BatchSizeConfig:
    """Configuration for batch size optimization."""
    min_batch_size: int = 1
    max_batch_size: int = 16
    initial_batch_size: int = 4
    memory_threshold: float = 0.85  # 85% GPU memory usage threshold
    latency_threshold: float = 30.0  # 30 seconds latency threshold
    throughput_threshold: float = 0.8  # 80% of peak throughput threshold
    optimization_interval: int = 300  # 5 minutes
    warmup_requests: int = 10
    stability_window: int = 5  # Number of stable measurements before considering optimized


@dataclass
class PerformanceMetrics:
    """Performance metrics for batch size optimization."""
    batch_size: int
    throughput: float  # requests per second
    latency: float  # average latency in seconds
    gpu_memory_usage: float  # percentage of GPU memory used
    gpu_utilization: float  # percentage of GPU utilization
    timestamp: float
    success_rate: float  # percentage of successful requests


@dataclass
class OptimizationResult:
    """Result of batch size optimization."""
    optimal_batch_size: int
    confidence: float  # 0.0 to 1.0
    metrics: PerformanceMetrics
    recommendation: str
    state: BatchSizeState


class Gemma3BatchSizeOptimizer:
    """Dynamic batch size optimizer for Gemma3 12B model."""

    def __init__(self, config: BatchSizeConfig):
        self.config = config
        self.current_batch_size = config.initial_batch_size
        self.state = BatchSizeState.INITIALIZING
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_history: List[OptimizationResult] = []
        self.last_optimization_time = 0.0
        self.stable_count = 0
        self.best_batch_size = config.initial_batch_size
        self.best_throughput = 0.0

        # Initialize GPU monitoring if available
        self.gpu_available = False
        if pynvml:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                logger.info("GPU monitoring initialized")
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

    async def collect_metrics(
        self,
        batch_size: int,
        processing_time: float,
        success: bool
    ) -> PerformanceMetrics:
        """Collect performance metrics for a batch size."""
        gpu_metrics = self._get_gpu_metrics()

        # Calculate throughput (requests per second)
        throughput = 1.0 / processing_time if processing_time > 0 else 0.0

        metrics = PerformanceMetrics(
            batch_size=batch_size,
            throughput=throughput,
            latency=processing_time,
            gpu_memory_usage=gpu_metrics["memory_usage"],
            gpu_utilization=gpu_metrics["utilization"],
            timestamp=time.time(),
            success_rate=1.0 if success else 0.0
        )

        self.metrics_history.append(metrics)

        # Keep only recent metrics (last hour)
        cutoff_time = time.time() - 3600
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]

        logger.info(
            "Collected metrics",
            batch_size=batch_size,
            throughput=throughput,
            latency=processing_time,
            gpu_memory_usage=gpu_metrics["memory_usage"],
            gpu_utilization=gpu_metrics["utilization"],
            success=success
        )

        return metrics

    def _calculate_optimal_batch_size(self) -> OptimizationResult:
        """Calculate optimal batch size based on collected metrics."""
        if not self.metrics_history:
            return OptimizationResult(
                optimal_batch_size=self.config.initial_batch_size,
                confidence=0.0,
                metrics=PerformanceMetrics(
                    batch_size=self.config.initial_batch_size,
                    throughput=0.0,
                    latency=0.0,
                    gpu_memory_usage=0.0,
                    gpu_utilization=0.0,
                    timestamp=time.time(),
                    success_rate=0.0
                ),
                recommendation="Insufficient data for optimization",
                state=BatchSizeState.INITIALIZING
            )

        # Group metrics by batch size
        batch_metrics: Dict[int, List[PerformanceMetrics]] = {}
        for metric in self.metrics_history:
            if metric.batch_size not in batch_metrics:
                batch_metrics[metric.batch_size] = []
            batch_metrics[metric.batch_size].append(metric)

        # Calculate average metrics for each batch size
        batch_averages: Dict[int, PerformanceMetrics] = {}
        for batch_size, metrics_list in batch_metrics.items():
            if len(metrics_list) < 3:  # Need at least 3 samples
                continue

            avg_throughput = sum(m.throughput for m in metrics_list) / len(metrics_list)
            avg_latency = sum(m.latency for m in metrics_list) / len(metrics_list)
            avg_memory_usage = sum(m.gpu_memory_usage for m in metrics_list) / len(metrics_list)
            avg_utilization = sum(m.gpu_utilization for m in metrics_list) / len(metrics_list)
            avg_success_rate = sum(m.success_rate for m in metrics_list) / len(metrics_list)

            batch_averages[batch_size] = PerformanceMetrics(
                batch_size=batch_size,
                throughput=avg_throughput,
                latency=avg_latency,
                gpu_memory_usage=avg_memory_usage,
                gpu_utilization=avg_utilization,
                timestamp=time.time(),
                success_rate=avg_success_rate
            )

        if not batch_averages:
            return OptimizationResult(
                optimal_batch_size=self.current_batch_size,
                confidence=0.0,
                metrics=self.metrics_history[-1] if self.metrics_history else PerformanceMetrics(
                    batch_size=self.current_batch_size,
                    throughput=0.0,
                    latency=0.0,
                    gpu_memory_usage=0.0,
                    gpu_utilization=0.0,
                    timestamp=time.time(),
                    success_rate=0.0
                ),
                recommendation="Insufficient data for optimization",
                state=BatchSizeState.INITIALIZING
            )

        # Find optimal batch size
        optimal_batch_size = self.current_batch_size
        best_score = 0.0
        best_metrics = None

        for batch_size, metrics in batch_averages.items():
            # Calculate composite score
            # Factors: throughput (40%), memory efficiency (30%), success rate (20%), latency (10%)
            throughput_score = min(metrics.throughput / 10.0, 1.0)  # Normalize to 0-1
            memory_score = 1.0 - metrics.gpu_memory_usage  # Lower memory usage is better
            success_score = metrics.success_rate
            latency_score = max(0.0, 1.0 - (metrics.latency / self.config.latency_threshold))

            composite_score = (
                0.4 * throughput_score +
                0.3 * memory_score +
                0.2 * success_score +
                0.1 * latency_score
            )

            # Penalize if memory usage is too high
            if metrics.gpu_memory_usage > self.config.memory_threshold:
                composite_score *= 0.5

            # Penalize if success rate is too low
            if metrics.success_rate < 0.95:
                composite_score *= 0.7

            if composite_score > best_score:
                best_score = composite_score
                optimal_batch_size = batch_size
                best_metrics = metrics

        # Determine state and recommendation
        if best_metrics:
            if best_metrics.gpu_memory_usage > self.config.memory_threshold:
                state = BatchSizeState.DEGRADED
                recommendation = "High memory usage detected, consider reducing batch size"
            elif best_metrics.success_rate < 0.95:
                state = BatchSizeState.DEGRADED
                recommendation = "Low success rate detected, check system stability"
            elif abs(optimal_batch_size - self.current_batch_size) <= 1:
                state = BatchSizeState.STABLE
                recommendation = "Batch size is near optimal"
            else:
                state = BatchSizeState.OPTIMIZING
                recommendation = f"Recommended batch size: {optimal_batch_size}"
        else:
            state = BatchSizeState.INITIALIZING
            recommendation = "Collecting more data for optimization"

        # Calculate confidence based on data quality
        confidence = min(len(self.metrics_history) / 100.0, 1.0)  # Max confidence at 100 samples

        result = OptimizationResult(
            optimal_batch_size=optimal_batch_size,
            confidence=confidence,
            metrics=best_metrics or self.metrics_history[-1],
            recommendation=recommendation,
            state=state
        )

        self.optimization_history.append(result)

        # Keep only recent optimization results
        cutoff_time = time.time() - 3600
        self.optimization_history = [
            r for r in self.optimization_history
            if r.metrics.timestamp > cutoff_time
        ]

        return result

    async def optimize_batch_size(self) -> OptimizationResult:
        """Optimize batch size based on current performance."""
        current_time = time.time()

        # Check if enough time has passed since last optimization
        if current_time - self.last_optimization_time < self.config.optimization_interval:
            return OptimizationResult(
                optimal_batch_size=self.current_batch_size,
                confidence=0.0,
                metrics=self.metrics_history[-1] if self.metrics_history else PerformanceMetrics(
                    batch_size=self.current_batch_size,
                    throughput=0.0,
                    latency=0.0,
                    gpu_memory_usage=0.0,
                    gpu_utilization=0.0,
                    timestamp=time.time(),
                    success_rate=0.0
                ),
                recommendation="Optimization interval not reached",
                state=self.state
            )

        # Calculate optimal batch size
        result = self._calculate_optimal_batch_size()

        # Update state
        self.state = result.state

        # Apply optimization if confidence is high enough
        if result.confidence > 0.7 and result.optimal_batch_size != self.current_batch_size:
            old_batch_size = self.current_batch_size
            self.current_batch_size = result.optimal_batch_size
            self.last_optimization_time = current_time

            logger.info(
                "Batch size optimized",
                old_batch_size=old_batch_size,
                new_batch_size=self.current_batch_size,
                confidence=result.confidence,
                recommendation=result.recommendation
            )

        return result

    def get_current_batch_size(self) -> int:
        """Get current optimized batch size."""
        return self.current_batch_size

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        latest_result = self.optimization_history[-1] if self.optimization_history else None

        return {
            "current_batch_size": self.current_batch_size,
            "state": self.state.value,
            "metrics_count": len(self.metrics_history),
            "optimization_count": len(self.optimization_history),
            "last_optimization_time": self.last_optimization_time,
            "latest_result": {
                "optimal_batch_size": latest_result.optimal_batch_size if latest_result else None,
                "confidence": latest_result.confidence if latest_result else None,
                "recommendation": latest_result.recommendation if latest_result else None,
                "state": latest_result.state.value if latest_result else None
            } if latest_result else None,
            "gpu_available": self.gpu_available
        }

    def reset_optimization(self) -> None:
        """Reset optimization state."""
        self.current_batch_size = self.config.initial_batch_size
        self.state = BatchSizeState.INITIALIZING
        self.metrics_history.clear()
        self.optimization_history.clear()
        self.last_optimization_time = 0.0
        self.stable_count = 0
        self.best_batch_size = self.config.initial_batch_size
        self.best_throughput = 0.0

        logger.info("Batch size optimization reset")

    def get_recommended_batch_sizes(self) -> Dict[str, int]:
        """Get recommended batch sizes for different scenarios."""
        current_metrics = self.metrics_history[-1] if self.metrics_history else None

        if not current_metrics:
            return {
                "conservative": self.config.min_batch_size,
                "balanced": self.config.initial_batch_size,
                "aggressive": min(self.config.max_batch_size, self.config.initial_batch_size * 2)
            }

        # Conservative: prioritize stability
        conservative = max(
            self.config.min_batch_size,
            min(self.current_batch_size, 4)
        )

        # Balanced: current optimized size
        balanced = self.current_batch_size

        # Aggressive: try larger batch size if memory allows
        aggressive = min(
            self.config.max_batch_size,
            self.current_batch_size + 2
        )

        return {
            "conservative": conservative,
            "balanced": balanced,
            "aggressive": aggressive
        }


# Global optimizer instance
_optimizer: Optional[Gemma3BatchSizeOptimizer] = None


def get_batch_size_optimizer() -> Gemma3BatchSizeOptimizer:
    """Get global batch size optimizer instance."""
    global _optimizer
    if _optimizer is None:
        config = BatchSizeConfig()
        _optimizer = Gemma3BatchSizeOptimizer(config)
    return _optimizer


def create_batch_size_optimizer(config: BatchSizeConfig) -> Gemma3BatchSizeOptimizer:
    """Create a new batch size optimizer instance."""
    return Gemma3BatchSizeOptimizer(config)
