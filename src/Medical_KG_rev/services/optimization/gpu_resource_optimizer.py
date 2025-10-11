"""GPU Resource Optimizer for GPU Services.

This module provides GPU resource optimization capabilities including
monitoring, allocation, and memory optimization for GPU services.
"""

import asyncio
import logging
import time
from typing import Any

import pynvml

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class GPUResourceMonitor:
    """Monitors GPU resource utilization and performance."""

    def __init__(self) -> None:
        """Initialize GPU resource monitor."""
        self._initialize_nvidia_ml()
        self._setup_metrics()

    def _initialize_nvidia_ml(self) -> None:
        """Initialize NVIDIA ML library."""
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"Initialized NVIDIA ML with {self.device_count} devices")
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA ML: {e}")
            self.device_count = 0

    def _setup_metrics(self) -> None:
        """Set up Prometheus metrics for GPU monitoring."""
        # GPU Utilization Metrics
        self.gpu_utilization = Gauge(
            "gpu_utilization_percent", "GPU utilization percentage", ["device_id", "service_name"]
        )

        self.gpu_memory_usage = Gauge(
            "gpu_memory_usage_mb", "GPU memory usage in MB", ["device_id", "service_name"]
        )

        self.gpu_memory_total = Gauge(
            "gpu_memory_total_mb", "Total GPU memory in MB", ["device_id", "service_name"]
        )

        self.gpu_memory_free = Gauge(
            "gpu_memory_free_mb", "Free GPU memory in MB", ["device_id", "service_name"]
        )

        self.gpu_temperature = Gauge(
            "gpu_temperature_celsius", "GPU temperature in Celsius", ["device_id", "service_name"]
        )

        self.gpu_power_usage = Gauge(
            "gpu_power_usage_watts", "GPU power usage in watts", ["device_id", "service_name"]
        )

        # Performance Metrics
        self.gpu_throughput = Counter(
            "gpu_throughput_ops_total",
            "Total GPU operations",
            ["device_id", "service_name", "operation_type"],
        )

        self.gpu_latency = Histogram(
            "gpu_operation_latency_seconds",
            "GPU operation latency",
            ["device_id", "service_name", "operation_type"],
        )

        # Resource Allocation Metrics
        self.gpu_allocation_requests = Counter(
            "gpu_allocation_requests_total",
            "Total GPU allocation requests",
            ["device_id", "service_name", "status"],
        )

        self.gpu_allocation_duration = Histogram(
            "gpu_allocation_duration_seconds",
            "GPU allocation duration",
            ["device_id", "service_name"],
        )

    def get_gpu_info(self, device_id: int) -> dict[str, Any]:
        """Get comprehensive GPU information.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary with GPU information

        """
        if device_id >= self.device_count:
            return {}

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # Basic info
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            uuid = pynvml.nvmlDeviceGetUUID(handle).decode("utf-8")

            # Memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = memory_info.total // (1024 * 1024)
            memory_used = memory_info.used // (1024 * 1024)
            memory_free = memory_info.free // (1024 * 1024)

            # Utilization info
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            memory_util = utilization.memory

            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temperature = 0

            # Power usage
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000
            except Exception:
                power_usage = 0

            # Clock speeds
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                graphics_clock = 0
                memory_clock = 0

            return {
                "device_id": device_id,
                "name": name,
                "uuid": uuid,
                "memory_total_mb": memory_total,
                "memory_used_mb": memory_used,
                "memory_free_mb": memory_free,
                "memory_utilization_percent": memory_util,
                "gpu_utilization_percent": gpu_util,
                "temperature_celsius": temperature,
                "power_usage_watts": power_usage,
                "graphics_clock_mhz": graphics_clock,
                "memory_clock_mhz": memory_clock,
            }

        except Exception as e:
            logger.error(f"Error getting GPU info for device {device_id}: {e}")
            return {}

    def get_all_gpu_info(self) -> list[dict[str, Any]]:
        """Get information for all GPU devices.

        Returns:
            List of GPU information dictionaries

        """
        gpu_info: list[dict[str, Any]] = []
        for device_id in range(self.device_count):
            info = self.get_gpu_info(device_id)
            if info:
                gpu_info.append(info)
        return gpu_info

    def update_metrics(self, service_name: str) -> None:
        """Update Prometheus metrics for a service.

        Args:
            service_name: Name of the service

        """
        for device_id in range(self.device_count):
            info = self.get_gpu_info(device_id)
            if not info:
                continue

            # Update utilization metrics
            self.gpu_utilization.labels(device_id=str(device_id), service_name=service_name).set(
                info["gpu_utilization_percent"]
            )

            # Update memory metrics
            self.gpu_memory_usage.labels(device_id=str(device_id), service_name=service_name).set(
                info["memory_used_mb"]
            )

            self.gpu_memory_total.labels(device_id=str(device_id), service_name=service_name).set(
                info["memory_total_mb"]
            )

            self.gpu_memory_free.labels(device_id=str(device_id), service_name=service_name).set(
                info["memory_free_mb"]
            )

            # Update temperature and power metrics
            self.gpu_temperature.labels(device_id=str(device_id), service_name=service_name).set(
                info["temperature_celsius"]
            )

            self.gpu_power_usage.labels(device_id=str(device_id), service_name=service_name).set(
                info["power_usage_watts"]
            )


class GPUAllocationManager:
    """Manages GPU allocation and deallocation for services."""

    def __init__(self, monitor: GPUResourceMonitor) -> None:
        """Initialize GPU allocation manager.

        Args:
            monitor: GPU resource monitor instance

        """
        self.monitor = monitor
        self.allocations: dict[str, dict[str, Any]] = {}
        self._allocation_lock = asyncio.Lock()

    async def allocate_gpu(
        self, service_name: str, required_memory_mb: int, preferred_device_id: int | None = None
    ) -> int | None:
        """Allocate a GPU for a service.

        Args:
            service_name: Name of the service requesting allocation
            required_memory_mb: Required memory in MB
            preferred_device_id: Preferred GPU device ID

        Returns:
            Allocated GPU device ID or None if allocation failed

        """
        async with self._allocation_lock:
            start_time = time.time()

            try:
                # Get available GPUs
                available_gpus = await self._get_available_gpus(required_memory_mb)

                if not available_gpus:
                    logger.warning(f"No available GPUs for service {service_name}")
                    return None

                # Select best GPU
                selected_gpu = self._select_best_gpu(available_gpus, preferred_device_id)

                if selected_gpu is None:
                    logger.warning(f"No suitable GPU found for service {service_name}")
                    return None

                # Record allocation
                allocation_id = f"{service_name}_{int(time.time())}"
                self.allocations[allocation_id] = {
                    "service_name": service_name,
                    "device_id": selected_gpu,
                    "allocated_at": time.time(),
                    "required_memory_mb": required_memory_mb,
                }

                # Update metrics
                self.monitor.gpu_allocation_requests.labels(
                    device_id=str(selected_gpu), service_name=service_name, status="success"
                ).inc()

                duration = time.time() - start_time
                self.monitor.gpu_allocation_duration.labels(
                    device_id=str(selected_gpu), service_name=service_name
                ).observe(duration)

                logger.info(
                    f"Allocated GPU {selected_gpu} to service {service_name} "
                    f"(required: {required_memory_mb}MB)"
                )

                return selected_gpu

            except Exception as e:
                logger.error(f"Error allocating GPU for service {service_name}: {e}")

                # Update metrics for failed allocation
                if preferred_device_id is not None:
                    self.monitor.gpu_allocation_requests.labels(
                        device_id=str(preferred_device_id),
                        service_name=service_name,
                        status="failed",
                    ).inc()

                return None

    async def deallocate_gpu(self, service_name: str, device_id: int) -> bool:
        """Deallocate a GPU from a service.

        Args:
            service_name: Name of the service
            device_id: GPU device ID to deallocate

        Returns:
            True if deallocation successful, False otherwise

        """
        async with self._allocation_lock:
            try:
                # Find and remove allocation
                allocation_to_remove = None
                for allocation_id, allocation in self.allocations.items():
                    if (
                        allocation["service_name"] == service_name
                        and allocation["device_id"] == device_id
                    ):
                        allocation_to_remove = allocation_id
                        break

                if allocation_to_remove:
                    del self.allocations[allocation_to_remove]
                    logger.info(f"Deallocated GPU {device_id} from service {service_name}")
                    return True
                else:
                    logger.warning(
                        f"No allocation found for service {service_name} " f"on GPU {device_id}"
                    )
                    return False

            except Exception as e:
                logger.error(f"Error deallocating GPU {device_id}: {e}")
                return False

    async def _get_available_gpus(self, required_memory_mb: int) -> list[dict[str, Any]]:
        """Get list of available GPUs with sufficient memory.

        Args:
            required_memory_mb: Required memory in MB

        Returns:
            List of available GPU information

        """
        available_gpus: list[dict[str, Any]] = []

        for device_id in range(self.monitor.device_count):
            gpu_info = self.monitor.get_gpu_info(device_id)
            if not gpu_info:
                continue

            # Check if GPU has sufficient memory
            if gpu_info["memory_free_mb"] >= required_memory_mb:
                # Check if GPU is not already allocated
                is_allocated = any(
                    allocation["device_id"] == device_id for allocation in self.allocations.values()
                )

                if not is_allocated:
                    available_gpus.append(gpu_info)

        return available_gpus

    def _select_best_gpu(
        self, available_gpus: list[dict[str, Any]], preferred_device_id: int | None
    ) -> int | None:
        """Select the best GPU from available options.

        Args:
            available_gpus: List of available GPU information
            preferred_device_id: Preferred GPU device ID

        Returns:
            Selected GPU device ID or None

        """
        if not available_gpus:
            return None

        # If preferred device is available, use it
        if preferred_device_id is not None:
            for gpu_info in available_gpus:
                if gpu_info["device_id"] == preferred_device_id:
                    return preferred_device_id

        # Select GPU with highest free memory
        best_gpu = max(available_gpus, key=lambda gpu: gpu["memory_free_mb"])
        return best_gpu["device_id"]

    def get_allocation_status(self) -> dict[str, Any]:
        """Get current allocation status.

        Returns:
            Dictionary with allocation status

        """
        return {
            "total_allocations": len(self.allocations),
            "allocations": list(self.allocations.values()),
        }


class GPUMemoryOptimizer:
    """Optimizes GPU memory usage for services."""

    def __init__(self, monitor: GPUResourceMonitor) -> None:
        """Initialize GPU memory optimizer.

        Args:
            monitor: GPU resource monitor instance

        """
        self.monitor = monitor
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.optimization_interval = 300  # 5 minutes

    async def optimize_memory_usage(self, service_name: str) -> dict[str, Any]:
        """Optimize GPU memory usage for a service.

        Args:
            service_name: Name of the service

        Returns:
            Dictionary with optimization results

        """
        optimization_results: dict[str, Any] = {
            "service_name": service_name,
            "optimizations_applied": [],
            "memory_freed_mb": 0,
            "timestamp": time.time(),
        }

        try:
            for device_id in range(self.monitor.device_count):
                gpu_info = self.monitor.get_gpu_info(device_id)
                if not gpu_info:
                    continue

                memory_utilization = gpu_info["memory_used_mb"] / gpu_info["memory_total_mb"]

                if memory_utilization > self.memory_threshold:
                    # Apply memory optimization
                    freed_memory = await self._apply_memory_optimization(device_id, service_name)

                    if freed_memory > 0:
                        optimization_results["optimizations_applied"].append(
                            {
                                "device_id": device_id,
                                "memory_freed_mb": freed_memory,
                                "optimization_type": "memory_cleanup",
                            }
                        )
                        optimization_results["memory_freed_mb"] += freed_memory

        except Exception as e:
            logger.error(f"Error optimizing memory for service {service_name}: {e}")

        return optimization_results

    async def _apply_memory_optimization(self, device_id: int, service_name: str) -> int:
        """Apply memory optimization for a specific GPU.

        Args:
            device_id: GPU device ID
            service_name: Name of the service

        Returns:
            Amount of memory freed in MB

        """
        try:
            # Get initial memory usage
            initial_info = self.monitor.get_gpu_info(device_id)
            if not initial_info:
                return 0

            initial_memory = initial_info["memory_used_mb"]

            # Apply memory optimization (this would be service-specific)
            # For now, we'll simulate the optimization
            await asyncio.sleep(0.1)  # Simulate optimization time

            # Get final memory usage
            final_info = self.monitor.get_gpu_info(device_id)
            if not final_info:
                return 0

            final_memory = final_info["memory_used_mb"]
            freed_memory = initial_memory - final_memory

            if freed_memory > 0:
                logger.info(
                    f"Freed {freed_memory}MB of memory on GPU {device_id} "
                    f"for service {service_name}"
                )

            return max(0, freed_memory)

        except Exception as e:
            logger.error(f"Error applying memory optimization on GPU {device_id}: {e}")
            return 0

    def get_memory_usage_report(self) -> dict[str, Any]:
        """Get comprehensive memory usage report.

        Returns:
            Dictionary with memory usage report

        """
        report: dict[str, Any] = {
            "timestamp": time.time(),
            "gpus": [],
            "total_memory_mb": 0,
            "total_used_mb": 0,
            "total_free_mb": 0,
            "average_utilization_percent": 0,
        }

        total_memory = 0
        total_used = 0
        total_free = 0
        gpu_count = 0

        for device_id in range(self.monitor.device_count):
            gpu_info = self.monitor.get_gpu_info(device_id)
            if not gpu_info:
                continue

            gpu_report = {
                "device_id": device_id,
                "name": gpu_info["name"],
                "memory_total_mb": gpu_info["memory_total_mb"],
                "memory_used_mb": gpu_info["memory_used_mb"],
                "memory_free_mb": gpu_info["memory_free_mb"],
                "memory_utilization_percent": gpu_info["memory_utilization_percent"],
                "gpu_utilization_percent": gpu_info["gpu_utilization_percent"],
                "temperature_celsius": gpu_info["temperature_celsius"],
                "power_usage_watts": gpu_info["power_usage_watts"],
            }

            report["gpus"].append(gpu_report)

            total_memory += gpu_info["memory_total_mb"]
            total_used += gpu_info["memory_used_mb"]
            total_free += gpu_info["memory_free_mb"]
            gpu_count += 1

        report["total_memory_mb"] = total_memory
        report["total_used_mb"] = total_used
        report["total_free_mb"] = total_free

        if gpu_count > 0:
            report["average_utilization_percent"] = (total_used / total_memory) * 100

        return report


class GPUResourceOptimizer:
    """Main GPU resource optimizer coordinating all optimization activities."""

    def __init__(self) -> None:
        """Initialize GPU resource optimizer."""
        self.monitor = GPUResourceMonitor()
        self.allocation_manager = GPUAllocationManager(self.monitor)
        self.memory_optimizer = GPUMemoryOptimizer(self.monitor)
        self._running = False

    async def start_optimization_loop(self, interval: int = 300) -> None:
        """Start the optimization loop.

        Args:
            interval: Optimization interval in seconds

        """
        self._running = True

        while self._running:
            try:
                # Update metrics for all services
                self.monitor.update_metrics("gpu-optimizer")

                # Get memory usage report
                memory_report = self.memory_optimizer.get_memory_usage_report()

                # Log optimization status
                logger.info(
                    f"GPU Resource Optimization Status: "
                    f"Total Memory: {memory_report['total_memory_mb']}MB, "
                    f"Used: {memory_report['total_used_mb']}MB, "
                    f"Free: {memory_report['total_free_mb']}MB, "
                    f"Utilization: {memory_report['average_utilization_percent']:.1f}%"
                )

                # Apply memory optimization if needed
                if memory_report["average_utilization_percent"] > 80:
                    optimization_results = await self.memory_optimizer.optimize_memory_usage(
                        "gpu-optimizer"
                    )

                    if optimization_results["memory_freed_mb"] > 0:
                        logger.info(
                            f"Memory optimization freed {optimization_results['memory_freed_mb']}MB"
                        )

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(interval)

    def stop_optimization_loop(self) -> None:
        """Stop the optimization loop."""
        self._running = False

    def get_optimization_status(self) -> dict[str, Any]:
        """Get current optimization status.

        Returns:
            Dictionary with optimization status

        """
        return {
            "running": self._running,
            "allocation_status": self.allocation_manager.get_allocation_status(),
            "memory_report": self.memory_optimizer.get_memory_usage_report(),
            "gpu_info": self.monitor.get_all_gpu_info(),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Resource Optimizer")
    parser.add_argument(
        "--interval", type=int, default=300, help="Optimization interval in seconds"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create and start optimizer
    optimizer = GPUResourceOptimizer()

    try:
        asyncio.run(optimizer.start_optimization_loop(args.interval))
    except KeyboardInterrupt:
        optimizer.stop_optimization_loop()
        print("GPU Resource Optimizer stopped")
