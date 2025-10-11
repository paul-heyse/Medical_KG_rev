#!/usr/bin/env python3
"""GPU Resource Optimization Script.

This script provides command-line tools for optimizing GPU resource
utilization across GPU services.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.Medical_KG_rev.services.optimization.gpu_resource_optimizer import (
    GPUResourceOptimizer,
)

logger = logging.getLogger(__name__)


class GPUOptimizationCLI:
    """Command-line interface for GPU resource optimization."""

    def __init__(self) -> None:
        """Initialize GPU optimization CLI."""
        self.optimizer = GPUResourceOptimizer()

    async def show_status(self) -> None:
        """Show current GPU resource status."""
        print("=== GPU Resource Status ===")

        # Get optimization status
        status = self.optimizer.get_optimization_status()

        # Display GPU information
        print("\nGPU Devices:")
        for gpu_info in status["gpu_info"]:
            print(f"  GPU {gpu_info['device_id']}: {gpu_info['name']}")
            print(
                f"    Memory: {gpu_info['memory_used_mb']:,}MB / {gpu_info['memory_total_mb']:,}MB "
                f"({gpu_info['memory_utilization_percent']:.1f}%)"
            )
            print(f"    Utilization: {gpu_info['gpu_utilization_percent']:.1f}%")
            print(f"    Temperature: {gpu_info['temperature_celsius']}°C")
            print(f"    Power: {gpu_info['power_usage_watts']}W")
            print()

        # Display allocation status
        allocation_status = status["allocation_status"]
        print(f"Active Allocations: {allocation_status['total_allocations']}")
        for allocation in allocation_status["allocations"]:
            print(
                f"  {allocation['service_name']} -> GPU {allocation['device_id']} "
                f"(required: {allocation['required_memory_mb']}MB)"
            )

        # Display memory report
        memory_report = status["memory_report"]
        print("\nMemory Summary:")
        print(f"  Total Memory: {memory_report['total_memory_mb']:,}MB")
        print(f"  Used Memory: {memory_report['total_used_mb']:,}MB")
        print(f"  Free Memory: {memory_report['total_free_mb']:,}MB")
        print(f"  Average Utilization: {memory_report['average_utilization_percent']:.1f}%")

    async def optimize_memory(self, service_name: str) -> None:
        """Optimize memory usage for a specific service.

        Args:
        ----
            service_name: Name of the service to optimize

        """
        print(f"Optimizing memory usage for service: {service_name}")

        try:
            results = await self.optimizer.memory_optimizer.optimize_memory_usage(service_name)

            print("Optimization Results:")
            print(f"  Service: {results['service_name']}")
            print(f"  Memory Freed: {results['memory_freed_mb']}MB")
            print(f"  Optimizations Applied: {len(results['optimizations_applied'])}")

            for optimization in results["optimizations_applied"]:
                print(
                    f"    GPU {optimization['device_id']}: "
                    f"Freed {optimization['memory_freed_mb']}MB "
                    f"({optimization['optimization_type']})"
                )

        except Exception as e:
            logger.error(f"Error optimizing memory for service {service_name}: {e}")
            print(f"Error: {e}")

    async def allocate_gpu(
        self, service_name: str, required_memory_mb: int, preferred_device_id: int | None = None
    ) -> None:
        """Allocate a GPU for a service.

        Args:
        ----
            service_name: Name of the service
            required_memory_mb: Required memory in MB
            preferred_device_id: Preferred GPU device ID

        """
        print(f"Allocating GPU for service: {service_name}")
        print(f"Required Memory: {required_memory_mb}MB")
        if preferred_device_id is not None:
            print(f"Preferred Device: GPU {preferred_device_id}")

        try:
            device_id = await self.optimizer.allocation_manager.allocate_gpu(
                service_name, required_memory_mb, preferred_device_id
            )

            if device_id is not None:
                print(f"Successfully allocated GPU {device_id}")
            else:
                print("Failed to allocate GPU - no suitable device available")

        except Exception as e:
            logger.error(f"Error allocating GPU: {e}")
            print(f"Error: {e}")

    async def deallocate_gpu(self, service_name: str, device_id: int) -> None:
        """Deallocate a GPU from a service.

        Args:
        ----
            service_name: Name of the service
            device_id: GPU device ID to deallocate

        """
        print(f"Deallocating GPU {device_id} from service: {service_name}")

        try:
            success = await self.optimizer.allocation_manager.deallocate_gpu(
                service_name, device_id
            )

            if success:
                print("Successfully deallocated GPU")
            else:
                print("Failed to deallocate GPU - allocation not found")

        except Exception as e:
            logger.error(f"Error deallocating GPU: {e}")
            print(f"Error: {e}")

    async def run_optimization_loop(self, interval: int = 300) -> None:
        """Run continuous optimization loop.

        Args:
        ----
            interval: Optimization interval in seconds

        """
        print(f"Starting GPU resource optimization loop (interval: {interval}s)")
        print("Press Ctrl+C to stop")

        try:
            await self.optimizer.start_optimization_loop(interval)
        except KeyboardInterrupt:
            self.optimizer.stop_optimization_loop()
            print("\nOptimization loop stopped")

    def export_status(self, output_file: str) -> None:
        """Export current status to JSON file.

        Args:
        ----
            output_file: Output file path

        """
        print(f"Exporting status to: {output_file}")

        try:
            status = self.optimizer.get_optimization_status()

            Path(output_file).write_text(json.dumps(status, indent=2, default=str))

            print("Status exported successfully")

        except Exception as e:
            logger.error(f"Error exporting status: {e}")
            print(f"Error: {e}")

    def generate_report(self) -> None:
        """Generate a comprehensive GPU resource report."""
        print("=== GPU Resource Report ===")

        status = self.optimizer.get_optimization_status()

        # System overview
        print("\nSystem Overview:")
        print(f"  Total GPUs: {len(status['gpu_info'])}")
        print(f"  Active Allocations: {status['allocation_status']['total_allocations']}")
        print(f"  Optimization Running: {status['running']}")

        # Memory analysis
        memory_report = status["memory_report"]
        print("\nMemory Analysis:")
        print(f"  Total Memory: {memory_report['total_memory_mb']:,}MB")
        print(f"  Used Memory: {memory_report['total_used_mb']:,}MB")
        print(f"  Free Memory: {memory_report['total_free_mb']:,}MB")
        print(f"  Utilization: {memory_report['average_utilization_percent']:.1f}%")

        # GPU details
        print("\nGPU Details:")
        for gpu_info in status["gpu_info"]:
            print(f"  GPU {gpu_info['device_id']}: {gpu_info['name']}")
            print(
                f"    Memory: {gpu_info['memory_used_mb']:,}MB / {gpu_info['memory_total_mb']:,}MB "
                f"({gpu_info['memory_utilization_percent']:.1f}%)"
            )
            print(f"    Utilization: {gpu_info['gpu_utilization_percent']:.1f}%")
            print(f"    Temperature: {gpu_info['temperature_celsius']}°C")
            print(f"    Power: {gpu_info['power_usage_watts']}W")

        # Allocation details
        print("\nAllocation Details:")
        for allocation in status["allocation_status"]["allocations"]:
            print(
                f"  {allocation['service_name']} -> GPU {allocation['device_id']} "
                f"(required: {allocation['required_memory_mb']}MB)"
            )

        # Recommendations
        print("\nRecommendations:")
        if memory_report["average_utilization_percent"] > 80:
            print("  ⚠️  High memory utilization detected - consider memory optimization")
        if memory_report["average_utilization_percent"] < 20:
            print("  ℹ️  Low memory utilization - consider consolidating services")
        if status["allocation_status"]["total_allocations"] == 0:
            print("  ℹ️  No active allocations - all GPUs are available")
        else:
            print(f"  ℹ️  {status['allocation_status']['total_allocations']} active allocations")


def main() -> None:
    """Run the main function."""
    parser = argparse.ArgumentParser(description="GPU Resource Optimization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    subparsers.add_parser("status", help="Show GPU resource status")

    # Optimize memory command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize memory usage")
    optimize_parser.add_argument("service", help="Service name to optimize")

    # Allocate GPU command
    allocate_parser = subparsers.add_parser("allocate", help="Allocate GPU for service")
    allocate_parser.add_argument("service", help="Service name")
    allocate_parser.add_argument("memory", type=int, help="Required memory in MB")
    allocate_parser.add_argument("--device", type=int, help="Preferred GPU device ID")

    # Deallocate GPU command
    deallocate_parser = subparsers.add_parser("deallocate", help="Deallocate GPU from service")
    deallocate_parser.add_argument("service", help="Service name")
    deallocate_parser.add_argument("device", type=int, help="GPU device ID")

    # Run optimization loop command
    loop_parser = subparsers.add_parser("loop", help="Run optimization loop")
    loop_parser.add_argument("--interval", type=int, default=300, help="Interval in seconds")

    # Export status command
    export_parser = subparsers.add_parser("export", help="Export status to JSON")
    export_parser.add_argument("output", help="Output file path")

    # Generate report command
    subparsers.add_parser("report", help="Generate comprehensive report")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create CLI
    cli = GPUOptimizationCLI()

    try:
        if args.command == "status":
            asyncio.run(cli.show_status())
        elif args.command == "optimize":
            asyncio.run(cli.optimize_memory(args.service))
        elif args.command == "allocate":
            asyncio.run(cli.allocate_gpu(args.service, args.memory, args.device))
        elif args.command == "deallocate":
            asyncio.run(cli.deallocate_gpu(args.service, args.device))
        elif args.command == "loop":
            asyncio.run(cli.run_optimization_loop(args.interval))
        elif args.command == "export":
            cli.export_status(args.output)
        elif args.command == "report":
            cli.generate_report()
        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
