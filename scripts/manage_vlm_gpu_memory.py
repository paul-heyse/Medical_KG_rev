#!/usr/bin/env python3
"""CLI script to manage VLM GPU memory optimization.

This script provides commands to monitor, optimize, and manage GPU memory
usage for VLM models.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from Medical_KG_rev.services.optimization.vlm_gpu_memory_optimizer import (
        MemoryConfig,
        MemoryOptimizationStrategy,
        MemoryStatus,
        VLMGPUMemoryOptimizer,
        create_vlm_memory_optimizer,
        get_vlm_memory_optimizer,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

app = typer.Typer(
    name="manage-vlm-gpu-memory", help="Manage VLM GPU memory optimization", rich_markup_mode="rich"
)
console = Console()


def display_memory_status(status: dict[str, Any]) -> None:
    """Display memory status in a formatted table."""
    table = Table(title="GPU Memory Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Status
    status_color = (
        "red"
        if status["status"] in ["critical", "oom"]
        else "yellow"
        if status["status"] == "warning"
        else "green"
    )
    table.add_row("Status", Text(status["status"].upper(), style=status_color))

    # GPU availability
    gpu_available = "Yes" if status["gpu_available"] else "No"
    table.add_row("GPU Available", gpu_available)

    # Batch size
    table.add_row("Current Batch Size", str(status["current_batch_size"]))

    # Optimization status
    optimization_status = "Enabled" if status["optimization_enabled"] else "Disabled"
    table.add_row("Optimization", optimization_status)

    if status["gpu_available"] and "memory_usage_percent" in status:
        # Memory usage
        memory_usage = status["memory_usage_percent"] * 100
        memory_color = "red" if memory_usage > 90 else "yellow" if memory_usage > 80 else "green"
        table.add_row("Memory Usage", f"{memory_usage:.1f}%", style=memory_color)

        # Memory details
        table.add_row("Used Memory", f"{status['memory_usage_mb']:.1f} MB")
        table.add_row("Total Memory", f"{status['total_memory_mb']:.1f} MB")
        table.add_row("Free Memory", f"{status['free_memory_mb']:.1f} MB")

        # GPU metrics
        table.add_row("GPU Utilization", f"{status['gpu_utilization'] * 100:.1f}%")
        table.add_row("GPU Temperature", f"{status['gpu_temperature']:.1f}°C")
        table.add_row("GPU Power Usage", f"{status['gpu_power_usage']:.1f} W")

        # Fragmentation
        table.add_row("Memory Fragmentation", f"{status['memory_fragmentation'] * 100:.1f}%")

    console.print(table)


def display_optimization_history(history: list[dict[str, Any]]) -> None:
    """Display optimization history in a formatted table."""
    if not history:
        console.print("No optimization history available")
        return

    table = Table(title="Optimization History")
    table.add_column("Action", style="cyan")
    table.add_column("Success", style="green")
    table.add_column("Memory Freed", style="yellow")
    table.add_column("New Batch Size", style="blue")
    table.add_column("GC Performed", style="magenta")
    table.add_column("Error", style="red")

    for entry in history[-10:]:  # Show last 10 entries
        success = "✓" if entry["success"] else "✗"
        success_style = "green" if entry["success"] else "red"

        table.add_row(
            entry["action_taken"],
            Text(success, style=success_style),
            f"{entry['memory_freed_mb']} MB",
            str(entry["new_batch_size"]),
            "Yes" if entry["gc_performed"] else "No",
            entry["error_message"] or "-",
        )

    console.print(table)


@app.command()
def status() -> None:
    """Show current GPU memory status."""
    console.print(Panel("GPU Memory Status", style="bold blue"))

    optimizer = get_vlm_memory_optimizer()
    status_info = optimizer.get_memory_status()

    display_memory_status(status_info)


@app.command()
def monitor(
    duration: int = typer.Option(60, "--duration", "-d", help="Monitoring duration in seconds"),
    interval: int = typer.Option(5, "--interval", "-i", help="Update interval in seconds"),
) -> None:
    """Monitor GPU memory usage in real-time."""
    console.print(Panel("Real-time GPU Memory Monitoring", style="bold blue"))

    optimizer = get_vlm_memory_optimizer()

    async def monitor_loop() -> None:
        start_time = asyncio.get_event_loop().time()

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Monitoring GPU memory...", total=None)

            while (asyncio.get_event_loop().time() - start_time) < duration:
                status_info = optimizer.get_memory_status()

                # Clear previous output
                console.clear()
                console.print(Panel("Real-time GPU Memory Monitoring", style="bold blue"))

                # Display current status
                display_memory_status(status_info)

                # Display optimization history
                history = optimizer.get_optimization_history()
                if history:
                    console.print("\n")
                    display_optimization_history(history)

                await asyncio.sleep(interval)

    asyncio.run(monitor_loop())


@app.command()
def optimize() -> None:
    """Perform immediate memory optimization."""
    console.print(Panel("Performing Memory Optimization", style="bold blue"))

    optimizer = get_vlm_memory_optimizer()

    async def run_optimization() -> None:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Optimizing memory...", total=None)

            result = await optimizer.optimize_memory()

            progress.stop()

            if result.success:
                console.print("✓ Optimization completed successfully", style="green")
                console.print(f"Action taken: {result.action_taken}")
                console.print(f"Memory freed: {result.memory_freed_mb} MB")
                console.print(f"New batch size: {result.new_batch_size}")
                console.print(f"GC performed: {'Yes' if result.gc_performed else 'No'}")
                console.print(
                    f"Fragmentation reduced: {'Yes' if result.fragmentation_reduced else 'No'}"
                )
            else:
                console.print("✗ Optimization failed", style="red")
                console.print(f"Error: {result.error_message}")

    asyncio.run(run_optimization())


@app.command()
def set_batch_size(batch_size: int = typer.Argument(..., help="New batch size")) -> None:
    """Set the batch size manually."""
    console.print(Panel("Setting Batch Size", style="bold blue"))

    optimizer = get_vlm_memory_optimizer()

    if optimizer.set_batch_size(batch_size):
        console.print(f"✓ Batch size set to {batch_size}", style="green")
    else:
        console.print(f"✗ Failed to set batch size to {batch_size}", style="red")
        console.print("Make sure the batch size is within the configured limits")


@app.command()
def enable() -> None:
    """Enable memory optimization."""
    console.print(Panel("Enabling Memory Optimization", style="bold blue"))

    optimizer = get_vlm_memory_optimizer()
    optimizer.enable_optimization()

    console.print("✓ Memory optimization enabled", style="green")


@app.command()
def disable() -> None:
    """Disable memory optimization."""
    console.print(Panel("Disabling Memory Optimization", style="bold blue"))

    optimizer = get_vlm_memory_optimizer()
    optimizer.disable_optimization()

    console.print("✓ Memory optimization disabled", style="yellow")


@app.command()
def start_monitoring() -> None:
    """Start background memory monitoring."""
    console.print(Panel("Starting Background Monitoring", style="bold blue"))

    optimizer = get_vlm_memory_optimizer()

    async def start_monitoring_async() -> None:
        await optimizer.start_monitoring()
        console.print("✓ Background monitoring started", style="green")
        console.print("Monitoring will continue in the background")

    asyncio.run(start_monitoring_async())


@app.command()
def stop_monitoring() -> None:
    """Stop background memory monitoring."""
    console.print(Panel("Stopping Background Monitoring", style="bold blue"))

    optimizer = get_vlm_memory_optimizer()

    async def stop_monitoring_async() -> None:
        await optimizer.stop_monitoring()
        console.print("✓ Background monitoring stopped", style="green")

    asyncio.run(stop_monitoring_async())


@app.command()
def configure(
    strategy: str = typer.Option("balanced", "--strategy", "-s", help="Optimization strategy"),
    warning_threshold: float = typer.Option(
        0.8, "--warning-threshold", "-w", help="Warning threshold (0.0-1.0)"
    ),
    critical_threshold: float = typer.Option(
        0.9, "--critical-threshold", "-c", help="Critical threshold (0.0-1.0)"
    ),
    oom_threshold: float = typer.Option(
        0.95, "--oom-threshold", "-o", help="OOM threshold (0.0-1.0)"
    ),
    max_batch_size: int = typer.Option(16, "--max-batch-size", "-m", help="Maximum batch size"),
    min_batch_size: int = typer.Option(1, "--min-batch-size", "-n", help="Minimum batch size"),
    monitoring_interval: int = typer.Option(
        5, "--monitoring-interval", "-i", help="Monitoring interval in seconds"
    ),
    optimization_interval: int = typer.Option(
        60, "--optimization-interval", "-p", help="Optimization interval in seconds"
    ),
) -> None:
    """Configure memory optimization settings."""
    console.print(Panel("Configuring Memory Optimization", style="bold blue"))

    # Validate strategy
    try:
        strategy_enum = MemoryOptimizationStrategy(strategy.lower())
    except ValueError:
        console.print(f"✗ Invalid strategy: {strategy}", style="red")
        console.print("Valid strategies: conservative, balanced, aggressive, adaptive")
        return

    # Validate thresholds
    if not (0.0 <= warning_threshold <= 1.0):
        console.print("✗ Warning threshold must be between 0.0 and 1.0", style="red")
        return

    if not (0.0 <= critical_threshold <= 1.0):
        console.print("✗ Critical threshold must be between 0.0 and 1.0", style="red")
        return

    if not (0.0 <= oom_threshold <= 1.0):
        console.print("✗ OOM threshold must be between 0.0 and 1.0", style="red")
        return

    if warning_threshold >= critical_threshold:
        console.print("✗ Warning threshold must be less than critical threshold", style="red")
        return

    if critical_threshold >= oom_threshold:
        console.print("✗ Critical threshold must be less than OOM threshold", style="red")
        return

    # Validate batch sizes
    if min_batch_size > max_batch_size:
        console.print(
            "✗ Minimum batch size must be less than or equal to maximum batch size", style="red"
        )
        return

    # Create new configuration
    config = MemoryConfig(
        strategy=strategy_enum,
        memory_threshold_warning=warning_threshold,
        memory_threshold_critical=critical_threshold,
        memory_threshold_oom=oom_threshold,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        monitoring_interval=monitoring_interval,
        optimization_interval=optimization_interval,
    )

    # Create new optimizer with configuration
    optimizer = create_vlm_memory_optimizer(config)

    console.print("✓ Configuration updated successfully", style="green")
    console.print(f"Strategy: {strategy}")
    console.print(f"Warning threshold: {warning_threshold * 100:.1f}%")
    console.print(f"Critical threshold: {critical_threshold * 100:.1f}%")
    console.print(f"OOM threshold: {oom_threshold * 100:.1f}%")
    console.print(f"Batch size range: {min_batch_size}-{max_batch_size}")
    console.print(f"Monitoring interval: {monitoring_interval}s")
    console.print(f"Optimization interval: {optimization_interval}s")


@app.command()
def history() -> None:
    """Show optimization history."""
    console.print(Panel("Optimization History", style="bold blue"))

    optimizer = get_vlm_memory_optimizer()
    history = optimizer.get_optimization_history()

    display_optimization_history(history)


@app.command()
def export_config(
    output_file: str = typer.Option(
        "vlm_memory_config.json", "--output", "-o", help="Output file path"
    ),
) -> None:
    """Export current configuration to JSON file."""
    console.print(Panel("Exporting Configuration", style="bold blue"))

    optimizer = get_vlm_memory_optimizer()
    config = optimizer.config

    config_dict = {
        "enabled": config.enabled,
        "strategy": config.strategy.value,
        "monitoring_interval": config.monitoring_interval,
        "memory_threshold_warning": config.memory_threshold_warning,
        "memory_threshold_critical": config.memory_threshold_critical,
        "memory_threshold_oom": config.memory_threshold_oom,
        "temperature_threshold": config.temperature_threshold,
        "optimization_interval": config.optimization_interval,
        "cleanup_interval": config.cleanup_interval,
        "max_batch_size": config.max_batch_size,
        "min_batch_size": config.min_batch_size,
        "memory_reserve_mb": config.memory_reserve_mb,
        "gc_threshold": config.gc_threshold,
        "fragmentation_threshold": config.fragmentation_threshold,
    }

    try:
        with open(output_file, "w") as f:
            json.dump(config_dict, f, indent=2)

        console.print(f"✓ Configuration exported to {output_file}", style="green")
    except Exception as e:
        console.print(f"✗ Failed to export configuration: {e}", style="red")


@app.command()
def import_config(input_file: str = typer.Argument(..., help="Input file path")) -> None:
    """Import configuration from JSON file."""
    console.print(Panel("Importing Configuration", style="bold blue"))

    try:
        with open(input_file) as f:
            config_dict = json.load(f)

        # Create configuration from dict
        config = MemoryConfig(
            enabled=config_dict.get("enabled", True),
            strategy=MemoryOptimizationStrategy(config_dict.get("strategy", "balanced")),
            monitoring_interval=config_dict.get("monitoring_interval", 5),
            memory_threshold_warning=config_dict.get("memory_threshold_warning", 0.8),
            memory_threshold_critical=config_dict.get("memory_threshold_critical", 0.9),
            memory_threshold_oom=config_dict.get("memory_threshold_oom", 0.95),
            temperature_threshold=config_dict.get("temperature_threshold", 80.0),
            optimization_interval=config_dict.get("optimization_interval", 60),
            cleanup_interval=config_dict.get("cleanup_interval", 300),
            max_batch_size=config_dict.get("max_batch_size", 16),
            min_batch_size=config_dict.get("min_batch_size", 1),
            memory_reserve_mb=config_dict.get("memory_reserve_mb", 1024),
            gc_threshold=config_dict.get("gc_threshold", 0.85),
            fragmentation_threshold=config_dict.get("fragmentation_threshold", 0.3),
        )

        # Create new optimizer with imported configuration
        optimizer = create_vlm_memory_optimizer(config)

        console.print(f"✓ Configuration imported from {input_file}", style="green")
        console.print(f"Strategy: {config.strategy.value}")
        console.print(f"Warning threshold: {config.memory_threshold_warning * 100:.1f}%")
        console.print(f"Critical threshold: {config.memory_threshold_critical * 100:.1f}%")
        console.print(f"OOM threshold: {config.memory_threshold_oom * 100:.1f}%")
        console.print(f"Batch size range: {config.min_batch_size}-{config.max_batch_size}")

    except FileNotFoundError:
        console.print(f"✗ File not found: {input_file}", style="red")
    except json.JSONDecodeError:
        console.print(f"✗ Invalid JSON file: {input_file}", style="red")
    except Exception as e:
        console.print(f"✗ Failed to import configuration: {e}", style="red")


@app.command()
def benchmark(
    duration: int = typer.Option(300, "--duration", "-d", help="Benchmark duration in seconds"),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file for results"
    ),
) -> None:
    """Run a benchmark to test memory optimization performance."""
    console.print(Panel("Running Memory Optimization Benchmark", style="bold blue"))

    optimizer = get_vlm_memory_optimizer()

    async def run_benchmark() -> None:
        start_time = asyncio.get_event_loop().time()
        benchmark_data = []

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Running benchmark...", total=duration)

            while (asyncio.get_event_loop().time() - start_time) < duration:
                # Get current status
                status_info = optimizer.get_memory_status()

                # Perform optimization
                result = await optimizer.optimize_memory()

                # Record benchmark data
                benchmark_data.append(
                    {
                        "timestamp": asyncio.get_event_loop().time() - start_time,
                        "memory_usage_percent": status_info.get("memory_usage_percent", 0.0),
                        "batch_size": status_info["current_batch_size"],
                        "optimization_success": result.success,
                        "action_taken": result.action_taken,
                        "memory_freed_mb": result.memory_freed_mb,
                    }
                )

                await asyncio.sleep(10)  # Sample every 10 seconds

        progress.stop()

        # Calculate benchmark results
        total_optimizations = len(benchmark_data)
        successful_optimizations = sum(1 for d in benchmark_data if d["optimization_success"])
        avg_memory_usage = sum(d["memory_usage_percent"] for d in benchmark_data) / len(
            benchmark_data
        )
        total_memory_freed = sum(d["memory_freed_mb"] for d in benchmark_data)

        console.print("✓ Benchmark completed", style="green")
        console.print(f"Duration: {duration} seconds")
        console.print(f"Total optimizations: {total_optimizations}")
        console.print(f"Successful optimizations: {successful_optimizations}")
        console.print(f"Success rate: {successful_optimizations / total_optimizations * 100:.1f}%")
        console.print(f"Average memory usage: {avg_memory_usage * 100:.1f}%")
        console.print(f"Total memory freed: {total_memory_freed} MB")

        # Export results if requested
        if output_file:
            try:
                with open(output_file, "w") as f:
                    json.dump(benchmark_data, f, indent=2)
                console.print(f"✓ Benchmark results exported to {output_file}", style="green")
            except Exception as e:
                console.print(f"✗ Failed to export results: {e}", style="red")

    asyncio.run(run_benchmark())


if __name__ == "__main__":
    app()
