#!/usr/bin/env python3
"""CLI script for optimizing Gemma3 12B batch sizes.

This script provides command-line interface for managing batch size optimization
for the Gemma3 12B model in the Docling VLM service.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import the optimizer
try:
    from src.Medical_KG_rev.services.optimization.batch_size_optimizer import (
        BatchSizeConfig,
        Gemma3BatchSizeOptimizer,
        get_batch_size_optimizer,
    )
except ImportError:
    # Fallback for development
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.Medical_KG_rev.services.optimization.batch_size_optimizer import (
        BatchSizeConfig,
        Gemma3BatchSizeOptimizer,
        get_batch_size_optimizer,
    )

console = Console()
app = typer.Typer()


@app.command()
def status() -> None:
    """Show current batch size optimization status."""
    optimizer = get_batch_size_optimizer()
    status_info = optimizer.get_optimization_status()

    table = Table(title="Gemma3 Batch Size Optimization Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Current Batch Size", str(status_info["current_batch_size"]))
    table.add_row("State", status_info["state"])
    table.add_row("Metrics Collected", str(status_info["metrics_count"]))
    table.add_row("Optimizations Performed", str(status_info["optimization_count"]))
    table.add_row("GPU Monitoring", "Available" if status_info["gpu_available"] else "Not Available")

    if status_info["latest_result"]:
        latest = status_info["latest_result"]
        table.add_row("Optimal Batch Size", str(latest["optimal_batch_size"]))
        table.add_row("Confidence", f"{latest['confidence']:.2f}")
        table.add_row("Recommendation", latest["recommendation"])

    console.print(table)


@app.command()
def optimize(
    min_batch_size: int = typer.Option(1, help="Minimum batch size"),
    max_batch_size: int = typer.Option(16, help="Maximum batch size"),
    initial_batch_size: int = typer.Option(4, help="Initial batch size"),
    memory_threshold: float = typer.Option(0.85, help="GPU memory usage threshold"),
    latency_threshold: float = typer.Option(30.0, help="Latency threshold in seconds"),
    optimization_interval: int = typer.Option(300, help="Optimization interval in seconds")
) -> None:
    """Run batch size optimization."""
    config = BatchSizeConfig(
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        initial_batch_size=initial_batch_size,
        memory_threshold=memory_threshold,
        latency_threshold=latency_threshold,
        optimization_interval=optimization_interval
    )

    optimizer = Gemma3BatchSizeOptimizer(config)

    async def _optimize():
        with Progress(
            SpinnerColumn(),
            TextColumn("Optimizing batch size..."),
            console=console
        ) as progress:
            task = progress.add_task("Optimizing", total=None)

            # Simulate optimization process
            for i in range(10):
                await asyncio.sleep(1)

                # Simulate metrics collection
                await optimizer.collect_metrics(
                    batch_size=config.initial_batch_size + i % 4,
                    processing_time=10.0 + i * 0.5,
                    success=True
                )

                progress.advance(task)

            # Run optimization
            result = await optimizer.optimize_batch_size()

            progress.update(task, description="Optimization complete")

            # Display results
            console.print(f"\n[bold green]Optimization Complete![/bold green]")
            console.print(f"Optimal Batch Size: [bold]{result.optimal_batch_size}[/bold]")
            console.print(f"Confidence: [bold]{result.confidence:.2f}[/bold]")
            console.print(f"Recommendation: [bold]{result.recommendation}[/bold]")
            console.print(f"State: [bold]{result.state.value}[/bold]")

    asyncio.run(_optimize())


@app.command()
def recommend() -> None:
    """Get recommended batch sizes for different scenarios."""
    optimizer = get_batch_size_optimizer()
    recommendations = optimizer.get_recommended_batch_sizes()

    table = Table(title="Recommended Batch Sizes")
    table.add_column("Scenario", style="cyan")
    table.add_column("Batch Size", style="green")
    table.add_column("Description", style="yellow")

    table.add_row(
        "Conservative",
        str(recommendations["conservative"]),
        "Prioritizes stability and low memory usage"
    )
    table.add_row(
        "Balanced",
        str(recommendations["balanced"]),
        "Current optimized batch size"
    )
    table.add_row(
        "Aggressive",
        str(recommendations["aggressive"]),
        "Maximizes throughput if resources allow"
    )

    console.print(table)


@app.command()
def reset() -> None:
    """Reset optimization state."""
    optimizer = get_batch_size_optimizer()
    optimizer.reset_optimization()
    console.print("[bold green]Optimization state reset successfully![/bold green]")


@app.command()
def simulate(
    duration: int = typer.Option(60, help="Simulation duration in seconds"),
    batch_sizes: str = typer.Option("1,2,4,8", help="Comma-separated batch sizes to test")
) -> None:
    """Simulate batch size optimization with synthetic data."""
    batch_size_list = [int(x.strip()) for x in batch_sizes.split(",")]
    optimizer = get_batch_size_optimizer()

    async def _simulate():
        console.print(f"[bold blue]Starting simulation for {duration} seconds[/bold blue]")
        console.print(f"Testing batch sizes: {batch_size_list}")

        start_time = time.time()
        request_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("Simulating..."),
            console=console
        ) as progress:
            task = progress.add_task("Simulating", total=duration)

            while time.time() - start_time < duration:
                # Simulate processing with different batch sizes
                batch_size = batch_size_list[request_count % len(batch_size_list)]

                # Simulate processing time (inversely related to batch size)
                processing_time = 15.0 / batch_size + (request_count % 3) * 0.5

                # Simulate occasional failures
                success = request_count % 20 != 0

                # Collect metrics
                await optimizer.collect_metrics(
                    batch_size=batch_size,
                    processing_time=processing_time,
                    success=success
                )

                request_count += 1
                elapsed = time.time() - start_time
                progress.update(task, completed=elapsed)

                await asyncio.sleep(0.1)

            # Run optimization
            result = await optimizer.optimize_batch_size()

            progress.update(task, description="Simulation complete")

            # Display results
            console.print(f"\n[bold green]Simulation Complete![/bold green]")
            console.print(f"Requests processed: [bold]{request_count}[/bold]")
            console.print(f"Optimal Batch Size: [bold]{result.optimal_batch_size}[/bold]")
            console.print(f"Confidence: [bold]{result.confidence:.2f}[/bold]")
            console.print(f"Recommendation: [bold]{result.recommendation}[/bold]")

    asyncio.run(_simulate())


@app.command()
def export(
    output_file: str = typer.Option("batch_optimization_data.json", help="Output file path")
) -> None:
    """Export optimization data to JSON file."""
    optimizer = get_batch_size_optimizer()
    status_info = optimizer.get_optimization_status()

    export_data = {
        "status": status_info,
        "metrics_history": [
            {
                "batch_size": m.batch_size,
                "throughput": m.throughput,
                "latency": m.latency,
                "gpu_memory_usage": m.gpu_memory_usage,
                "gpu_utilization": m.gpu_utilization,
                "timestamp": m.timestamp,
                "success_rate": m.success_rate
            }
            for m in optimizer.metrics_history
        ],
        "optimization_history": [
            {
                "optimal_batch_size": r.optimal_batch_size,
                "confidence": r.confidence,
                "recommendation": r.recommendation,
                "state": r.state.value,
                "timestamp": r.metrics.timestamp
            }
            for r in optimizer.optimization_history
        ],
        "export_timestamp": time.time()
    }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    console.print(f"[bold green]Data exported to {output_file}[/bold green]")


@app.command()
def import_data(
    input_file: str = typer.Argument(..., help="Input file path")
) -> None:
    """Import optimization data from JSON file."""
    if not Path(input_file).exists():
        console.print(f"[bold red]File not found: {input_file}[/bold red]")
        raise typer.Exit(1)

    with open(input_file, "r") as f:
        data = json.load(f)

    # Create new optimizer with imported data
    config = BatchSizeConfig()
    optimizer = Gemma3BatchSizeOptimizer(config)

    # Import metrics history
    for metric_data in data.get("metrics_history", []):
        from src.Medical_KG_rev.services.optimization.batch_size_optimizer import PerformanceMetrics
        metric = PerformanceMetrics(**metric_data)
        optimizer.metrics_history.append(metric)

    console.print(f"[bold green]Data imported from {input_file}[/bold green]")
    console.print(f"Imported {len(optimizer.metrics_history)} metrics")


if __name__ == "__main__":
    app()
