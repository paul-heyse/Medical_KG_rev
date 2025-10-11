#!/usr/bin/env python3
"""CLI script for managing model warm-up procedures.

This script provides command-line interface for managing model warm-up
procedures for VLM models, ensuring consistent performance.
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

# Import the warm-up manager
try:
    from src.Medical_KG_rev.services.optimization.model_warmup import (
        ModelWarmupManager,
        WarmupConfig,
        WarmupStrategy,
        get_warmup_manager,
    )
except ImportError:
    # Fallback for development
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.Medical_KG_rev.services.optimization.model_warmup import (
        ModelWarmupManager,
        WarmupConfig,
        WarmupStrategy,
        get_warmup_manager,
    )

console = Console()
app = typer.Typer()


@app.command()
def status() -> None:
    """Show current warm-up status."""
    manager = get_warmup_manager()
    status_info = manager.get_status()

    table = Table(title="Model Warm-up Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Status", status_info["status"])
    table.add_row("Enabled", "Yes" if status_info["enabled"] else "No")
    table.add_row("Strategy", status_info["strategy"])
    table.add_row("GPU Available", "Yes" if status_info["gpu_available"] else "No")

    if status_info["start_time"]:
        table.add_row("Start Time", time.ctime(status_info["start_time"]))

    if status_info["end_time"]:
        table.add_row("End Time", time.ctime(status_info["end_time"]))

    if status_info["duration"]:
        table.add_row("Duration", f"{status_info['duration']:.2f} seconds")

    console.print(table)

    # Metrics table
    if status_info["metrics"]:
        metrics_table = Table(title="Warm-up Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        metrics = status_info["metrics"]
        metrics_table.add_row("Total Requests", str(metrics["total_requests"]))
        metrics_table.add_row("Successful Requests", str(metrics["successful_requests"]))
        metrics_table.add_row("Failed Requests", str(metrics["failed_requests"]))
        metrics_table.add_row("Average Duration", f"{metrics['average_duration']:.2f}s")
        metrics_table.add_row("Min Duration", f"{metrics['min_duration']:.2f}s")
        metrics_table.add_row("Max Duration", f"{metrics['max_duration']:.2f}s")
        metrics_table.add_row("Performance Score", f"{metrics['performance_score']:.2f}")
        metrics_table.add_row("GPU Memory Peak", f"{metrics['gpu_memory_peak']:.2%}")
        metrics_table.add_row("GPU Utilization Peak", f"{metrics['gpu_utilization_peak']:.2%}")
        metrics_table.add_row("GPU Temperature Peak", f"{metrics['gpu_temperature_peak']:.1f}°C")

        console.print(metrics_table)


@app.command()
def start(
    strategy: str = typer.Option("standard", help="Warm-up strategy (minimal, standard, comprehensive, custom)"),
    requests: int = typer.Option(10, help="Number of warm-up requests"),
    timeout: int = typer.Option(300, help="Warm-up timeout in seconds"),
    batch_sizes: str = typer.Option("1,2,4,8", help="Comma-separated batch sizes"),
    retry_attempts: int = typer.Option(3, help="Number of retry attempts")
) -> None:
    """Start model warm-up procedure."""
    try:
        warmup_strategy = WarmupStrategy(strategy)
    except ValueError:
        console.print(f"[bold red]Invalid strategy: {strategy}[/bold red]")
        console.print(f"Valid strategies: {[s.value for s in WarmupStrategy]}")
        raise typer.Exit(1)

    batch_size_list = [int(x.strip()) for x in batch_sizes.split(",")]

    config = WarmupConfig(
        enabled=True,
        strategy=warmup_strategy,
        warmup_requests=requests,
        warmup_timeout=timeout,
        batch_sizes=batch_size_list,
        retry_attempts=retry_attempts
    )

    manager = ModelWarmupManager(config)

    async def _warmup():
        console.print(f"[bold blue]Starting model warm-up[/bold blue]")
        console.print(f"Strategy: [bold]{strategy}[/bold]")
        console.print(f"Requests: [bold]{requests}[/bold]")
        console.print(f"Timeout: [bold]{timeout}s[/bold]")
        console.print(f"Batch sizes: [bold]{batch_size_list}[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("Warming up model..."),
            console=console
        ) as progress:
            task = progress.add_task("Warming up", total=None)

            # Mock VLM client for demonstration
            class MockVLMClient:
                async def process_pdf(self, *args, **kwargs):
                    await asyncio.sleep(0.1)  # Simulate processing
                    return {"processed": True}

            mock_client = MockVLMClient()

            # Start warm-up
            success = await manager.warmup(mock_client)

            if success:
                progress.update(task, description="Warm-up completed successfully")
                console.print("\n[bold green]Model warm-up completed successfully![/bold green]")
            else:
                progress.update(task, description="Warm-up failed")
                console.print("\n[bold red]Model warm-up failed![/bold red]")

    asyncio.run(_warmup())


@app.command()
def stop() -> None:
    """Stop ongoing warm-up procedure."""
    manager = get_warmup_manager()

    if manager.status.value == "in_progress":
        # In a real implementation, this would stop the warm-up
        console.print("[bold yellow]Stopping warm-up procedure...[/bold yellow]")
        # For now, we just reset the state
        manager.reset()
        console.print("[bold green]Warm-up procedure stopped![/bold green]")
    else:
        console.print("[bold yellow]No warm-up procedure in progress[/bold yellow]")


@app.command()
def reset() -> None:
    """Reset warm-up state."""
    manager = get_warmup_manager()
    manager.reset()
    console.print("[bold green]Warm-up state reset successfully![/bold green]")


@app.command()
def configure(
    enabled: Optional[bool] = typer.Option(None, help="Enable/disable warm-up"),
    strategy: Optional[str] = typer.Option(None, help="Warm-up strategy"),
    requests: Optional[int] = typer.Option(None, help="Number of warm-up requests"),
    timeout: Optional[int] = typer.Option(None, help="Warm-up timeout in seconds"),
    memory_threshold: Optional[float] = typer.Option(None, help="GPU memory threshold"),
    temperature_threshold: Optional[float] = typer.Option(None, help="GPU temperature threshold"),
    performance_threshold: Optional[float] = typer.Option(None, help="Performance threshold"),
    retry_attempts: Optional[int] = typer.Option(None, help="Number of retry attempts"),
    retry_delay: Optional[int] = typer.Option(None, help="Retry delay in seconds")
) -> None:
    """Configure warm-up settings."""
    # Get current config
    manager = get_warmup_manager()
    current_config = manager.config

    # Create new config with updated values
    new_config = WarmupConfig(
        enabled=enabled if enabled is not None else current_config.enabled,
        strategy=WarmupStrategy(strategy) if strategy else current_config.strategy,
        warmup_requests=requests if requests is not None else current_config.warmup_requests,
        warmup_timeout=timeout if timeout is not None else current_config.warmup_timeout,
        batch_sizes=current_config.batch_sizes,
        request_types=current_config.request_types,
        gpu_memory_threshold=memory_threshold if memory_threshold is not None else current_config.gpu_memory_threshold,
        temperature_threshold=temperature_threshold if temperature_threshold is not None else current_config.temperature_threshold,
        performance_threshold=performance_threshold if performance_threshold is not None else current_config.performance_threshold,
        retry_attempts=retry_attempts if retry_attempts is not None else current_config.retry_attempts,
        retry_delay=retry_delay if retry_delay is not None else current_config.retry_delay,
        monitoring_interval=current_config.monitoring_interval
    )

    # Create new manager instance with updated config
    new_manager = ModelWarmupManager(new_config)

    # Update global manager instance
    import src.Medical_KG_rev.services.optimization.model_warmup as warmup_module
    warmup_module._warmup_manager = new_manager

    console.print("[bold green]Warm-up configuration updated![/bold green]")

    # Show new configuration
    table = Table(title="Updated Warm-up Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Enabled", "Yes" if new_config.enabled else "No")
    table.add_row("Strategy", new_config.strategy.value)
    table.add_row("Warm-up Requests", str(new_config.warmup_requests))
    table.add_row("Timeout (seconds)", str(new_config.warmup_timeout))
    table.add_row("Batch Sizes", str(new_config.batch_sizes))
    table.add_row("GPU Memory Threshold", f"{new_config.gpu_memory_threshold:.1%}")
    table.add_row("Temperature Threshold", f"{new_config.temperature_threshold:.1f}°C")
    table.add_row("Performance Threshold", f"{new_config.performance_threshold:.1%}")
    table.add_row("Retry Attempts", str(new_config.retry_attempts))
    table.add_row("Retry Delay (seconds)", str(new_config.retry_delay))

    console.print(table)


@app.command()
def test(
    duration: int = typer.Option(60, help="Test duration in seconds"),
    interval: int = typer.Option(5, help="Update interval in seconds")
) -> None:
    """Test warm-up procedure with simulated requests."""
    manager = get_warmup_manager()

    async def _test():
        console.print(f"[bold blue]Testing warm-up procedure for {duration} seconds[/bold blue]")
        console.print(f"Update interval: {interval} seconds")

        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("Testing..."),
            console=console
        ) as progress:
            task = progress.add_task("Testing", total=duration)

            while time.time() - start_time < duration:
                status_info = manager.get_status()

                # Update progress
                elapsed = time.time() - start_time
                progress.update(task, completed=elapsed)

                # Display current status
                console.print(f"\n[bold]Warm-up Status (t+{elapsed:.0f}s):[/bold]")
                console.print(f"Status: {status_info['status']}")
                console.print(f"GPU Available: {'Yes' if status_info['gpu_available'] else 'No'}")

                if status_info["metrics"]:
                    metrics = status_info["metrics"]
                    console.print(f"Total Requests: {metrics['total_requests']}")
                    console.print(f"Successful: {metrics['successful_requests']}")
                    console.print(f"Failed: {metrics['failed_requests']}")
                    console.print(f"Performance Score: {metrics['performance_score']:.2f}")

                await asyncio.sleep(interval)

            progress.update(task, description="Test complete")

    asyncio.run(_test())


@app.command()
def export(
    output_file: str = typer.Option("warmup_data.json", help="Output file path")
) -> None:
    """Export warm-up data to JSON file."""
    manager = get_warmup_manager()
    status_info = manager.get_status()

    export_data = {
        "warmup_status": status_info,
        "export_timestamp": time.time()
    }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    console.print(f"[bold green]Warm-up data exported to {output_file}[/bold green]")


@app.command()
def ready() -> None:
    """Check if model is ready (warm-up completed successfully)."""
    manager = get_warmup_manager()

    if manager.is_ready():
        console.print("[bold green]Model is ready![/bold green]")
        console.print("Warm-up completed successfully.")
    else:
        console.print("[bold yellow]Model is not ready[/bold yellow]")
        console.print(f"Current status: {manager.status.value}")

        if manager.status.value == "failed":
            console.print("Warm-up failed. Check logs for details.")
        elif manager.status.value == "not_started":
            console.print("Warm-up has not been started.")
        elif manager.status.value == "in_progress":
            console.print("Warm-up is currently in progress.")


if __name__ == "__main__":
    app()
