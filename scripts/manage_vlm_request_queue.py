#!/usr/bin/env python3
"""CLI script to manage VLM request queue.

This script provides commands to monitor, manage, and control the VLM request queue.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from Medical_KG_rev.services.queuing.vlm_request_queue import (
        QueueConfig,
        QueueStrategy,
        RequestPriority,
        RequestStatus,
        VLMRequestQueue,
        create_vlm_request_queue,
        get_vlm_request_queue,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

app = typer.Typer(
    name="manage-vlm-request-queue",
    help="Manage VLM request queue",
    rich_markup_mode="rich"
)
console = Console()


def display_queue_status(status: Dict[str, Any]) -> None:
    """Display queue status in a formatted table."""
    table = Table(title="VLM Request Queue Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Basic status
    status_color = "green" if status["is_running"] else "red"
    table.add_row("Status", Text("Running" if status["is_running"] else "Stopped", style=status_color))

    # Queue size
    queue_size = status["queue_size"]
    max_size = status["max_queue_size"]
    utilization = (queue_size / max_size) * 100 if max_size > 0 else 0
    utilization_color = "red" if utilization > 80 else "yellow" if utilization > 60 else "green"
    table.add_row("Queue Size", f"{queue_size}/{max_size} ({utilization:.1f}%)", style=utilization_color)

    # Active requests
    active = status["active_requests"]
    max_concurrent = status["max_concurrent_requests"]
    active_utilization = (active / max_concurrent) * 100 if max_concurrent > 0 else 0
    active_color = "red" if active_utilization > 90 else "yellow" if active_utilization > 70 else "green"
    table.add_row("Active Requests", f"{active}/{max_concurrent} ({active_utilization:.1f}%)", style=active_color)

    # Strategy
    table.add_row("Strategy", status["strategy"])

    # Metrics
    metrics = status["metrics"]
    table.add_row("Total Requests", str(metrics["total_requests"]))
    table.add_row("Completed Requests", str(metrics["completed_requests"]))
    table.add_row("Failed Requests", str(metrics["failed_requests"]))
    table.add_row("Cancelled Requests", str(metrics["cancelled_requests"]))
    table.add_row("Timeout Requests", str(metrics["timeout_requests"]))
    table.add_row("Average Processing Time", f"{metrics['average_processing_time']:.2f}s")
    table.add_row("Queue Utilization", f"{metrics['queue_utilization'] * 100:.1f}%")
    table.add_row("Error Rate", f"{metrics['error_rate'] * 100:.1f}%")
    table.add_row("Throughput", f"{metrics['throughput']:.1f} req/min")

    console.print(table)


def display_request_status(status: Dict[str, Any]) -> None:
    """Display request status in a formatted table."""
    table = Table(title=f"Request Status: {status['request_id']}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    # Basic info
    table.add_row("Request ID", status["request_id"])
    table.add_row("Status", status["status"])
    table.add_row("Priority", status["priority"])
    table.add_row("Created At", str(status["created_at"]))

    # Timing info
    if status.get("started_at"):
        table.add_row("Started At", str(status["started_at"]))
    if status.get("completed_at"):
        table.add_row("Completed At", str(status["completed_at"]))
    if status.get("processing_time"):
        table.add_row("Processing Time", f"{status['processing_time']:.2f}s")

    # Retry info
    table.add_row("Retry Count", str(status["retry_count"]))

    # Result/Error
    if status.get("result"):
        table.add_row("Result", "Available")
    if status.get("error"):
        table.add_row("Error", status["error"])

    # Metadata
    if status.get("metadata"):
        table.add_row("Metadata", json.dumps(status["metadata"], indent=2))

    console.print(table)


def display_metrics_history(history: List[Dict[str, Any]]) -> None:
    """Display metrics history in a formatted table."""
    if not history:
        console.print("No metrics history available")
        return

    table = Table(title="Metrics History")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Total", style="green")
    table.add_column("Pending", style="yellow")
    table.add_column("Processing", style="blue")
    table.add_column("Completed", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Error Rate", style="red")
    table.add_column("Throughput", style="magenta")

    for entry in history[-10:]:  # Show last 10 entries
        table.add_row(
            str(entry["timestamp"]),
            str(entry["total_requests"]),
            str(entry["pending_requests"]),
            str(entry["processing_requests"]),
            str(entry["completed_requests"]),
            str(entry["failed_requests"]),
            f"{entry['error_rate'] * 100:.1f}%",
            f"{entry['throughput']:.1f}"
        )

    console.print(table)


@app.command()
def status() -> None:
    """Show current queue status."""
    console.print(Panel("VLM Request Queue Status", style="bold blue"))

    queue = get_vlm_request_queue()
    status_info = queue.get_queue_status()

    display_queue_status(status_info)


@app.command()
def start() -> None:
    """Start the request queue."""
    console.print(Panel("Starting VLM Request Queue", style="bold blue"))

    queue = get_vlm_request_queue()

    async def start_queue() -> None:
        await queue.start()
        console.print("✓ Request queue started", style="green")

    asyncio.run(start_queue())


@app.command()
def stop() -> None:
    """Stop the request queue."""
    console.print(Panel("Stopping VLM Request Queue", style="bold blue"))

    queue = get_vlm_request_queue()

    async def stop_queue() -> None:
        await queue.stop()
        console.print("✓ Request queue stopped", style="green")

    asyncio.run(stop_queue())


@app.command()
def submit(
    pdf_file: str = typer.Argument(..., help="Path to PDF file"),
    priority: str = typer.Option("normal", "--priority", "-p", help="Request priority"),
    timeout: float = typer.Option(300.0, "--timeout", "-t", help="Request timeout in seconds"),
    max_retries: int = typer.Option(3, "--max-retries", "-r", help="Maximum retries"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path")
) -> None:
    """Submit a request to the queue."""
    console.print(Panel("Submitting Request", style="bold blue"))

    # Validate priority
    try:
        priority_enum = RequestPriority(priority.lower())
    except ValueError:
        console.print(f"✗ Invalid priority: {priority}", style="red")
        console.print("Valid priorities: low, normal, high, critical")
        return

    # Read PDF file
    try:
        with open(pdf_file, 'rb') as f:
            pdf_content = f.read()
    except FileNotFoundError:
        console.print(f"✗ PDF file not found: {pdf_file}", style="red")
        return
    except Exception as e:
        console.print(f"✗ Failed to read PDF file: {e}", style="red")
        return

    # Load config if provided
    config = {}
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            console.print(f"✗ Failed to load config file: {e}", style="red")
            return

    # Submit request
    queue = get_vlm_request_queue()

    async def submit_request() -> None:
        try:
            request_id = await queue.submit_request(
                pdf_content=pdf_content,
                config=config,
                options={},
                priority=priority_enum,
                timeout=timeout,
                max_retries=max_retries
            )

            console.print(f"✓ Request submitted successfully", style="green")
            console.print(f"Request ID: {request_id}")
            console.print(f"Priority: {priority}")
            console.print(f"Timeout: {timeout}s")
            console.print(f"Max Retries: {max_retries}")

        except Exception as e:
            console.print(f"✗ Failed to submit request: {e}", style="red")

    asyncio.run(submit_request())


@app.command()
def request_status(
    request_id: str = typer.Argument(..., help="Request ID")
) -> None:
    """Get status of a specific request."""
    console.print(Panel(f"Request Status: {request_id}", style="bold blue"))

    queue = get_vlm_request_queue()

    async def get_status() -> None:
        status_info = await queue.get_request_status(request_id)

        if status_info["status"] == "not_found":
            console.print("✗ Request not found", style="red")
        else:
            display_request_status(status_info)

    asyncio.run(get_status())


@app.command()
def cancel(
    request_id: str = typer.Argument(..., help="Request ID")
) -> None:
    """Cancel a request."""
    console.print(Panel(f"Cancelling Request: {request_id}", style="bold blue"))

    queue = get_vlm_request_queue()

    async def cancel_request() -> None:
        success = await queue.cancel_request(request_id)

        if success:
            console.print("✓ Request cancelled successfully", style="green")
        else:
            console.print("✗ Failed to cancel request", style="red")
            console.print("Request may not exist or may have already completed")

    asyncio.run(cancel_request())


@app.command()
def monitor(
    duration: int = typer.Option(60, "--duration", "-d", help="Monitoring duration in seconds"),
    interval: int = typer.Option(5, "--interval", "-i", help="Update interval in seconds")
) -> None:
    """Monitor queue status in real-time."""
    console.print(Panel("Real-time Queue Monitoring", style="bold blue"))

    queue = get_vlm_request_queue()

    async def monitor_loop() -> None:
        start_time = asyncio.get_event_loop().time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Monitoring queue...", total=None)

            while (asyncio.get_event_loop().time() - start_time) < duration:
                status_info = queue.get_queue_status()

                # Clear previous output
                console.clear()
                console.print(Panel("Real-time Queue Monitoring", style="bold blue"))

                # Display current status
                display_queue_status(status_info)

                # Display metrics history
                history = queue.get_metrics_history()
                if history:
                    console.print("\n")
                    display_metrics_history(history)

                await asyncio.sleep(interval)

    asyncio.run(monitor_loop())


@app.command()
def configure(
    strategy: str = typer.Option("priority", "--strategy", "-s", help="Queue strategy"),
    max_queue_size: int = typer.Option(1000, "--max-queue-size", "-q", help="Maximum queue size"),
    max_concurrent: int = typer.Option(10, "--max-concurrent", "-c", help="Maximum concurrent requests"),
    default_timeout: float = typer.Option(300.0, "--default-timeout", "-t", help="Default timeout in seconds"),
    max_retries: int = typer.Option(3, "--max-retries", "-r", help="Maximum retries"),
    retry_delay: float = typer.Option(1.0, "--retry-delay", "-d", help="Retry delay in seconds"),
    health_check_interval: float = typer.Option(30.0, "--health-check-interval", "-h", help="Health check interval in seconds"),
    cleanup_interval: float = typer.Option(300.0, "--cleanup-interval", "-u", help="Cleanup interval in seconds")
) -> None:
    """Configure queue settings."""
    console.print(Panel("Configuring Queue Settings", style="bold blue"))

    # Validate strategy
    try:
        strategy_enum = QueueStrategy(strategy.lower())
    except ValueError:
        console.print(f"✗ Invalid strategy: {strategy}", style="red")
        console.print("Valid strategies: fifo, priority, round_robin, weighted")
        return

    # Validate parameters
    if max_queue_size <= 0:
        console.print("✗ Max queue size must be positive", style="red")
        return

    if max_concurrent <= 0:
        console.print("✗ Max concurrent requests must be positive", style="red")
        return

    if default_timeout <= 0:
        console.print("✗ Default timeout must be positive", style="red")
        return

    if max_retries < 0:
        console.print("✗ Max retries must be non-negative", style="red")
        return

    if retry_delay < 0:
        console.print("✗ Retry delay must be non-negative", style="red")
        return

    if health_check_interval <= 0:
        console.print("✗ Health check interval must be positive", style="red")
        return

    if cleanup_interval <= 0:
        console.print("✗ Cleanup interval must be positive", style="red")
        return

    # Create new configuration
    config = QueueConfig(
        strategy=strategy_enum,
        max_queue_size=max_queue_size,
        max_concurrent_requests=max_concurrent,
        default_timeout=default_timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        health_check_interval=health_check_interval,
        cleanup_interval=cleanup_interval
    )

    # Create new queue with configuration
    queue = create_vlm_request_queue(config)

    console.print("✓ Configuration updated successfully", style="green")
    console.print(f"Strategy: {strategy}")
    console.print(f"Max queue size: {max_queue_size}")
    console.print(f"Max concurrent requests: {max_concurrent}")
    console.print(f"Default timeout: {default_timeout}s")
    console.print(f"Max retries: {max_retries}")
    console.print(f"Retry delay: {retry_delay}s")
    console.print(f"Health check interval: {health_check_interval}s")
    console.print(f"Cleanup interval: {cleanup_interval}s")


@app.command()
def metrics() -> None:
    """Show queue metrics history."""
    console.print(Panel("Queue Metrics History", style="bold blue"))

    queue = get_vlm_request_queue()
    history = queue.get_metrics_history()

    display_metrics_history(history)


@app.command()
def export_config(
    output_file: str = typer.Option("vlm_queue_config.json", "--output", "-o", help="Output file path")
) -> None:
    """Export current configuration to JSON file."""
    console.print(Panel("Exporting Configuration", style="bold blue"))

    queue = get_vlm_request_queue()
    config = queue.config

    config_dict = {
        "max_queue_size": config.max_queue_size,
        "max_concurrent_requests": config.max_concurrent_requests,
        "strategy": config.strategy.value,
        "default_timeout": config.default_timeout,
        "default_priority": config.default_priority.value,
        "max_retries": config.max_retries,
        "retry_delay": config.retry_delay,
        "health_check_interval": config.health_check_interval,
        "cleanup_interval": config.cleanup_interval,
        "enable_metrics": config.enable_metrics,
        "enable_tracing": config.enable_tracing
    }

    try:
        with open(output_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

        console.print(f"✓ Configuration exported to {output_file}", style="green")
    except Exception as e:
        console.print(f"✗ Failed to export configuration: {e}", style="red")


@app.command()
def import_config(
    input_file: str = typer.Argument(..., help="Input file path")
) -> None:
    """Import configuration from JSON file."""
    console.print(Panel("Importing Configuration", style="bold blue"))

    try:
        with open(input_file, 'r') as f:
            config_dict = json.load(f)

        # Create configuration from dict
        config = QueueConfig(
            max_queue_size=config_dict.get("max_queue_size", 1000),
            max_concurrent_requests=config_dict.get("max_concurrent_requests", 10),
            strategy=QueueStrategy(config_dict.get("strategy", "priority")),
            default_timeout=config_dict.get("default_timeout", 300.0),
            default_priority=RequestPriority(config_dict.get("default_priority", "normal")),
            max_retries=config_dict.get("max_retries", 3),
            retry_delay=config_dict.get("retry_delay", 1.0),
            health_check_interval=config_dict.get("health_check_interval", 30.0),
            cleanup_interval=config_dict.get("cleanup_interval", 300.0),
            enable_metrics=config_dict.get("enable_metrics", True),
            enable_tracing=config_dict.get("enable_tracing", True)
        )

        # Create new queue with imported configuration
        queue = create_vlm_request_queue(config)

        console.print(f"✓ Configuration imported from {input_file}", style="green")
        console.print(f"Strategy: {config.strategy.value}")
        console.print(f"Max queue size: {config.max_queue_size}")
        console.print(f"Max concurrent requests: {config.max_concurrent_requests}")
        console.print(f"Default timeout: {config.default_timeout}s")

    except FileNotFoundError:
        console.print(f"✗ File not found: {input_file}", style="red")
    except json.JSONDecodeError:
        console.print(f"✗ Invalid JSON file: {input_file}", style="red")
    except Exception as e:
        console.print(f"✗ Failed to import configuration: {e}", style="red")


@app.command()
def benchmark(
    duration: int = typer.Option(300, "--duration", "-d", help="Benchmark duration in seconds"),
    request_rate: int = typer.Option(10, "--request-rate", "-r", help="Requests per second"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results")
) -> None:
    """Run a benchmark to test queue performance."""
    console.print(Panel("Running Queue Benchmark", style="bold blue"))

    queue = get_vlm_request_queue()

    async def run_benchmark() -> None:
        start_time = asyncio.get_event_loop().time()
        benchmark_data = []
        request_ids = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running benchmark...", total=duration)

            # Submit requests at specified rate
            request_interval = 1.0 / request_rate
            next_request_time = start_time

            while (asyncio.get_event_loop().time() - start_time) < duration:
                current_time = asyncio.get_event_loop().time()

                # Submit requests
                if current_time >= next_request_time:
                    try:
                        # Create dummy PDF content
                        dummy_pdf = b"dummy pdf content"

                        request_id = await queue.submit_request(
                            pdf_content=dummy_pdf,
                            config={},
                            options={}
                        )
                        request_ids.append(request_id)
                        next_request_time += request_interval
                    except Exception as e:
                        console.print(f"Failed to submit request: {e}")

                # Collect metrics
                status_info = queue.get_queue_status()
                benchmark_data.append({
                    "timestamp": current_time - start_time,
                    "queue_size": status_info["queue_size"],
                    "active_requests": status_info["active_requests"],
                    "completed_requests": status_info["metrics"]["completed_requests"],
                    "failed_requests": status_info["metrics"]["failed_requests"],
                    "error_rate": status_info["metrics"]["error_rate"],
                    "throughput": status_info["metrics"]["throughput"]
                })

                await asyncio.sleep(1)  # Sample every second

        progress.stop()

        # Calculate benchmark results
        total_requests = len(request_ids)
        final_status = queue.get_queue_status()
        completed_requests = final_status["metrics"]["completed_requests"]
        failed_requests = final_status["metrics"]["failed_requests"]

        console.print("✓ Benchmark completed", style="green")
        console.print(f"Duration: {duration} seconds")
        console.print(f"Total requests submitted: {total_requests}")
        console.print(f"Completed requests: {completed_requests}")
        console.print(f"Failed requests: {failed_requests}")
        console.print(f"Success rate: {completed_requests / total_requests * 100:.1f}%")
        console.print(f"Final queue size: {final_status['queue_size']}")
        console.print(f"Final error rate: {final_status['metrics']['error_rate'] * 100:.1f}%")
        console.print(f"Final throughput: {final_status['metrics']['throughput']:.1f} req/min")

        # Export results if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(benchmark_data, f, indent=2)
                console.print(f"✓ Benchmark results exported to {output_file}", style="green")
            except Exception as e:
                console.print(f"✗ Failed to export results: {e}", style="red")

    asyncio.run(run_benchmark())


if __name__ == "__main__":
    app()
