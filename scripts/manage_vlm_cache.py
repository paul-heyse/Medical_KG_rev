#!/usr/bin/env python3
"""CLI script for managing VLM cache.

This script provides command-line interface for managing the VLM cache,
including cache statistics, clearing cache, and performance monitoring.
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

# Import the cache
try:
    from src.Medical_KG_rev.services.caching.vlm_cache import (
        CacheConfig,
        CacheLevel,
        VLMCache,
        get_vlm_cache,
    )
except ImportError:
    # Fallback for development
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.Medical_KG_rev.services.caching.vlm_cache import (
        CacheConfig,
        CacheLevel,
        VLMCache,
        get_vlm_cache,
    )

console = Console()
app = typer.Typer()


@app.command()
def status() -> None:
    """Show cache status and statistics."""
    cache = get_vlm_cache()
    cache_info = cache.get_cache_info()
    stats = cache.get_stats()

    # Main status table
    table = Table(title="VLM Cache Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Enabled", "Yes" if cache_info["enabled"] else "No")
    table.add_row("Strategy", cache_info["strategy"])
    table.add_row("Max Size", str(cache_info["max_size"]))
    table.add_row("TTL (seconds)", str(cache_info["ttl_seconds"]))
    table.add_row("Memory Limit (MB)", str(cache_info["memory_limit_mb"]))
    table.add_row("Redis Available", "Yes" if cache_info["redis_available"] else "No")
    table.add_row("Compression", "Enabled" if cache_info["compression_enabled"] else "Disabled")
    table.add_row("Encryption", "Enabled" if cache_info["encryption_enabled"] else "Disabled")

    console.print(table)

    # Statistics table
    stats_table = Table(title="Cache Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Total Entries", str(stats.total_entries))
    stats_table.add_row("Total Size (bytes)", str(stats.total_size_bytes))
    stats_table.add_row("Cache Hits", str(stats.hits))
    stats_table.add_row("Cache Misses", str(stats.misses))
    stats_table.add_row("Hit Rate", f"{stats.hit_rate:.2%}")
    stats_table.add_row("Miss Rate", f"{stats.miss_rate:.2%}")
    stats_table.add_row("Evictions", str(stats.evictions))

    console.print(stats_table)

    # Entries by level table
    entries_table = Table(title="Entries by Cache Level")
    entries_table.add_column("Level", style="cyan")
    entries_table.add_column("Count", style="green")

    for level, count in cache_info["entries_by_level"].items():
        entries_table.add_row(level, str(count))

    console.print(entries_table)


@app.command()
def clear() -> None:
    """Clear all cached values."""
    cache = get_vlm_cache()

    async def _clear():
        with Progress(
            SpinnerColumn(),
            TextColumn("Clearing cache..."),
            console=console
        ) as progress:
            task = progress.add_task("Clearing", total=None)

            await cache.clear()

            progress.update(task, description="Cache cleared")

    asyncio.run(_clear())
    console.print("[bold green]Cache cleared successfully![/bold green]")


@app.command()
def invalidate(
    pdf_path: str = typer.Argument(..., help="Path to PDF file"),
    level: str = typer.Option("doctags_result", help="Cache level to invalidate")
) -> None:
    """Invalidate cached value for a specific PDF."""
    cache = get_vlm_cache()

    async def _invalidate():
        try:
            # Read PDF content
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()

            # Determine cache level
            try:
                cache_level = CacheLevel(level)
            except ValueError:
                console.print(f"[bold red]Invalid cache level: {level}[/bold red]")
                console.print(f"Valid levels: {[l.value for l in CacheLevel]}")
                raise typer.Exit(1)

            # Invalidate cache
            success = await cache.invalidate(
                pdf_content=pdf_content,
                config={},
                options={},
                level=cache_level
            )

            if success:
                console.print(f"[bold green]Cache invalidated for {pdf_path}[/bold green]")
            else:
                console.print(f"[bold yellow]No cache entry found for {pdf_path}[/bold yellow]")

        except FileNotFoundError:
            console.print(f"[bold red]PDF file not found: {pdf_path}[/bold red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[bold red]Error invalidating cache: {e}[/bold red]")
            raise typer.Exit(1)

    asyncio.run(_invalidate())


@app.command()
def test(
    pdf_path: str = typer.Argument(..., help="Path to PDF file"),
    iterations: int = typer.Option(5, help="Number of test iterations")
) -> None:
    """Test cache performance with a PDF file."""
    cache = get_vlm_cache()

    async def _test():
        try:
            # Read PDF content
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()

            console.print(f"[bold blue]Testing cache with {pdf_path}[/bold blue]")
            console.print(f"Iterations: {iterations}")

            # Test cache performance
            hit_times = []
            miss_times = []

            for i in range(iterations):
                start_time = time.time()

                # Try to get from cache
                result = await cache.get(
                    pdf_content=pdf_content,
                    config={"test": True},
                    options={"iteration": i},
                    level=CacheLevel.DOCTAGS_RESULT
                )

                get_time = time.time() - start_time

                if result is None:
                    # Cache miss - simulate processing and cache result
                    miss_times.append(get_time)

                    # Simulate processing time
                    await asyncio.sleep(0.1)

                    # Cache the result
                    await cache.set(
                        pdf_content=pdf_content,
                        config={"test": True},
                        options={"iteration": i},
                        level=CacheLevel.DOCTAGS_RESULT,
                        value={"processed": True, "iteration": i},
                        ttl_seconds=60
                    )

                    console.print(f"Iteration {i+1}: [bold red]MISS[/bold red] ({get_time:.3f}s)")
                else:
                    # Cache hit
                    hit_times.append(get_time)
                    console.print(f"Iteration {i+1}: [bold green]HIT[/bold red] ({get_time:.3f}s)")

            # Calculate statistics
            if hit_times:
                avg_hit_time = sum(hit_times) / len(hit_times)
                console.print(f"\n[bold green]Average hit time: {avg_hit_time:.3f}s[/bold green]")

            if miss_times:
                avg_miss_time = sum(miss_times) / len(miss_times)
                console.print(f"[bold red]Average miss time: {avg_miss_time:.3f}s[/bold red]")

            # Show final cache stats
            stats = cache.get_stats()
            console.print(f"\n[bold blue]Final cache stats:[/bold blue]")
            console.print(f"Hits: {stats.hits}")
            console.print(f"Misses: {stats.misses}")
            console.print(f"Hit rate: {stats.hit_rate:.2%}")

        except FileNotFoundError:
            console.print(f"[bold red]PDF file not found: {pdf_path}[/bold red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[bold red]Error testing cache: {e}[/bold red]")
            raise typer.Exit(1)

    asyncio.run(_test())


@app.command()
def configure(
    enabled: Optional[bool] = typer.Option(None, help="Enable/disable cache"),
    strategy: Optional[str] = typer.Option(None, help="Cache strategy (lru, lfu, ttl, adaptive)"),
    max_size: Optional[int] = typer.Option(None, help="Maximum cache size"),
    ttl_seconds: Optional[int] = typer.Option(None, help="TTL in seconds"),
    memory_limit_mb: Optional[int] = typer.Option(None, help="Memory limit in MB"),
    redis_url: Optional[str] = typer.Option(None, help="Redis URL"),
    cache_directory: Optional[str] = typer.Option(None, help="Cache directory"),
    compression: Optional[bool] = typer.Option(None, help="Enable/disable compression"),
    encryption: Optional[bool] = typer.Option(None, help="Enable/disable encryption")
) -> None:
    """Configure cache settings."""
    # Get current config
    cache = get_vlm_cache()
    current_config = cache.config

    # Create new config with updated values
    new_config = CacheConfig(
        enabled=enabled if enabled is not None else current_config.enabled,
        strategy=CacheStrategy(strategy) if strategy else current_config.strategy,
        max_size=max_size if max_size is not None else current_config.max_size,
        ttl_seconds=ttl_seconds if ttl_seconds is not None else current_config.ttl_seconds,
        memory_limit_mb=memory_limit_mb if memory_limit_mb is not None else current_config.memory_limit_mb,
        redis_url=redis_url if redis_url is not None else current_config.redis_url,
        cache_directory=cache_directory if cache_directory is not None else current_config.cache_directory,
        compression_enabled=compression if compression is not None else current_config.compression_enabled,
        encryption_enabled=encryption if encryption is not None else current_config.encryption_enabled,
        encryption_key=current_config.encryption_key
    )

    # Create new cache instance with updated config
    new_cache = VLMCache(new_config)

    # Update global cache instance
    import src.Medical_KG_rev.services.caching.vlm_cache as vlm_cache_module
    vlm_cache_module._vlm_cache = new_cache

    console.print("[bold green]Cache configuration updated![/bold green]")

    # Show new configuration
    table = Table(title="Updated Cache Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Enabled", "Yes" if new_config.enabled else "No")
    table.add_row("Strategy", new_config.strategy.value)
    table.add_row("Max Size", str(new_config.max_size))
    table.add_row("TTL (seconds)", str(new_config.ttl_seconds))
    table.add_row("Memory Limit (MB)", str(new_config.memory_limit_mb))
    table.add_row("Redis URL", new_config.redis_url or "Not configured")
    table.add_row("Cache Directory", new_config.cache_directory)
    table.add_row("Compression", "Enabled" if new_config.compression_enabled else "Disabled")
    table.add_row("Encryption", "Enabled" if new_config.encryption_enabled else "Disabled")

    console.print(table)


@app.command()
def export(
    output_file: str = typer.Option("vlm_cache_data.json", help="Output file path")
) -> None:
    """Export cache data to JSON file."""
    cache = get_vlm_cache()
    cache_info = cache.get_cache_info()

    export_data = {
        "cache_info": cache_info,
        "export_timestamp": time.time()
    }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    console.print(f"[bold green]Cache data exported to {output_file}[/bold green]")


@app.command()
def monitor(
    duration: int = typer.Option(60, help="Monitoring duration in seconds"),
    interval: int = typer.Option(5, help="Update interval in seconds")
) -> None:
    """Monitor cache performance in real-time."""
    cache = get_vlm_cache()

    async def _monitor():
        console.print(f"[bold blue]Monitoring cache for {duration} seconds[/bold blue]")
        console.print(f"Update interval: {interval} seconds")

        start_time = time.time()
        initial_stats = cache.get_stats()

        with Progress(
            SpinnerColumn(),
            TextColumn("Monitoring..."),
            console=console
        ) as progress:
            task = progress.add_task("Monitoring", total=duration)

            while time.time() - start_time < duration:
                current_stats = cache.get_stats()

                # Calculate deltas
                hits_delta = current_stats.hits - initial_stats.hits
                misses_delta = current_stats.misses - initial_stats.misses
                evictions_delta = current_stats.evictions - initial_stats.evictions

                # Update progress
                elapsed = time.time() - start_time
                progress.update(task, completed=elapsed)

                # Display current stats
                console.print(f"\n[bold]Cache Stats (t+{elapsed:.0f}s):[/bold]")
                console.print(f"Hits: {current_stats.hits} (+{hits_delta})")
                console.print(f"Misses: {current_stats.misses} (+{misses_delta})")
                console.print(f"Hit Rate: {current_stats.hit_rate:.2%}")
                console.print(f"Evictions: {current_stats.evictions} (+{evictions_delta})")
                console.print(f"Total Entries: {current_stats.total_entries}")
                console.print(f"Total Size: {current_stats.total_size_bytes} bytes")

                await asyncio.sleep(interval)

            progress.update(task, description="Monitoring complete")

    asyncio.run(_monitor())


if __name__ == "__main__":
    app()
