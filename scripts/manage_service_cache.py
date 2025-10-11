#!/usr/bin/env python3
"""Service cache management script.

This script provides command-line interface for managing service caches,
including clearing, monitoring, and configuring cache settings.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Medical_KG_rev.services.caching.cache_integration import CacheIntegrationConfig
from Medical_KG_rev.services.caching.cache_manager import (
    CacheConfig,
    create_cache_manager,
)
from Medical_KG_rev.services.caching.service_cache import create_service_cache_config

app = typer.Typer(help="Service Cache Management Tool")
console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Cache management operations."""

    def __init__(self, cache_type: str = "memory", redis_url: str | None = None):
        self.cache_type = cache_type
        self.redis_url = redis_url
        self._cache_manager = None

    async def _get_cache_manager(self):
        """Get or create cache manager."""
        if self._cache_manager is None:
            config = CacheConfig()
            self._cache_manager = create_cache_manager(self.cache_type, config)
        return self._cache_manager

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        cache_manager = await self._get_cache_manager()
        return await cache_manager.get_stats()

    async def clear_cache(self) -> None:
        """Clear all cache entries."""
        cache_manager = await self._get_cache_manager()
        await cache_manager.clear()
        console.print("✅ Cache cleared successfully", style="green")

    async def get_cache_info(self) -> dict[str, Any]:
        """Get cache information."""
        stats = await self.get_stats()
        return {"cache_type": self.cache_type, "redis_url": self.redis_url, "stats": stats}

    async def close(self) -> None:
        """Close cache manager."""
        if self._cache_manager:
            await self._cache_manager.close()


@app.command()
def stats(
    cache_type: str = typer.Option("memory", help="Cache backend type"),
    redis_url: str | None = typer.Option(None, help="Redis URL for Redis cache"),
):
    """Display cache statistics."""

    async def _stats():
        manager = CacheManager(cache_type, redis_url)

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Fetching cache statistics...", total=None)

            try:
                info = await manager.get_cache_info()
                progress.stop()

                # Display cache information
                console.print("\n📊 Cache Statistics", style="bold blue")

                # Cache type info
                cache_info_table = Table(title="Cache Configuration")
                cache_info_table.add_column("Property", style="cyan")
                cache_info_table.add_column("Value", style="green")

                cache_info_table.add_row("Cache Type", info["cache_type"])
                cache_info_table.add_row("Redis URL", info["redis_url"] or "N/A")

                console.print(cache_info_table)

                # Statistics table
                stats_table = Table(title="Cache Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="green")

                stats = info["stats"]
                for key, value in stats.items():
                    if isinstance(value, float):
                        value = f"{value:.2f}"
                    stats_table.add_row(key.replace("_", " ").title(), str(value))

                console.print(stats_table)

            except Exception as e:
                progress.stop()
                console.print(f"❌ Error fetching cache statistics: {e}", style="red")
            finally:
                await manager.close()

    asyncio.run(_stats())


@app.command()
def clear(
    cache_type: str = typer.Option("memory", help="Cache backend type"),
    redis_url: str | None = typer.Option(None, help="Redis URL for Redis cache"),
    confirm: bool = typer.Option(False, "--confirm", help="Skip confirmation prompt"),
):
    """Clear all cache entries."""

    async def _clear():
        manager = CacheManager(cache_type, redis_url)

        if not confirm:
            console.print("⚠️  This will clear ALL cache entries.", style="yellow")
            if not typer.confirm("Are you sure you want to continue?"):
                console.print("Operation cancelled.", style="yellow")
                return

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Clearing cache...", total=None)

            try:
                await manager.clear_cache()
                progress.stop()
            except Exception as e:
                progress.stop()
                console.print(f"❌ Error clearing cache: {e}", style="red")
            finally:
                await manager.close()

    asyncio.run(_clear())


@app.command()
def monitor(
    cache_type: str = typer.Option("memory", help="Cache backend type"),
    redis_url: str | None = typer.Option(None, help="Redis URL for Redis cache"),
    interval: int = typer.Option(5, help="Monitoring interval in seconds"),
):
    """Monitor cache statistics in real-time."""

    async def _monitor():
        manager = CacheManager(cache_type, redis_url)

        try:
            console.print("🔄 Starting cache monitoring...", style="blue")
            console.print(f"📊 Refresh interval: {interval} seconds", style="blue")
            console.print("Press Ctrl+C to stop monitoring\n", style="blue")

            while True:
                try:
                    info = await manager.get_cache_info()
                    stats = info["stats"]

                    # Clear screen and display stats
                    console.clear()
                    console.print("📊 Cache Monitoring Dashboard", style="bold blue")
                    console.print(f"Cache Type: {info['cache_type']}", style="cyan")
                    console.print(f"Redis URL: {info['redis_url'] or 'N/A'}", style="cyan")
                    console.print()

                    # Create stats table
                    stats_table = Table(title="Live Cache Statistics")
                    stats_table.add_column("Metric", style="cyan")
                    stats_table.add_column("Value", style="green")

                    for key, value in stats.items():
                        if isinstance(value, float):
                            value = f"{value:.2f}"
                        stats_table.add_row(key.replace("_", " ").title(), str(value))

                    console.print(stats_table)
                    console.print(f"\n🔄 Next update in {interval} seconds...", style="blue")

                    await asyncio.sleep(interval)

                except KeyboardInterrupt:
                    console.print("\n🛑 Monitoring stopped by user", style="yellow")
                    break
                except Exception as e:
                    console.print(f"❌ Error during monitoring: {e}", style="red")
                    await asyncio.sleep(interval)

        finally:
            await manager.close()

    asyncio.run(_monitor())


@app.command()
def config(
    cache_type: str = typer.Option("memory", help="Cache backend type"),
    redis_url: str | None = typer.Option(None, help="Redis URL for Redis cache"),
    ttl: int = typer.Option(3600, help="Default TTL in seconds"),
    max_size: int = typer.Option(1024, help="Maximum cache size in MB"),
    compression: bool = typer.Option(True, help="Enable compression"),
    format: str = typer.Option("json", help="Serialization format"),
):
    """Display cache configuration."""

    async def _config():
        manager = CacheManager(cache_type, redis_url)

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Loading cache configuration...", total=None)

            try:
                # Create configuration
                cache_config = CacheConfig(
                    ttl_seconds=ttl,
                    max_size_bytes=max_size * 1024 * 1024,
                    compression_enabled=compression,
                    serialization_format=format,
                )

                service_config = create_service_cache_config(cache_type, redis_url)
                integration_config = CacheIntegrationConfig(
                    enabled=True, cache_type=cache_type, redis_url=redis_url
                )

                progress.stop()

                # Display configuration
                console.print("\n⚙️  Cache Configuration", style="bold blue")

                # Cache manager config
                manager_table = Table(title="Cache Manager Configuration")
                manager_table.add_column("Property", style="cyan")
                manager_table.add_column("Value", style="green")

                manager_table.add_row("Cache Type", cache_type)
                manager_table.add_row("Redis URL", redis_url or "N/A")
                manager_table.add_row("TTL (seconds)", str(ttl))
                manager_table.add_row("Max Size (MB)", str(max_size))
                manager_table.add_row("Compression", "Enabled" if compression else "Disabled")
                manager_table.add_row("Serialization Format", format)

                console.print(manager_table)

                # Service config
                service_table = Table(title="Service Cache Configuration")
                service_table.add_column("Service", style="cyan")
                service_table.add_column("Policy", style="green")

                for service, policy in service_config.policies.items():
                    service_table.add_row(
                        service, f"TTL: {policy.ttl_seconds}s, Enabled: {policy.enabled}"
                    )

                console.print(service_table)

                # Integration config
                integration_table = Table(title="Cache Integration Configuration")
                integration_table.add_column("Property", style="cyan")
                integration_table.add_column("Value", style="green")

                integration_table.add_row("Integration Enabled", str(integration_config.enabled))
                integration_table.add_row("Cache Type", integration_config.cache_type)
                integration_table.add_row("Redis URL", integration_config.redis_url or "N/A")

                console.print(integration_table)

            except Exception as e:
                progress.stop()
                console.print(f"❌ Error loading configuration: {e}", style="red")
            finally:
                await manager.close()

    asyncio.run(_config())


@app.command()
def test(
    cache_type: str = typer.Option("memory", help="Cache backend type"),
    redis_url: str | None = typer.Option(None, help="Redis URL for Redis cache"),
    operations: int = typer.Option(100, help="Number of test operations"),
):
    """Test cache performance."""

    async def _test():
        manager = CacheManager(cache_type, redis_url)

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Running cache performance test...", total=operations)

            try:
                cache_manager = await manager._get_cache_manager()

                # Test data
                test_data = {
                    "text": "This is a test document for cache performance testing.",
                    "model": "test-model",
                    "config": {"batch_size": 32, "max_length": 512},
                }

                # Performance metrics
                import time

                start_time = time.time()
                hits = 0
                misses = 0

                # Run test operations
                for i in range(operations):
                    # Alternate between set and get operations
                    if i % 2 == 0:
                        # Set operation
                        await cache_manager.set(
                            "test_service", "test_operation", test_data, f"result_{i}"
                        )
                    else:
                        # Get operation
                        result = await cache_manager.get(
                            "test_service", "test_operation", test_data
                        )
                        if result is not None:
                            hits += 1
                        else:
                            misses += 1

                    progress.update(task, advance=1)

                end_time = time.time()
                duration = end_time - start_time

                progress.stop()

                # Display results
                console.print("\n🧪 Cache Performance Test Results", style="bold blue")

                results_table = Table(title="Test Results")
                results_table.add_column("Metric", style="cyan")
                results_table.add_column("Value", style="green")

                results_table.add_row("Total Operations", str(operations))
                results_table.add_row("Duration (seconds)", f"{duration:.2f}")
                results_table.add_row("Operations per Second", f"{operations / duration:.2f}")
                results_table.add_row("Cache Hits", str(hits))
                results_table.add_row("Cache Misses", str(misses))
                results_table.add_row("Hit Rate", f"{(hits / max(hits + misses, 1)) * 100:.2f}%")

                console.print(results_table)

                # Performance rating
                ops_per_sec = operations / duration
                if ops_per_sec > 1000:
                    rating = "🚀 Excellent"
                    style = "green"
                elif ops_per_sec > 500:
                    rating = "✅ Good"
                    style = "green"
                elif ops_per_sec > 100:
                    rating = "⚠️  Average"
                    style = "yellow"
                else:
                    rating = "❌ Poor"
                    style = "red"

                console.print(f"\nPerformance Rating: {rating}", style=style)

            except Exception as e:
                progress.stop()
                console.print(f"❌ Error during performance test: {e}", style="red")
            finally:
                await manager.close()

    asyncio.run(_test())


@app.command()
def health(
    cache_type: str = typer.Option("memory", help="Cache backend type"),
    redis_url: str | None = typer.Option(None, help="Redis URL for Redis cache"),
):
    """Check cache health."""

    async def _health():
        manager = CacheManager(cache_type, redis_url)

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Checking cache health...", total=None)

            try:
                # Test basic operations
                cache_manager = await manager._get_cache_manager()

                # Test set/get
                test_data = {"test": "health_check"}
                await cache_manager.set("health", "check", test_data, "test_value")
                result = await cache_manager.get("health", "check", test_data)

                # Test delete
                deleted = await cache_manager.delete("health", "check", test_data)

                # Get stats
                stats = await cache_manager.get_stats()

                progress.stop()

                # Display health status
                console.print("\n🏥 Cache Health Check", style="bold blue")

                health_table = Table(title="Health Status")
                health_table.add_column("Check", style="cyan")
                health_table.add_column("Status", style="green")
                health_table.add_column("Details", style="white")

                # Set/Get test
                if result == "test_value":
                    health_table.add_row("Set/Get Operations", "✅ PASS", "Basic operations working")
                else:
                    health_table.add_row("Set/Get Operations", "❌ FAIL", "Basic operations failed")

                # Delete test
                if deleted:
                    health_table.add_row("Delete Operations", "✅ PASS", "Delete operations working")
                else:
                    health_table.add_row("Delete Operations", "❌ FAIL", "Delete operations failed")

                # Stats test
                if stats:
                    health_table.add_row("Statistics", "✅ PASS", "Statistics available")
                else:
                    health_table.add_row("Statistics", "❌ FAIL", "Statistics unavailable")

                console.print(health_table)

                # Overall health
                all_passed = result == "test_value" and deleted and stats
                if all_passed:
                    console.print("\n🎉 Cache is healthy!", style="green")
                else:
                    console.print("\n⚠️  Cache has issues that need attention", style="yellow")

            except Exception as e:
                progress.stop()
                console.print(f"❌ Cache health check failed: {e}", style="red")
                console.print(
                    "🔧 Please check your cache configuration and connectivity", style="yellow"
                )
            finally:
                await manager.close()

    asyncio.run(_health())


if __name__ == "__main__":
    app()
