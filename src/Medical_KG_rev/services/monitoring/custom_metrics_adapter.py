"""Custom metrics adapter for monitoring."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class CustomMetricsAdapter:
    """Adapter for custom metrics collection."""

    def __init__(self, prometheus_url: str = "http://localhost:9090") -> None:
        """Initialize the custom metrics adapter."""
        self.logger = logger
        self.prometheus_url = prometheus_url
        self._running = False

    async def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the metrics adapter."""
        try:
            self._running = True
            self.logger.info(f"Starting custom metrics adapter on {host}:{port}")

            # Mock server implementation
            while self._running:
                await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"Failed to start metrics adapter: {e}")
            raise

    async def stop(self) -> None:
        """Stop the metrics adapter."""
        self._running = False
        self.logger.info("Custom metrics adapter stopped")

    def collect_metrics(self) -> dict[str, Any]:
        """Collect custom metrics."""
        try:
            # Mock metrics collection
            metrics = {
                "custom_metric_1": 42,
                "custom_metric_2": 3.14,
                "custom_metric_3": "test_value",
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return {}

    def health_check(self) -> dict[str, Any]:
        """Check adapter health."""
        return {
            "adapter": "custom_metrics",
            "status": "healthy",
            "running": self._running,
            "prometheus_url": self.prometheus_url,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            "running": self._running,
            "prometheus_url": self.prometheus_url,
        }


class CustomMetricsAdapterFactory:
    """Factory for creating custom metrics adapters."""

    @staticmethod
    def create(prometheus_url: str = "http://localhost:9090") -> CustomMetricsAdapter:
        """Create a custom metrics adapter instance."""
        return CustomMetricsAdapter(prometheus_url)

    @staticmethod
    def create_with_config(config: dict[str, Any]) -> CustomMetricsAdapter:
        """Create a custom metrics adapter with configuration."""
        prometheus_url = config.get("prometheus_url", "http://localhost:9090")
        return CustomMetricsAdapter(prometheus_url)


# Global custom metrics adapter instance
_custom_metrics_adapter: CustomMetricsAdapter | None = None


def get_custom_metrics_adapter() -> CustomMetricsAdapter:
    """Get the global custom metrics adapter instance."""
    global _custom_metrics_adapter

    if _custom_metrics_adapter is None:
        _custom_metrics_adapter = CustomMetricsAdapterFactory.create()

    return _custom_metrics_adapter


def create_custom_metrics_adapter(prometheus_url: str = "http://localhost:9090") -> CustomMetricsAdapter:
    """Create a new custom metrics adapter instance."""
    return CustomMetricsAdapterFactory.create(prometheus_url)


async def main():
    """Main function for running the adapter."""
    import argparse

    parser = argparse.ArgumentParser(description="Custom Metrics Adapter")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--prometheus-url", default="http://localhost:9090", help="Prometheus URL")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create and start adapter
    adapter = CustomMetricsAdapter(args.prometheus_url)

    try:
        await adapter.start(args.host, args.port)
    except KeyboardInterrupt:
        print("Custom metrics adapter stopped")


if __name__ == "__main__":
    asyncio.run(main())
