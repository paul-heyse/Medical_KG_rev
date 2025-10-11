"""GPU resource optimiser shim for torch-free deployments.

The original implementation depended heavily on NVML bindings and the legacy
GPU microservices. For the torch-free fork we retain the public entrypoints so
that CLI wrappers continue to function, but the optimiser now only logs basic
heartbeat information. This keeps the module importable and syntactically valid
without reintroducing the full GPU stack.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Optional


class GPUResourceOptimizer:
    """Minimal no-op optimiser that preserves the public API surface."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

    async def start_optimization_loop(self, interval_seconds: int) -> None:
        """Periodically emit a heartbeat message."""
        if self._running:
            self._logger.debug("optimizer.already_running")
            return

        self._running = True
        self._logger.info("optimizer.loop.start", interval=interval_seconds)

        try:
            while self._running:
                self._logger.debug("optimizer.loop.heartbeat")
                await asyncio.sleep(interval_seconds)
        finally:
            self._running = False
            self._logger.info("optimizer.loop.stop")

    def stop_optimization_loop(self) -> None:
        """Request loop shutdown."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    def run(self, interval_seconds: int) -> None:
        """Run the optimiser loop synchronously."""
        try:
            asyncio.run(self.start_optimization_loop(interval_seconds))
        except KeyboardInterrupt:
            self.stop_optimization_loop()


def main() -> None:
    """CLI entrypoint mirroring the legacy interface."""
    parser = argparse.ArgumentParser(description="GPU Resource Optimizer (stub)")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Optimization interval in seconds",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    optimizer = GPUResourceOptimizer()
    optimizer.run(args.interval)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
