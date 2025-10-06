"""Container healthcheck entrypoint for GPU microservices."""

from __future__ import annotations

import argparse
import sys

import structlog

from Medical_KG_rev.services.gpu import GpuManager, GpuNotAvailableError

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU microservice healthcheck")
    parser.add_argument("--service", default="gpu", help="Service name to check")
    parser.add_argument("--timeout", type=float, default=2.0, help="Wait duration in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.service != "gpu":
        logger.warning("healthcheck.unknown_service", service=args.service)
        return 0
    manager = GpuManager()
    try:
        manager.wait_for_gpu(timeout=args.timeout)
    except GpuNotAvailableError as exc:
        logger.error("healthcheck.gpu_unavailable", error=str(exc))
        return 1
    logger.info("healthcheck.ok")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
