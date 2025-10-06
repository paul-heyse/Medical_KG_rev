"""Token bucket rate limiter."""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from ..config.settings import AppSettings, RateLimitSettings, get_settings

logger = structlog.get_logger(__name__)


@dataclass
class TokenBucket:
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float
    updated_at: float

    def consume(self, amount: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.updated_at
        self.updated_at = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


class RateLimitExceeded(RuntimeError):
    """Raised when the caller exceeds their rate limit."""

    def __init__(self, retry_after: float):
        super().__init__("Rate limit exceeded")
        self.retry_after = retry_after


class RateLimiter:
    """Per-identity rate limiter with endpoint overrides."""

    def __init__(self, settings: RateLimitSettings) -> None:
        self.settings = settings
        self._buckets: dict[str, TokenBucket] = {}

    def check(self, identity: str, endpoint: str) -> None:
        limit = self.settings.endpoint_overrides.get(endpoint, self.settings.requests_per_minute)
        burst = max(self.settings.burst, 1)
        key = f"{identity}:{endpoint}"
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(
                capacity=burst, refill_rate=limit / 60.0, tokens=burst, updated_at=time.monotonic()
            )
            self._buckets[key] = bucket
        if not bucket.consume():
            retry = max(1.0, (1 - bucket.tokens) / bucket.refill_rate)
            logger.warning(
                "security.rate_limit_exceeded",
                identity=identity,
                endpoint=endpoint,
                retry_after=retry,
            )
            raise RateLimitExceeded(retry)
        logger.info(
            "security.rate_limit", identity=identity, endpoint=endpoint, remaining=bucket.tokens
        )


def build_rate_limiter(settings: AppSettings | None = None) -> RateLimiter:
    cfg = (settings or get_settings()).security.rate_limit
    return RateLimiter(cfg)
