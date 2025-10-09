"""Rate limiting utilities for authentication-protected endpoints."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import time
from dataclasses import dataclass

import structlog

from ..config.settings import AppSettings, RateLimitSettings, get_settings

logger = structlog.get_logger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class TokenBucket:
    """Token bucket state used for rate limiting decisions.

    Attributes:
        capacity: Maximum number of tokens that can be stored.
        refill_rate: Tokens added per second.
        tokens: Current token count.
        updated_at: Monotonic timestamp of the last refill or consumption.
    """

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float
    updated_at: float

    def consume(self, amount: int = 1) -> bool:
        """Attempt to consume tokens from the bucket.

        Args:
            amount: Number of tokens required to satisfy the request.

        Returns:
            ``True`` when the bucket contained enough tokens, ``False`` otherwise.
        """

        now = time.monotonic()
        elapsed = now - self.updated_at
        self.updated_at = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


# ============================================================================
# ERRORS
# ============================================================================


class RateLimitExceeded(RuntimeError):
    """Raised when the caller exceeds their rate limit."""

    def __init__(self, retry_after: float):
        """Initialize exception with retry-after seconds.

        Args:
            retry_after: Number of seconds a caller should wait before retrying.
        """

        super().__init__("Rate limit exceeded")
        self.retry_after = retry_after


# ============================================================================
# RATE LIMITER IMPLEMENTATION
# ============================================================================


class RateLimiter:
    """Per-identity rate limiter with endpoint overrides."""

    def __init__(self, settings: RateLimitSettings) -> None:
        """Initialize the limiter with application settings.

        Args:
            settings: Rate limit configuration sourced from application settings.
        """

        self.settings = settings
        self._buckets: dict[str, TokenBucket] = {}

    def check(self, identity: str, endpoint: str) -> None:
        """Ensure the caller is within rate limits for the given endpoint.

        Args:
            identity: Unique caller identifier (user, API key, or tenant).
            endpoint: Endpoint identifier for which rate limits apply.

        Raises:
            RateLimitExceeded: When the caller has exhausted their rate limit.
        """

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


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def build_rate_limiter(settings: AppSettings | None = None) -> RateLimiter:
    """Construct a rate limiter instance from application settings.

    Args:
        settings: Optional application settings override.

    Returns:
        Configured :class:`RateLimiter` ready for dependency injection.
    """

    cfg = (settings or get_settings()).security.rate_limit
    return RateLimiter(cfg)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["RateLimitExceeded", "RateLimiter", "TokenBucket", "build_rate_limiter"]
