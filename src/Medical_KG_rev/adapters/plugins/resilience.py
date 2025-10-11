"""Adapter resilience helpers (placeholder)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResilienceConfig:
    retry_attempts: int = 3


class CircuitBreaker:
    def record_success(self) -> None:
        pass

    def record_failure(self, exc: Exception) -> None:
        pass


def circuit_breaker(func):  # type: ignore
    return func


def retry_on_failure(func):  # type: ignore
    return func


__all__ = [
    "ResilienceConfig",
    "CircuitBreaker",
    "circuit_breaker",
    "retry_on_failure",
]
