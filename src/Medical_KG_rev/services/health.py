"""Infrastructure health checks for the gateway."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class CheckResult:
    status: str
    detail: str = ""


HealthCheck = Callable[[], CheckResult]


@dataclass
class HealthService:
    checks: Mapping[str, HealthCheck]
    version: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def uptime_seconds(self) -> float:
        delta = datetime.now(UTC) - self.started_at
        return round(delta.total_seconds(), 3)

    def liveness(self) -> dict[str, object]:
        return {
            "status": "ok",
            "version": self.version,
            "uptime_seconds": self.uptime_seconds(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def readiness(self) -> dict[str, object]:
        results: dict[str, dict[str, object]] = {}
        overall_status = "ok"
        for name, check in self.checks.items():
            try:
                result = check()
            except Exception as exc:  # pragma: no cover - defensive
                results[name] = {"status": "error", "detail": str(exc)}
                overall_status = "error"
                continue
            results[name] = {"status": result.status, "detail": result.detail}
            if result.status != "ok" and overall_status != "error":
                overall_status = "degraded"
        payload = self.liveness()
        payload.update({"status": overall_status, "checks": results})
        return payload


def success(detail: str = "") -> CheckResult:
    return CheckResult(status="ok", detail=detail)


def failure(detail: str) -> CheckResult:
    return CheckResult(status="error", detail=detail)


__all__ = ["CheckResult", "HealthService", "failure", "success"]
