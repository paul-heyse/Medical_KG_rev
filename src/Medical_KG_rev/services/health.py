"""Infrastructure health checks for the gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Mapping


@dataclass(slots=True)
class CheckResult:
    status: str
    detail: str = ""


HealthCheck = Callable[[], CheckResult]


@dataclass
class HealthService:
    checks: Mapping[str, HealthCheck]
    version: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def uptime_seconds(self) -> float:
        delta = datetime.now(timezone.utc) - self.started_at
        return round(delta.total_seconds(), 3)

    def liveness(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "version": self.version,
            "uptime_seconds": self.uptime_seconds(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def readiness(self) -> Dict[str, object]:
        results: Dict[str, Dict[str, object]] = {}
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


__all__ = ["HealthService", "CheckResult", "success", "failure"]
