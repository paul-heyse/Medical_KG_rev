"""Tenant-aware reranking policy with deterministic A/B assignments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(slots=True)
class RerankDecision:
    """Outcome of evaluating whether a request should be reranked."""

    enabled: bool
    cohort: str
    reason: str

    def as_metadata(self) -> dict[str, object]:
        return {"enabled": self.enabled, "cohort": self.cohort, "reason": self.reason}


@dataclass(slots=True)
class TenantRerankPolicy:
    """Encapsulates tenant defaults and experimentation for reranking."""

    default_enabled: bool = False
    tenant_defaults: Mapping[str, bool] = field(default_factory=dict)
    experiment_ratio: float = 0.0

    @classmethod
    def from_file(cls, path: str | Path | None) -> TenantRerankPolicy:
        if path is None:
            return cls()
        candidate = Path(path)
        if not candidate.exists():
            return cls()
        payload = yaml.safe_load(candidate.read_text("utf-8")) or {}
        default_enabled = bool(payload.get("default_enabled", False))
        tenant_defaults = {
            str(key): bool(value) for key, value in (payload.get("tenants") or {}).items()
        }
        experiment = payload.get("experiment") or {}
        ratio = float(experiment.get("rerank_ratio", 0.0))
        ratio = max(0.0, min(1.0, ratio))
        return cls(
            default_enabled=default_enabled,
            tenant_defaults=tenant_defaults,
            experiment_ratio=ratio,
        )

    def decide(
        self,
        tenant_id: str,
        query: str,
        explicit: bool | None,
    ) -> RerankDecision:
        if explicit is not None:
            return RerankDecision(bool(explicit), "override", "request")
        if tenant_id in self.tenant_defaults:
            enabled = bool(self.tenant_defaults[tenant_id])
            cohort = f"tenant:{tenant_id}:{'on' if enabled else 'off'}"
            return RerankDecision(enabled, cohort, "tenant-config")
        if self.default_enabled:
            return RerankDecision(True, "default:on", "global-config")
        if self.experiment_ratio <= 0:
            return RerankDecision(False, "default:off", "global-config")
        seed = f"{tenant_id}:{query}".encode("utf-8")
        digest = hashlib.blake2b(seed, digest_size=8).digest()
        # Map digest to a floating point value in [0, 1).
        threshold = int.from_bytes(digest, "big") / float(1 << (8 * len(digest)))
        enabled = threshold < self.experiment_ratio
        cohort = "experiment:rerank" if enabled else "experiment:control"
        return RerankDecision(enabled, cohort, "experiment")
