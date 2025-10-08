"""Namespace access policy interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from time import perf_counter
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Tuple

import structlog

from Medical_KG_rev.services.embedding.namespace.access import (
    NamespaceAccessResult,
    validate_namespace_access,
)
from Medical_KG_rev.services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from Medical_KG_rev.services.embedding.namespace.schema import NamespaceConfig

from .telemetry import EmbeddingTelemetry

logger = structlog.get_logger(__name__)


@dataclass(slots=True, frozen=True)
class NamespaceAccessDecision:
    """Normalized decision produced by a namespace access policy."""

    namespace: str
    tenant_id: str
    scope: str
    allowed: bool
    reason: str | None = None
    config: NamespaceConfig | None = None
    policy: str = "standard"
    metadata: dict[str, object] = field(default_factory=dict)
    duration_ms: float | None = None

    def denied_due_to_tenant(self) -> bool:
        """Return True if the denial was caused by tenant restrictions."""

        if self.allowed:
            return False
        message = (self.reason or "").lower()
        return "tenant" in message


@dataclass(slots=True)
class NamespacePolicySettings:
    """Runtime configuration for namespace access policies."""

    cache_ttl_seconds: float = 60.0
    max_cache_entries: int = 512
    dry_run: bool = False


@dataclass(slots=True)
class _CacheEntry:
    decision: NamespaceAccessDecision
    expires_at: float


class NamespaceAccessPolicy(ABC):
    """Abstract base class for namespace validation and routing policies."""

    def __init__(
        self,
        registry: EmbeddingNamespaceRegistry,
        *,
        telemetry: EmbeddingTelemetry | None = None,
        settings: NamespacePolicySettings | None = None,
    ) -> None:
        self._registry = registry
        self._telemetry = telemetry
        self._settings = settings or NamespacePolicySettings()
        self._cache: MutableMapping[Tuple[str, str, str], _CacheEntry] = {}
        self._evaluations: int = 0
        self._denials: int = 0
        self._cache_hits: int = 0
        self._logger = logger.bind(policy=self.__class__.__name__)

    @property
    def registry(self) -> EmbeddingNamespaceRegistry:
        return self._registry

    @property
    def settings(self) -> NamespacePolicySettings:
        return self._settings

    def update_settings(self, **kwargs: object) -> None:
        """Hot-update policy configuration."""

        new_values = asdict(self._settings) | kwargs
        self._settings = NamespacePolicySettings(**new_values)
        self.invalidate()

    def evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        """Evaluate access for the supplied namespace/tenant/scope."""

        cache_key = (namespace, tenant_id, required_scope)
        entry = self._cache.get(cache_key)
        now = perf_counter()
        if entry and entry.expires_at > now:
            self._cache_hits += 1
            return entry.decision

        started = perf_counter()
        decision = self._evaluate(namespace=namespace, tenant_id=tenant_id, required_scope=required_scope)
        duration_ms = (perf_counter() - started) * 1000
        decision = replace(decision, duration_ms=duration_ms)

        self._evaluations += 1
        if not decision.allowed:
            self._denials += 1
            if self._telemetry:
                self._telemetry.record_policy_denied(decision)
        if self._telemetry:
            self._telemetry.record_policy_evaluation(decision)

        ttl = max(float(self._settings.cache_ttl_seconds), 0.0)
        if ttl:
            if len(self._cache) >= self._settings.max_cache_entries:
                self._cache.pop(next(iter(self._cache)), None)
            self._cache[cache_key] = _CacheEntry(decision=decision, expires_at=now + ttl)
        return decision

    def invalidate(self, namespace: str | None = None) -> None:
        """Invalidate cached policy decisions."""

        if namespace is None:
            self._cache.clear()
            return
        keys = [key for key in self._cache if key[0] == namespace]
        for key in keys:
            self._cache.pop(key, None)

    def health_status(self) -> Mapping[str, object]:
        """Expose health diagnostics for monitoring/alerting."""

        return {
            "policy": self.__class__.__name__,
            "evaluations": self._evaluations,
            "denials": self._denials,
            "cache_entries": len(self._cache),
        }

    def debug_snapshot(self) -> Mapping[str, object]:
        """Return an introspection snapshot useful during debugging."""

        return {
            "settings": asdict(self._settings),
            "cache_keys": [key for key in self._cache],
            "stats": self.stats(),
        }

    def stats(self) -> Mapping[str, object]:
        """Return current policy statistics."""

        return {
            "evaluations": self._evaluations,
            "denials": self._denials,
            "cache_hits": self._cache_hits,
        }

    @abstractmethod
    def _evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        """Perform the actual policy evaluation."""

    def operational_metrics(self) -> Mapping[str, object]:
        """Return metrics suitable for operational monitoring."""

        return {
            "policy": self.__class__.__name__,
            "evaluations_performed": self._evaluations,
            "denials": self._denials,
            "cache_hit_ratio": self._cache_hits / self._evaluations if self._evaluations else 0.0,
        }


class StandardNamespacePolicy(NamespaceAccessPolicy):
    """Default namespace access policy backed by the namespace registry."""

    def _evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        try:
            config = self.registry.get(namespace)
        except ValueError as exc:
            metadata: Dict[str, object] = {"error": str(exc)}
            return NamespaceAccessDecision(
                namespace=namespace,
                tenant_id=tenant_id,
                scope=required_scope,
                allowed=False,
                reason=str(exc),
                metadata=metadata,
            )

        result: NamespaceAccessResult = validate_namespace_access(
            self.registry,
            namespace=namespace,
            tenant_id=tenant_id,
            required_scope=required_scope,
        )
        metadata = {
            "allowed_tenants": list(config.allowed_tenants),
            "allowed_scopes": list(config.allowed_scopes),
            "provider": config.provider,
            "kind": config.kind,
        }
        return NamespaceAccessDecision(
            namespace=namespace,
            tenant_id=tenant_id,
            scope=required_scope,
            allowed=result.allowed,
            reason=result.reason,
            config=config,
            metadata=metadata,
            policy="standard",
        )


class DryRunNamespacePolicy(NamespaceAccessPolicy):
    """Policy variant that records denials but never blocks callers."""

    def __init__(
        self,
        delegate: NamespaceAccessPolicy,
        *,
        telemetry: EmbeddingTelemetry | None = None,
    ) -> None:
        super().__init__(
            delegate.registry,
            telemetry=telemetry or delegate._telemetry,
            settings=delegate.settings,
        )
        self._delegate = delegate

    def _evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        decision = self._delegate.evaluate(
            namespace=namespace,
            tenant_id=tenant_id,
            required_scope=required_scope,
        )
        if decision.allowed:
            return replace(decision, policy="dry_run")
        metadata = dict(decision.metadata)
        metadata["dry_run_denied"] = True
        return NamespaceAccessDecision(
            namespace=decision.namespace,
            tenant_id=decision.tenant_id,
            scope=decision.scope,
            allowed=True,
            reason=decision.reason,
            config=decision.config,
            metadata=metadata,
            policy="dry_run",
        )


class MockNamespacePolicy(NamespaceAccessPolicy):
    """Testing policy that uses a predefined decision map."""

    def __init__(
        self,
        registry: EmbeddingNamespaceRegistry,
        decisions: Mapping[Tuple[str, str, str], NamespaceAccessDecision] | None = None,
    ) -> None:
        super().__init__(registry)
        self._decisions = dict(decisions or {})

    def register_decision(
        self,
        *,
        namespace: str,
        tenant_id: str,
        scope: str,
        allowed: bool,
        reason: str | None = None,
    ) -> None:
        config = None
        try:
            config = self.registry.get(namespace)
        except ValueError:
            config = None
        decision = NamespaceAccessDecision(
            namespace=namespace,
            tenant_id=tenant_id,
            scope=scope,
            allowed=allowed,
            reason=reason,
            config=config,
            policy="mock",
        )
        self._decisions[(namespace, tenant_id, scope)] = decision

    def _evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        key = (namespace, tenant_id, required_scope)
        if key not in self._decisions:
            raise KeyError(f"Mock policy missing decision for {key}")
        return self._decisions[key]


class CustomNamespacePolicy(NamespaceAccessPolicy):
    """Policy that delegates to a custom callable for organization rules."""

    def __init__(
        self,
        registry: EmbeddingNamespaceRegistry,
        resolver: Callable[[str, str, str], NamespaceAccessDecision],
        *,
        telemetry: EmbeddingTelemetry | None = None,
        settings: NamespacePolicySettings | None = None,
    ) -> None:
        super().__init__(registry, telemetry=telemetry, settings=settings)
        self._resolver = resolver

    def _evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        decision = self._resolver(namespace, tenant_id, required_scope)
        if decision.config is None:
            try:
                config = self.registry.get(namespace)
            except ValueError:
                config = None
            if config is not None:
                decision = replace(decision, config=config)
        return decision


def build_policy_chain(
    registry: EmbeddingNamespaceRegistry,
    *,
    telemetry: EmbeddingTelemetry | None = None,
    settings: NamespacePolicySettings | None = None,
    dry_run: bool = False,
    extra_policies: Iterable[Callable[[NamespaceAccessPolicy], NamespaceAccessPolicy]] | None = None,
) -> NamespaceAccessPolicy:
    """Build a composed namespace access policy."""

    policy: NamespaceAccessPolicy = StandardNamespacePolicy(registry, telemetry=telemetry, settings=settings)
    if dry_run:
        policy = DryRunNamespacePolicy(policy, telemetry=telemetry)
    for factory in extra_policies or ():
        policy = factory(policy)
    return policy


__all__ = [
    "NamespaceAccessDecision",
    "NamespaceAccessPolicy",
    "NamespacePolicySettings",
    "StandardNamespacePolicy",
    "DryRunNamespacePolicy",
    "MockNamespacePolicy",
    "CustomNamespacePolicy",
    "build_policy_chain",
]
