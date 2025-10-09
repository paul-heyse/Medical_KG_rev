"""Namespace access policy interfaces and implementations.

This module provides namespace access control policies for embedding operations,
implementing tenant isolation, scope validation, and caching strategies. It
supports multiple policy implementations including standard, dry-run, mock,
and custom policies with configurable settings.

Key Responsibilities:
    - Define namespace access policy interfaces and decision models
    - Implement tenant isolation and scope validation
    - Provide caching for policy decisions with TTL
    - Support multiple policy implementations (standard, dry-run, mock, custom)
    - Integrate with telemetry for monitoring and alerting
    - Provide policy composition and chaining capabilities

Collaborators:
    - Upstream: Embedding coordinators, namespace registry
    - Downstream: Telemetry system, namespace access validation

Side Effects:
    - Updates policy evaluation counters and cache
    - Emits telemetry events for policy decisions
    - May perform I/O operations for namespace validation

Thread Safety:
    - Thread-safe: All operations use atomic updates and immutable decisions
    - Cache operations are designed for concurrent access

Performance Characteristics:
    - O(1) cache lookup time for policy decisions
    - Configurable cache TTL and size limits
    - Policy evaluation time depends on implementation complexity

Example:
    >>> from Medical_KG_rev.services.embedding.policy import build_policy_chain
    >>> policy = build_policy_chain(registry, dry_run=True)
    >>> decision = policy.evaluate(namespace="medical", tenant_id="tenant1", required_scope="read")
    >>> print(f"Access allowed: {decision.allowed}")
"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from time import perf_counter
from typing import Callable, Iterable, Mapping, MutableMapping, Tuple

import structlog
from Medical_KG_rev.services.embedding.namespace.access import (
    NamespaceAccessResult,
    validate_namespace_access,
)
from Medical_KG_rev.services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from Medical_KG_rev.services.embedding.namespace.schema import NamespaceConfig

from .telemetry import EmbeddingTelemetry

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = structlog.get_logger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(slots=True, frozen=True)
class NamespaceAccessDecision:
    """Normalized decision produced by a namespace access policy.

    Represents the result of a namespace access policy evaluation,
    containing all relevant information about the decision including
    access status, reasoning, and metadata.

    Attributes:
        namespace: Namespace name that was evaluated
        tenant_id: Tenant requesting access
        scope: Required scope for the operation
        allowed: Whether access is granted
        reason: Optional explanation for the decision
        config: Namespace configuration if available
        policy: Policy name that made the decision
        metadata: Additional decision metadata
        duration_ms: Time taken to evaluate the policy

    Invariants:
        - namespace, tenant_id, and scope are never empty
        - duration_ms is non-negative when provided
        - metadata is immutable after creation

    Example:
        >>> decision = NamespaceAccessDecision(
        ...     namespace="medical", tenant_id="tenant1", scope="read",
        ...     allowed=True, reason="Tenant has read access"
        ... )
        >>> print(f"Access granted: {decision.allowed}")
    """

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
        """Check if the denial was caused by tenant restrictions.

        Returns:
            True if the denial reason contains "tenant", False otherwise.

        Note:
            This is a heuristic check based on the reason text.
            Returns False for allowed decisions regardless of reason.

        Example:
            >>> decision = NamespaceAccessDecision(
            ...     namespace="medical", tenant_id="tenant1", scope="read",
            ...     allowed=False, reason="Tenant not authorized"
            ... )
            >>> assert decision.denied_due_to_tenant() == True
        """
        if self.allowed:
            return False
        message = (self.reason or "").lower()
        return "tenant" in message


@dataclass(slots=True)
class NamespacePolicySettings:
    """Runtime configuration for namespace access policies.

    Controls policy behavior including caching, dry-run mode,
    and performance characteristics.

    Attributes:
        cache_ttl_seconds: Time-to-live for cached decisions in seconds.
            Defaults to 60.0 seconds.
        max_cache_entries: Maximum number of cached decisions.
            Defaults to 512 entries.
        dry_run: Whether to run in dry-run mode (never deny access).
            Defaults to False.

    Example:
        >>> settings = NamespacePolicySettings(
        ...     cache_ttl_seconds=120.0, max_cache_entries=1024
        ... )
        >>> policy = StandardNamespacePolicy(registry, settings=settings)
    """

    cache_ttl_seconds: float = 60.0
    max_cache_entries: int = 512
    dry_run: bool = False


@dataclass(slots=True)
class _CacheEntry:
    """Internal cache entry for policy decisions.

    Represents a cached policy decision with expiration time
    for TTL-based cache management.

    Attributes:
        decision: The cached policy decision
        expires_at: Timestamp when the cache entry expires
    """

    decision: NamespaceAccessDecision
    expires_at: float


# ============================================================================
# INTERFACE (Protocols/ABCs)
# ============================================================================


class NamespaceAccessPolicy(ABC):
    """Abstract base class for namespace validation and routing policies.

    Defines the interface for namespace access control policies with
    caching, telemetry integration, and configurable settings. Provides
    common functionality for policy evaluation, caching, and monitoring.

    Attributes:
        _registry: Namespace registry for configuration lookup
        _telemetry: Optional telemetry system for monitoring
        _settings: Policy configuration and settings
        _cache: Internal cache for policy decisions
        _evaluations: Counter for total policy evaluations
        _denials: Counter for denied access attempts
        _cache_hits: Counter for cache hits
        _logger: Structured logger bound to policy name

    Invariants:
        - _registry is never None
        - _settings is never None
        - Counters are always non-negative
        - Cache entries respect TTL settings

    Thread Safety:
        - Thread-safe: All operations use atomic updates
        - Cache operations are designed for concurrent access

    Lifecycle:
        - Initialized with registry and optional dependencies
        - Settings can be updated dynamically
        - Cache can be invalidated on demand
        - Statistics accumulate over policy lifetime

    Example:
        >>> class CustomPolicy(NamespaceAccessPolicy):
        ...     def _evaluate(self, *, namespace, tenant_id, required_scope):
        ...         # Custom implementation
        ...         return NamespaceAccessDecision(...)
        >>> policy = CustomPolicy(registry)
        >>> decision = policy.evaluate(namespace="test", tenant_id="t1", required_scope="read")
    """

    def __init__(
        self,
        registry: EmbeddingNamespaceRegistry,
        *,
        telemetry: EmbeddingTelemetry | None = None,
        settings: NamespacePolicySettings | None = None,
    ) -> None:
        """Initialize policy with registry and optional dependencies.

        Args:
            registry: Namespace registry for configuration lookup
            telemetry: Optional telemetry system for monitoring
            settings: Optional policy configuration settings

        Note:
            Logger is automatically bound to the concrete class name
            for structured logging identification.
        """
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
        """Get the namespace registry.

        Returns:
            Registry instance for namespace configuration lookup.
        """
        return self._registry

    @property
    def settings(self) -> NamespacePolicySettings:
        """Get current policy settings.

        Returns:
            Current policy configuration including cache and dry-run settings.
        """
        return self._settings

    def update_settings(self, **kwargs: object) -> None:
        """Update policy settings with new values.

        Args:
            **kwargs: Settings to update. Valid keys: cache_ttl_seconds,
                max_cache_entries, dry_run.

        Note:
            Only provided settings are updated; others remain unchanged.
            Cache is invalidated after settings update.
        """
        new_values = asdict(self._settings) | kwargs
        self._settings = NamespacePolicySettings(**new_values)
        self.invalidate()

    def evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        """Evaluate access for the supplied namespace/tenant/scope.

        Args:
            namespace: Namespace name to evaluate access for
            tenant_id: Tenant requesting access
            required_scope: Required scope for the operation

        Returns:
            Policy decision with access status, reasoning, and metadata

        Note:
            This method implements caching with TTL and integrates with
            telemetry for monitoring. The actual policy logic is delegated
            to the _evaluate method implemented by subclasses.

        Example:
            >>> policy = StandardNamespacePolicy(registry)
            >>> decision = policy.evaluate(
            ...     namespace="medical", tenant_id="tenant1", required_scope="read"
            ... )
            >>> print(f"Access allowed: {decision.allowed}")
        """
        cache_key = (namespace, tenant_id, required_scope)
        entry = self._cache.get(cache_key)
        now = perf_counter()

        # Check cache for valid entry
        if entry and entry.expires_at > now:
            self._cache_hits += 1
            return entry.decision

        # Evaluate policy and measure duration
        started = perf_counter()
        decision = self._evaluate(namespace=namespace, tenant_id=tenant_id, required_scope=required_scope)
        duration_ms = (perf_counter() - started) * 1000
        decision = replace(decision, duration_ms=duration_ms)

        # Update counters and telemetry
        self._evaluations += 1
        if not decision.allowed:
            self._denials += 1
            if self._telemetry:
                self._telemetry.record_policy_denied(decision)
        if self._telemetry:
            self._telemetry.record_policy_evaluation(decision)

        # Cache the decision if TTL is enabled
        ttl = max(float(self._settings.cache_ttl_seconds), 0.0)
        if ttl:
            if len(self._cache) >= self._settings.max_cache_entries:
                # Remove oldest entry if cache is full
                self._cache.pop(next(iter(self._cache)), None)
            self._cache[cache_key] = _CacheEntry(decision=decision, expires_at=now + ttl)
        return decision

    def invalidate(self, namespace: str | None = None) -> None:
        """Invalidate cached policy decisions.

        Args:
            namespace: Optional namespace to invalidate. If None,
                invalidates all cached decisions.

        Note:
            This method is called automatically when settings are updated
            to ensure cache consistency.
        """
        if namespace is None:
            self._cache.clear()
            return
        keys = [key for key in self._cache if key[0] == namespace]
        for key in keys:
            self._cache.pop(key, None)

    def health_status(self) -> Mapping[str, object]:
        """Get health diagnostics for monitoring and alerting.

        Returns:
            Dictionary containing policy health information including
            evaluation counts, denial counts, and cache status.

        Example:
            >>> policy = StandardNamespacePolicy(registry)
            >>> health = policy.health_status()
            >>> print(f"Policy: {health['policy']}, Evaluations: {health['evaluations']}")
        """
        return {
            "policy": self.__class__.__name__,
            "evaluations": self._evaluations,
            "denials": self._denials,
            "cache_entries": len(self._cache),
        }

    def debug_snapshot(self) -> Mapping[str, object]:
        """Get introspection snapshot useful during debugging.

        Returns:
            Dictionary containing detailed policy state including
            settings, cache keys, and statistics.

        Note:
            This method is intended for debugging and development,
            not for production monitoring.
        """
        return {
            "settings": asdict(self._settings),
            "cache_keys": [key for key in self._cache],
            "stats": self.stats(),
        }

    def stats(self) -> Mapping[str, object]:
        """Get current policy statistics.

        Returns:
            Dictionary containing policy performance statistics
            including evaluation counts and cache hit information.

        Example:
            >>> policy = StandardNamespacePolicy(registry)
            >>> stats = policy.stats()
            >>> print(f"Cache hits: {stats['cache_hits']}")
        """
        return {
            "evaluations": self._evaluations,
            "denials": self._denials,
            "cache_hits": self._cache_hits,
        }

    @abstractmethod
    def _evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        """Perform the actual policy evaluation.

        Args:
            namespace: Namespace name to evaluate access for
            tenant_id: Tenant requesting access
            required_scope: Required scope for the operation

        Returns:
            Policy decision with access status, reasoning, and metadata

        Note:
            This method must be implemented by subclasses to provide
            the actual policy logic. The base class handles caching,
            telemetry, and statistics.
        """

    def operational_metrics(self) -> Mapping[str, object]:
        """Get metrics suitable for operational monitoring.

        Returns:
            Dictionary containing operational metrics including
            policy name, evaluation counts, denial counts, and
            cache hit ratio.

        Example:
            >>> policy = StandardNamespacePolicy(registry)
            >>> metrics = policy.operational_metrics()
            >>> print(f"Cache hit ratio: {metrics['cache_hit_ratio']:.2f}")
        """
        return {
            "policy": self.__class__.__name__,
            "evaluations_performed": self._evaluations,
            "denials": self._denials,
            "cache_hit_ratio": self._cache_hits / self._evaluations if self._evaluations else 0.0,
        }


class StandardNamespacePolicy(NamespaceAccessPolicy):
    """Default namespace access policy backed by the namespace registry.

    This policy implementation validates namespace access based on
    configuration stored in the embedding namespace registry. It checks
    namespace existence, enabled status, tenant authorization, and scope
    permissions.

    Attributes:
        Inherits all attributes from NamespaceAccessPolicy

    Invariants:
        - Registry must contain valid namespace configurations
        - Tenant IDs and scopes must be properly configured
        - Disabled namespaces are always denied

    Thread Safety:
        - Thread-safe: Inherits thread safety from base class
        - Registry lookups are read-only operations

    Example:
        >>> registry = EmbeddingNamespaceRegistry()
        >>> policy = StandardNamespacePolicy(registry)
        >>> decision = policy.evaluate(
        ...     namespace="medical", tenant_id="tenant1", required_scope="read"
        ... )
        >>> print(f"Access allowed: {decision.allowed}")
    """

    def _evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        """Evaluate access using the namespace registry configuration.

        Args:
            namespace: Namespace name to evaluate access for
            tenant_id: Tenant requesting access
            required_scope: Required scope for the operation

        Returns:
            Policy decision with access status and detailed reasoning

        Note:
            This method performs the following checks in order:
            1. Namespace existence in registry
            2. Namespace enabled status
            3. Tenant authorization
            4. Scope permissions

        Example:
            >>> policy = StandardNamespacePolicy(registry)
            >>> decision = policy._evaluate(
            ...     namespace="medical", tenant_id="tenant1", required_scope="read"
            ... )
            >>> print(f"Reason: {decision.reason}")
        """
        try:
            config = self.registry.get(namespace)
        except ValueError as exc:
            metadata: dict[str, object] = {"error": str(exc)}
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
    """Policy variant that records denials but never blocks callers.

    This policy wraps another policy and always grants access, even when
    the underlying policy would deny it. This is useful for testing,
    development, and gradual rollout scenarios where you want to monitor
    what would be denied without actually blocking requests.

    Attributes:
        _delegate: The underlying policy to evaluate
        Inherits all other attributes from NamespaceAccessPolicy

    Invariants:
        - Delegate policy is never None
        - Always returns allowed=True
        - Records dry-run denials in metadata

    Thread Safety:
        - Thread-safe: Inherits thread safety from base class
        - Delegate policy evaluation is thread-safe

    Example:
        >>> base_policy = StandardNamespacePolicy(registry)
        >>> dry_run_policy = DryRunNamespacePolicy(base_policy)
        >>> decision = dry_run_policy.evaluate(
        ...     namespace="medical", tenant_id="tenant1", required_scope="read"
        ... )
        >>> print(f"Always allowed: {decision.allowed}")
    """

    def __init__(
        self,
        delegate: NamespaceAccessPolicy,
        *,
        telemetry: EmbeddingTelemetry | None = None,
    ) -> None:
        """Initialize dry-run policy with delegate.

        Args:
            delegate: The underlying policy to evaluate
            telemetry: Optional telemetry system for monitoring

        Note:
            The dry-run policy inherits the delegate's registry and
            settings, but can override the telemetry system.
        """
        super().__init__(
            delegate.registry,
            telemetry=telemetry or delegate._telemetry,
            settings=delegate.settings,
        )
        self._delegate = delegate

    def _evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        """Evaluate access using delegate policy but always grant access.

        Args:
            namespace: Namespace name to evaluate access for
            tenant_id: Tenant requesting access
            required_scope: Required scope for the operation

        Returns:
            Policy decision that is always allowed, with metadata indicating
            if the underlying policy would have denied access

        Note:
            This method evaluates the delegate policy but always returns
            allowed=True. If the delegate would deny access, the metadata
            includes "dry_run_denied": True to indicate this was a dry-run
            denial.

        Example:
            >>> dry_run_policy = DryRunNamespacePolicy(base_policy)
            >>> decision = dry_run_policy._evaluate(
            ...     namespace="medical", tenant_id="tenant1", required_scope="read"
            ... )
            >>> print(f"Dry-run denied: {decision.metadata.get('dry_run_denied', False)}")
        """
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
    """Testing policy that uses a predefined decision map.

    This policy is designed for testing and development scenarios where
    you need predictable policy decisions. It maintains a mapping of
    (namespace, tenant_id, scope) tuples to predefined decisions.

    Attributes:
        _decisions: Mapping of (namespace, tenant_id, scope) to decisions
        Inherits all other attributes from NamespaceAccessPolicy

    Invariants:
        - Decisions map is never None
        - All registered decisions are valid NamespaceAccessDecision objects
        - Missing decisions raise KeyError

    Thread Safety:
        - Thread-safe: Inherits thread safety from base class
        - Decision map operations are thread-safe

    Example:
        >>> mock_policy = MockNamespacePolicy(registry)
        >>> mock_policy.register_decision(
        ...     namespace="test", tenant_id="t1", scope="read", allowed=True
        ... )
        >>> decision = mock_policy.evaluate(
        ...     namespace="test", tenant_id="t1", required_scope="read"
        ... )
        >>> print(f"Mock decision: {decision.allowed}")
    """

    def __init__(
        self,
        registry: EmbeddingNamespaceRegistry,
        decisions: Mapping[Tuple[str, str, str], NamespaceAccessDecision] | None = None,
    ) -> None:
        """Initialize mock policy with registry and optional decisions.

        Args:
            registry: Namespace registry for configuration lookup
            decisions: Optional mapping of (namespace, tenant_id, scope) to decisions

        Note:
            If no decisions are provided, an empty mapping is used.
            Decisions can be added later using register_decision.
        """
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
        """Register a predefined decision for a specific namespace/tenant/scope.

        Args:
            namespace: Namespace name
            tenant_id: Tenant ID
            scope: Required scope
            allowed: Whether access should be allowed
            reason: Optional reason for the decision

        Note:
            This method creates a NamespaceAccessDecision and stores it
            in the decisions map. If the namespace exists in the registry,
            the config is included in the decision.

        Example:
            >>> mock_policy = MockNamespacePolicy(registry)
            >>> mock_policy.register_decision(
            ...     namespace="test", tenant_id="t1", scope="read",
            ...     allowed=True, reason="Test access"
            ... )
        """
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
        """Evaluate access using predefined decisions.

        Args:
            namespace: Namespace name to evaluate access for
            tenant_id: Tenant requesting access
            required_scope: Required scope for the operation

        Returns:
            Predefined policy decision for the given parameters

        Raises:
            KeyError: If no decision is registered for the given parameters

        Note:
            This method looks up the decision in the predefined decisions map.
            If no decision is found, it raises a KeyError.

        Example:
            >>> mock_policy = MockNamespacePolicy(registry)
            >>> mock_policy.register_decision(
            ...     namespace="test", tenant_id="t1", scope="read", allowed=True
            ... )
            >>> decision = mock_policy._evaluate(
            ...     namespace="test", tenant_id="t1", required_scope="read"
            ... )
            >>> print(f"Decision: {decision.allowed}")
        """
        key = (namespace, tenant_id, required_scope)
        if key not in self._decisions:
            raise KeyError(f"Mock policy missing decision for {key}")
        return self._decisions[key]


class CustomNamespacePolicy(NamespaceAccessPolicy):
    """Policy that delegates to a custom callable for organization rules.

    This policy allows organizations to implement custom access control
    logic by providing a callable that takes (namespace, tenant_id, scope)
    and returns a NamespaceAccessDecision.

    Attributes:
        _resolver: Custom callable for policy evaluation
        Inherits all other attributes from NamespaceAccessPolicy

    Invariants:
        - Resolver callable is never None
        - Resolver must return valid NamespaceAccessDecision objects
        - Registry is used to fill in missing config information

    Thread Safety:
        - Thread-safe: Inherits thread safety from base class
        - Resolver callable must be thread-safe

    Example:
        >>> def custom_resolver(namespace, tenant_id, scope):
        ...     return NamespaceAccessDecision(
        ...         namespace=namespace, tenant_id=tenant_id, scope=scope,
        ...         allowed=True, reason="Custom logic"
        ...     )
        >>> policy = CustomNamespacePolicy(registry, custom_resolver)
        >>> decision = policy.evaluate(
        ...     namespace="test", tenant_id="t1", required_scope="read"
        ... )
    """

    def __init__(
        self,
        registry: EmbeddingNamespaceRegistry,
        resolver: Callable[[str, str, str], NamespaceAccessDecision],
        *,
        telemetry: EmbeddingTelemetry | None = None,
        settings: NamespacePolicySettings | None = None,
    ) -> None:
        """Initialize custom policy with registry and resolver.

        Args:
            registry: Namespace registry for configuration lookup
            resolver: Callable that takes (namespace, tenant_id, scope) and
                returns a NamespaceAccessDecision
            telemetry: Optional telemetry system for monitoring
            settings: Optional policy configuration settings

        Note:
            The resolver callable is responsible for implementing the
            custom access control logic.
        """
        super().__init__(registry, telemetry=telemetry, settings=settings)
        self._resolver = resolver

    def _evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        """Evaluate access using custom resolver callable.

        Args:
            namespace: Namespace name to evaluate access for
            tenant_id: Tenant requesting access
            required_scope: Required scope for the operation

        Returns:
            Policy decision from the custom resolver, with config filled in
            if missing

        Note:
            This method calls the custom resolver and then fills in the
            config information from the registry if it's missing from
            the resolver's decision.

        Example:
            >>> def custom_resolver(namespace, tenant_id, scope):
            ...     return NamespaceAccessDecision(
            ...         namespace=namespace, tenant_id=tenant_id, scope=scope,
            ...         allowed=True, reason="Custom logic"
            ...     )
            >>> policy = CustomNamespacePolicy(registry, custom_resolver)
            >>> decision = policy._evaluate(
            ...     namespace="test", tenant_id="t1", required_scope="read"
            ... )
        """
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
    """Build a composed namespace access policy.

    Args:
        registry: Namespace registry for configuration lookup
        telemetry: Optional telemetry system for monitoring
        settings: Optional policy configuration settings
        dry_run: Whether to wrap the policy in a dry-run wrapper
        extra_policies: Optional iterable of policy factory functions

    Returns:
        Composed namespace access policy with optional wrappers

    Note:
        This function builds a policy chain starting with StandardNamespacePolicy
        and optionally wrapping it with DryRunNamespacePolicy and any extra
        policies provided by the extra_policies parameter.

    Example:
        >>> registry = EmbeddingNamespaceRegistry()
        >>> policy = build_policy_chain(
        ...     registry, dry_run=True, telemetry=telemetry
        ... )
        >>> decision = policy.evaluate(
        ...     namespace="test", tenant_id="t1", required_scope="read"
        ... )
    """
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
