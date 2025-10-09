"""Asynchronous hybrid retrieval coordination utilities."""

from __future__ import annotations

import asyncio
import hashlib
import json
import unicodedata
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import structlog
import yaml
from structlog.stdlib import BoundLogger

DEFAULT_COMPONENT_CONFIG = Path("config/retrieval/components.yaml")


class CacheProtocol(Protocol):
    """Minimal cache contract used by the hybrid coordinator."""

    async def get(self, key: str) -> Any:  # pragma: no cover - protocol definition
        ...

    async def set(self, key: str, value: Any, *, ttl: int) -> None:  # pragma: no cover
        ...


class ComponentCallable(Protocol):
    """Callable signature for retrieval components."""

    async def __call__(
        self,
        *,
        index: str,
        query: str,
        k: int,
        filters: Mapping[str, object],
        context: Any | None = None,
    ) -> Sequence[Mapping[str, object]]:  # pragma: no cover - protocol definition
        ...


@dataclass(slots=True)
class HybridComponentSettings:
    """Configuration for the :class:`HybridSearchCoordinator`."""

    enable_splade: bool = True
    enable_dense: bool = True
    enable_query_expansion: bool = False
    timeout_ms: int = 300
    cache_ttl_seconds: int = 300
    default_components: Sequence[str] = field(default_factory=lambda: ("bm25", "splade", "dense"))
    component_timeouts: Mapping[str, int] = field(default_factory=dict)
    synonyms: Mapping[str, Sequence[str]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path | str) -> "HybridComponentSettings":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        defaults: Mapping[str, object] = (
            data.get("defaults", {}) if isinstance(data, Mapping) else {}
        )
        components_cfg: Mapping[str, object] = (
            data.get("components", {}) if isinstance(data, Mapping) else {}
        )
        synonyms_cfg: Mapping[str, Sequence[str]] = (
            data.get("synonyms", {}) if isinstance(data, Mapping) else {}
        )
        return cls(
            enable_splade=bool(defaults.get("enable_splade", True)),
            enable_dense=bool(defaults.get("enable_dense", True)),
            enable_query_expansion=bool(defaults.get("enable_query_expansion", False)),
            timeout_ms=int(defaults.get("timeout_ms", 300)),
            cache_ttl_seconds=int(defaults.get("cache_ttl_seconds", 300)),
            default_components=tuple(
                str(component)
                for component in defaults.get("components", ("bm25", "splade", "dense"))
            ),
            component_timeouts={
                str(name): int(cfg.get("timeout_ms", defaults.get("timeout_ms", 300)))
                for name, cfg in components_cfg.items()
                if isinstance(cfg, Mapping)
            },
            synonyms={
                str(token): tuple(str(value) for value in values)
                for token, values in synonyms_cfg.items()
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
            },
        )

    def timeout_for(self, component: str) -> float:
        timeout_ms = self.component_timeouts.get(component, self.timeout_ms)
        return max(timeout_ms, 50) / 1000.0

    def resolve_components(self, overrides: Sequence[str] | None = None) -> list[str]:
        base = list(overrides or self.default_components)
        resolved: list[str] = []
        for component in base:
            if component == "splade" and not self.enable_splade:
                continue
            if component == "dense" and not self.enable_dense:
                continue
            if component not in resolved:
                resolved.append(component)
        if "bm25" not in resolved:
            resolved.insert(0, "bm25")
        return resolved


@dataclass(slots=True)
class HybridSearchResult:
    component_results: dict[str, Sequence[Mapping[str, object]]]
    component_errors: list[str]
    timings_ms: dict[str, float]
    query: str
    normalized_query: str
    expanded_query: str | None
    correlation_id: str | None
    cache_hit: bool = False

    def to_cache(self) -> dict[str, Any]:
        return {
            "component_results": self.component_results,
            "component_errors": list(self.component_errors),
            "timings_ms": self.timings_ms,
            "query": self.query,
            "normalized_query": self.normalized_query,
            "expanded_query": self.expanded_query,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_cache(cls, payload: Mapping[str, Any]) -> "HybridSearchResult":
        return cls(
            component_results={
                str(name): list(results)
                for name, results in (payload.get("component_results", {}) or {}).items()
            },
            component_errors=[str(item) for item in payload.get("component_errors", []) or []],
            timings_ms={
                str(name): float(duration)
                for name, duration in (payload.get("timings_ms", {}) or {}).items()
            },
            query=str(payload.get("query", "")),
            normalized_query=str(payload.get("normalized_query", "")),
            expanded_query=(
                str(payload.get("expanded_query"))
                if payload.get("expanded_query") is not None
                else None
            ),
            correlation_id=(
                str(payload.get("correlation_id"))
                if payload.get("correlation_id") is not None
                else None
            ),
            cache_hit=True,
        )


class InMemoryHybridCache(CacheProtocol):
    """Lightweight cache implementation used in tests and defaults."""

    def __init__(self) -> None:
        self._store: MutableMapping[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any:
        async with self._lock:
            return self._store.get(key)

    async def set(self, key: str, value: Any, *, ttl: int) -> None:  # pragma: no cover - TTL unused
        async with self._lock:
            self._store[key] = value


class HybridSearchCoordinator:
    """Coordinates concurrent retrieval component execution."""

    def __init__(
        self,
        components: Mapping[str, ComponentCallable],
        *,
        settings: HybridComponentSettings | None = None,
        cache: CacheProtocol | None = None,
        logger: BoundLogger | None = None,
    ) -> None:
        self._components = dict(components)
        self._settings = settings or self._load_default_settings()
        self._cache = cache or InMemoryHybridCache()
        self._logger = logger or structlog.get_logger(__name__)

    @staticmethod
    def _load_default_settings() -> HybridComponentSettings:
        try:
            return HybridComponentSettings.from_file(DEFAULT_COMPONENT_CONFIG)
        except FileNotFoundError:  # pragma: no cover - deployment safety
            return HybridComponentSettings()

    async def search(
        self,
        *,
        index: str,
        query: str,
        k: int,
        filters: Mapping[str, object] | None = None,
        components: Sequence[str] | None = None,
        correlation_id: str | None = None,
        context: Any | None = None,
        cache_scope: str | None = None,
        use_cache: bool = True,
    ) -> HybridSearchResult:
        filters = filters or {}
        normalized_query = self._preprocess_query(query)
        expanded_query = self._expand_query(normalized_query)
        effective_query = expanded_query or normalized_query
        component_list = self._settings.resolve_components(components)
        cache_key = self._cache_key(
            index,
            effective_query,
            k,
            component_list,
            filters,
            cache_scope,
        )
        bound_logger = (
            self._logger.bind(correlation_id=correlation_id) if correlation_id else self._logger
        )
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                result = HybridSearchResult.from_cache(cached)
                bound_logger.debug(
                    "hybrid.cache_hit",
                    index=index,
                    components=component_list,
                    cache_key=cache_key,
                )
                return result

        tasks: dict[str, asyncio.Task[Sequence[Mapping[str, object]]]] = {}
        timings: dict[str, float] = {}
        results: dict[str, Sequence[Mapping[str, object]]] = {}
        errors: list[str] = []
        start_times: dict[str, float] = {}
        for component in component_list:
            handler = self._components.get(component)
            if handler is None:
                errors.append(f"{component}:unavailable")
                results[component] = []
                continue
            timeout = self._settings.timeout_for(component)
            start_times[component] = perf_counter()
            task = asyncio.create_task(
                self._invoke_component(
                    handler,
                    index=index,
                    query=effective_query,
                    k=k,
                    filters=filters,
                    timeout=timeout,
                    component=component,
                    logger=bound_logger,
                    context=context,
                )
            )
            tasks[component] = task

        for component, task in tasks.items():
            try:
                component_results = await task
            except asyncio.TimeoutError:
                errors.append(f"{component}:timeout")
                results[component] = []
            except Exception as exc:  # pragma: no cover - defensive guard
                errors.append(f"{component}:{exc.__class__.__name__}")
                bound_logger.warning(
                    "hybrid.component_failed",
                    component=component,
                    error=str(exc),
                )
                results[component] = []
            else:
                results[component] = component_results
            finally:
                started = start_times.get(component, perf_counter())
                timings[component] = (perf_counter() - started) * 1000.0

        for component in component_list:
            results.setdefault(component, [])
            timings.setdefault(component, 0.0)

        outcome = HybridSearchResult(
            component_results=results,
            component_errors=errors,
            timings_ms=timings,
            query=query,
            normalized_query=normalized_query,
            expanded_query=expanded_query,
            correlation_id=correlation_id,
        )
        if use_cache and not outcome.cache_hit and not errors:
            await self._cache.set(
                cache_key, outcome.to_cache(), ttl=self._settings.cache_ttl_seconds
            )
        return outcome

    def search_sync(
        self,
        *,
        index: str,
        query: str,
        k: int,
        filters: Mapping[str, object] | None = None,
        components: Sequence[str] | None = None,
        correlation_id: str | None = None,
        context: Any | None = None,
        cache_scope: str | None = None,
        use_cache: bool = True,
    ) -> HybridSearchResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.search(
                    index=index,
                    query=query,
                    k=k,
                    filters=filters,
                    components=components,
                    correlation_id=correlation_id,
                    context=context,
                    cache_scope=cache_scope,
                    use_cache=use_cache,
                )
            )
        else:  # pragma: no cover - sync calls should not occur inside event loop
            if loop.is_running():
                raise RuntimeError(
                    "HybridSearchCoordinator.search_sync called from running event loop"
                )
            return loop.run_until_complete(
                self.search(
                    index=index,
                    query=query,
                    k=k,
                    filters=filters,
                    components=components,
                    correlation_id=correlation_id,
                    context=context,
                    cache_scope=cache_scope,
                    use_cache=use_cache,
                )
            )

    async def _invoke_component(
        self,
        handler: ComponentCallable,
        *,
        index: str,
        query: str,
        k: int,
        filters: Mapping[str, object],
        timeout: float,
        component: str,
        logger: BoundLogger,
        context: Any | None,
    ) -> Sequence[Mapping[str, object]]:
        logger.debug(
            "hybrid.component_start",
            component=component,
            timeout=timeout,
            k=k,
        )
        return await asyncio.wait_for(
            handler(index=index, query=query, k=k, filters=filters, context=context),
            timeout,
        )

    def _preprocess_query(self, query: str) -> str:
        normalized = unicodedata.normalize("NFKC", query)
        lowered = normalized.lower()
        tokens = [token for token in lowered.split() if token]
        return " ".join(tokens)

    def _expand_query(self, query: str) -> str | None:
        if not self._settings.enable_query_expansion or not query:
            return None
        tokens = query.split()
        expanded: list[str] = list(tokens)
        for token in tokens:
            expansions = self._settings.synonyms.get(token)
            if not expansions:
                continue
            for synonym in expansions:
                if synonym not in expanded:
                    expanded.append(synonym)
        if expanded == tokens:
            return None
        return " ".join(expanded)

    def _cache_key(
        self,
        index: str,
        query: str,
        k: int,
        components: Sequence[str],
        filters: Mapping[str, object],
        cache_scope: str | None,
    ) -> str:
        payload = {
            "index": index,
            "query": query,
            "k": k,
            "components": list(components),
            "filters": self._serialise_filters(filters),
            "scope": cache_scope,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialise_filters(filters: Mapping[str, object]) -> Mapping[str, object]:
        serialised: dict[str, object] = {}
        for key, value in sorted(filters.items(), key=lambda item: item[0]):
            if isinstance(value, Mapping):
                serialised[key] = HybridSearchCoordinator._serialise_filters(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                serialised[key] = [
                    HybridSearchCoordinator._normalise_scalar(item) for item in value
                ]
            else:
                serialised[key] = HybridSearchCoordinator._normalise_scalar(value)
        return serialised

    @staticmethod
    def _normalise_scalar(value: object) -> object:
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)


__all__ = [
    "CacheProtocol",
    "ComponentCallable",
    "HybridComponentSettings",
    "HybridSearchCoordinator",
    "HybridSearchResult",
    "InMemoryHybridCache",
]
