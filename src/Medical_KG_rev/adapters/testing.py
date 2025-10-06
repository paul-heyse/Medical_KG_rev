"""Utilities to help test adapters."""

from __future__ import annotations

from collections.abc import Mapping

from .base import AdapterContext, AdapterResult, BaseAdapter


def run_adapter(
    adapter: BaseAdapter,
    *,
    tenant_id: str = "test",
    domain: str = "medical",
    parameters: Mapping[str, object] | None = None,
) -> AdapterResult:
    """Execute adapter using an in-memory context for tests."""

    context = AdapterContext(
        tenant_id=tenant_id,
        domain=domain,
        correlation_id="test-corr",
        parameters=parameters or {},
    )
    return adapter.run(context)
