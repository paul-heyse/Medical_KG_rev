#!/usr/bin/env python3
"""Export current namespace policy and persister configuration for migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Medical_KG_rev.gateway.services import GatewayService, get_gateway_service
from Medical_KG_rev.services.embedding.persister import PersisterRuntimeSettings
from Medical_KG_rev.services.embedding.policy import NamespacePolicySettings


def _serialise_policy(settings: NamespacePolicySettings) -> dict[str, object]:
    return {
        "cache_ttl_seconds": float(settings.cache_ttl_seconds),
        "max_cache_entries": int(settings.max_cache_entries),
        "dry_run": bool(settings.dry_run),
    }


def _serialise_persister(settings: PersisterRuntimeSettings) -> dict[str, object]:
    return {
        "backend": settings.backend,
        "cache_limit": int(settings.cache_limit),
        "hybrid_backends": dict(settings.hybrid_backends),
    }


def export_configuration(service: GatewayService) -> dict[str, object]:
    """Render the embedding runtime configuration for archival."""
    policy_settings = service.namespace_policy_settings or NamespacePolicySettings()
    persister_settings = service.embedding_persister_settings or PersisterRuntimeSettings()
    return {
        "embedding": {
            "policy": _serialise_policy(policy_settings),
            "persister": _serialise_persister(persister_settings),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the exported configuration as JSON",
    )
    args = parser.parse_args()

    # Instantiate the service to capture runtime configuration.
    service = get_gateway_service()
    payload = export_configuration(service)
    text = json.dumps(payload, indent=2, sort_keys=True)

    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
