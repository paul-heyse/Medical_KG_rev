#!/usr/bin/env python3
"""Fail if the GraphQL schema has changed relative to the stored SDL."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from Medical_KG_rev.gateway.graphql.schema import schema
except Exception as exc:  # pragma: no cover - missing deps
    print(f"Unable to load schema: {exc}", file=sys.stderr)
    sys.exit(1)

repo_schema = Path("docs/schema.graphql").read_text(encoding="utf-8").strip()
current_schema = schema.as_str().strip()

if repo_schema != current_schema:
    print("GraphQL schema drift detected. Run 'python scripts/update_graphql_schema.py' to refresh.", file=sys.stderr)
    sys.exit(1)

print("GraphQL schema matches stored SDL.")
