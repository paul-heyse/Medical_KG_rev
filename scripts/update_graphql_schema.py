#!/usr/bin/env python3
"""Regenerate the GraphQL SDL and write it to docs/schema.graphql."""

from __future__ import annotations

from pathlib import Path

from Medical_KG_rev.gateway.graphql.schema import schema

Path("docs/schema.graphql").write_text(schema.as_str(), encoding="utf-8")
print("Updated docs/schema.graphql")
