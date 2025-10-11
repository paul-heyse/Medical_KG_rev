"""Command line helpers for the multi-protocol gateway.

This module provides command-line utilities for the gateway application,
including schema export functionality for OpenAPI, GraphQL, and AsyncAPI
specifications. It serves as the entry point for CLI operations.

Key Responsibilities:
    - Export OpenAPI specification for REST endpoints
    - Export GraphQL schema for GraphQL endpoints
    - Export AsyncAPI specification for SSE endpoints
    - Provide CLI interface for schema generation

Collaborators:
    - Upstream: Command line interface (CLI)
    - Downstream: Gateway application, GraphQL schema

Side Effects:
    - Reads application configuration
    - Generates schema files
    - Writes output to stdout or files

Thread Safety:
    - Thread-safe: CLI operations are single-threaded

Performance Characteristics:
    - O(1) schema generation for static schemas
    - O(n) where n is schema complexity for dynamic schemas

Example:
-------
    >>> python -m Medical_KG_rev.gateway.main --export-openapi
    >>> python -m Medical_KG_rev.gateway.main --export-graphql
    >>> python -m Medical_KG_rev.gateway.main --export-asyncapi

"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================

from pathlib import Path
from typing import Any
import argparse

from yaml import safe_dump

from .app import create_app
from .graphql.schema import schema


# ==============================================================================
# TEMPLATES
# ==============================================================================

ASYNCAPI_TEMPLATE = """
asyncapi: '2.6.0'
info:
  title: Medical KG Job Streams
  version: '0.1.0'
  description: Server-sent event channels for ingestion jobs.
servers:
  development:
    url: http://localhost:8000
    protocol: sse
channels:
  jobs/{{jobId}}:
    subscribe:
      summary: Receive job lifecycle events.
      messages:
        jobEvent:
          payload:
            type: object
            properties:
              job_id:
                type: string
              type:
                type: string
                enum: [jobs.started, jobs.progress, jobs.completed, jobs.failed]
              payload:
                type: object
              emitted_at:
                type: string
                format: date-time
""".strip()


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================


def export_openapi() -> str:
    """Export OpenAPI specification for REST endpoints.

    Returns
    -------
        YAML-formatted OpenAPI specification.

    """
    app = create_app()
    openapi_schema: dict[str, Any] = app.openapi()
    return safe_dump(openapi_schema, sort_keys=False)


def export_graphql() -> str:
    """Export GraphQL schema for GraphQL endpoints.

    Returns
    -------
        GraphQL schema definition language (SDL) string.

    """
    return schema.as_str()


def export_asyncapi() -> str:
    """Export AsyncAPI specification for SSE endpoints.

    Returns
    -------
        YAML-formatted AsyncAPI specification.

    """
    return ASYNCAPI_TEMPLATE


# ==============================================================================
# CLI INTERFACE
# ==============================================================================


def main() -> None:
    """Main CLI entry point for gateway utilities.

    Provides command-line interface for exporting API specifications
    including OpenAPI, GraphQL, and AsyncAPI schemas.
    """
    parser = argparse.ArgumentParser(description="Gateway helper utilities")
    parser.add_argument("--export-openapi", action="store_true", help="Print OpenAPI document")
    parser.add_argument("--export-graphql", action="store_true", help="Print GraphQL SDL")
    parser.add_argument("--export-asyncapi", action="store_true", help="Print AsyncAPI YAML")
    parser.add_argument("--output", type=Path, default=None, help="Optional file path to write")
    args = parser.parse_args()

    if not any([args.export_openapi, args.export_graphql, args.export_asyncapi]):
        parser.error("Choose at least one export option")

    if args.export_openapi:
        content = export_openapi()
    elif args.export_graphql:
        content = export_graphql()
    elif args.export_asyncapi:
        content = export_asyncapi()
    else:  # pragma: no cover - parser guards this scenario
        content = ""

    if args.output:
        args.output.write_text(content)
    else:
        print(content)


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "export_asyncapi",
    "export_graphql",
    "export_openapi",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    main()
