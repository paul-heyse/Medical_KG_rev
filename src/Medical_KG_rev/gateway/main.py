"""Command line helpers for the multi-protocol gateway."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from yaml import safe_dump

from .app import create_app
from .graphql.schema import schema

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


def export_openapi() -> str:
    app = create_app()
    openapi_schema: dict[str, Any] = app.openapi()
    return safe_dump(openapi_schema, sort_keys=False)


def export_graphql() -> str:
    return schema.as_str()


def export_asyncapi() -> str:
    return ASYNCAPI_TEMPLATE


def main() -> None:
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


if __name__ == "__main__":  # pragma: no cover
    main()
