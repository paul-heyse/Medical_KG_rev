#!/usr/bin/env python3
"""Trigger the retrieval evaluation endpoint and persist summary metrics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from Medical_KG_rev.services.evaluation.test_sets import TestSetManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000", help="Gateway base URL")
    parser.add_argument(
        "--tenant-id", default="eval", help="Tenant identifier used for the evaluation request"
    )
    parser.add_argument("--token", default=None, help="Bearer token for authenticated deployments")
    parser.add_argument(
        "--test-set", default="test_set_v1", help="Name of the packaged evaluation dataset to use"
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional filesystem root containing custom test set YAML files",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of documents to evaluate (Recall@K baseline)"
    )
    parser.add_argument("--rerank", action="store_true", help="Enable reranking during evaluation")
    parser.add_argument("--rerank-model", default=None, help="Optional reranker model override")
    parser.add_argument(
        "--output", default=None, help="File path to persist the evaluation response as JSON"
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "tenant_id": args.tenant_id,
        "test_set_name": args.test_set,
        "top_k": args.top_k,
        "rerank": args.rerank,
        "rerank_model": args.rerank_model,
        "use_cache": False,
    }
    if args.dataset_root:
        manager = TestSetManager(root=Path(args.dataset_root))
        test_set = manager.load(args.test_set)
        payload["test_set_name"] = None
        payload["queries"] = [
            {
                "query_id": record.query_id,
                "query_text": record.query_text,
                "query_type": record.query_type.value,
                "relevant_docs": [
                    {"doc_id": doc_id, "grade": grade} for doc_id, grade in record.relevant_docs
                ],
            }
            for record in test_set.queries
        ]
    return payload


def main() -> None:
    args = parse_args()
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    payload = build_payload(args)
    url = f"{args.base_url.rstrip('/')}/v1/evaluate"
    with httpx.Client(timeout=args.timeout, headers=headers) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        body = response.json()
    if args.output:
        Path(args.output).write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
