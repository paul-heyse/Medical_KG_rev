"""Utilities for parsing OData style query parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request


@dataclass(slots=True)
class ODataParams:
    select: list[str] | None = None
    expand: list[str] | None = None
    filter: str | None = None
    top: int | None = None
    skip: int | None = None

    @classmethod
    def from_request(cls, request: Request) -> ODataParams:
        params: dict[str, Any] = {}
        qp = request.query_params
        if "$select" in qp:
            params["select"] = [
                value.strip() for value in qp["$select"].split(",") if value.strip()
            ]
        if "$expand" in qp:
            params["expand"] = [
                value.strip() for value in qp["$expand"].split(",") if value.strip()
            ]
        if "$filter" in qp:
            params["filter"] = qp["$filter"]
        if "$top" in qp:
            params["top"] = int(qp["$top"])
        if "$skip" in qp:
            params["skip"] = int(qp["$skip"])
        return cls(**params)
