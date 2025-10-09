"""Serialisation helpers for pipeline state payloads."""

from __future__ import annotations

import base64
import zlib
from typing import Any

import orjson

from .models import PipelineStateModel


def serialise_payload(payload: dict[str, Any]) -> PipelineStateModel:
    """Validate and normalise a PipelineState payload."""
    return PipelineStateModel.model_validate(payload)


def dumps_orjson(payload: dict[str, Any]) -> bytes:
    model = serialise_payload(payload)
    return orjson.dumps(model.model_dump(mode="json"))


def dumps_json(payload: dict[str, Any]) -> str:
    return dumps_orjson(payload).decode("utf-8")


def encode_base64(blob: bytes) -> str:
    compressed = zlib.compress(blob)
    return base64.b64encode(compressed).decode("ascii")


__all__ = [
    "dumps_json",
    "dumps_orjson",
    "encode_base64",
    "serialise_payload",
]
