"""Chunking command abstractions shared across protocols."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping
from uuid import uuid4

from Medical_KG_rev.chunking.exceptions import InvalidDocumentError


def _coerce_mapping(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not payload:
        return MappingProxyType({})
    if isinstance(payload, MappingProxyType):
        return payload
    sanitized: MutableMapping[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        sanitized[key] = value
    return MappingProxyType(dict(sanitized))


@dataclass(slots=True, frozen=True)
class ChunkCommand:
    """Rich command object describing a chunking request."""

    tenant_id: str
    document_id: str
    text: str
    strategy: str = "semantic"
    chunk_size: int | None = None
    overlap: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    profile: str | None = None
    correlation_id: str = field(default_factory=lambda: uuid4().hex)
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # noqa: D401 - dataclass validation
        """Validate base command invariants."""

        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise InvalidDocumentError("Chunking requires a tenant identifier")
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise InvalidDocumentError("Chunking requires a document identifier")
        if not isinstance(self.text, str) or not self.text.strip():
            raise InvalidDocumentError(
                "Chunking requests must include a non-empty 'text' field in options"
            )
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise InvalidDocumentError("chunk_size must be a positive integer when provided")
        if self.overlap is not None and not (0.0 <= self.overlap < 1.0):
            raise InvalidDocumentError("overlap must be between 0.0 and 1.0")
        object.__setattr__(self, "metadata", _coerce_mapping(self.metadata))
        object.__setattr__(self, "context", _coerce_mapping(self.context))
        normalized_profile = self.profile
        if not isinstance(normalized_profile, str) or not normalized_profile.strip():
            normalized_profile = None
        object.__setattr__(self, "profile", normalized_profile)

    @property
    def issued_at_iso(self) -> str:
        return self.issued_at.isoformat().replace("+00:00", "Z")

    def asdict(self, *, include_text: bool = False) -> dict[str, Any]:
        payload = {
            "tenant_id": self.tenant_id,
            "document_id": self.document_id,
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "metadata": dict(self.metadata),
            "profile": self.profile,
            "correlation_id": self.correlation_id,
            "issued_at": self.issued_at_iso,
            "context": dict(self.context),
        }
        if include_text:
            payload["text"] = self.text
        return payload

    def log_context(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "document_id": self.document_id,
            "strategy": self.strategy,
            "profile": self.profile or "default",
            "correlation_id": self.correlation_id,
        }

    def metric_tags(self) -> dict[str, str]:
        return {
            "strategy": self.strategy,
            "profile": (self.profile or "default"),
        }

    def with_context(self, **context: Any) -> "ChunkCommand":
        payload = dict(self.context)
        payload.update(context)
        return replace(self, context=MappingProxyType(payload))

    @classmethod
    def from_request(
        cls,
        request: "ChunkRequestProtocol",
        *,
        text: str,
        metadata: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> "ChunkCommand":
        profile_hint = None
        source_metadata = metadata or {}
        if isinstance(source_metadata, Mapping):
            raw_profile = source_metadata.get("profile")
            if isinstance(raw_profile, str) and raw_profile.strip():
                profile_hint = raw_profile.strip()
        return cls(
            tenant_id=request.tenant_id,
            document_id=request.document_id,
            text=text,
            strategy=request.strategy,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            metadata=source_metadata,
            profile=profile_hint,
            correlation_id=correlation_id or uuid4().hex,
            context=context or {},
        )


class ChunkRequestProtocol:
    """Structural subset of request attributes required to build commands."""

    tenant_id: str
    document_id: str
    strategy: str
    chunk_size: int | None
    overlap: float


__all__ = ["ChunkCommand", "ChunkRequestProtocol"]
