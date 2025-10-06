"""API key management utilities."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Optional

from ..config.settings import APIKeyRecord, AppSettings, SecretResolver, get_settings


@dataclass
class APIKey:
    key_id: str
    raw_secret: str
    hashed_secret: str
    tenant_id: str
    scopes: Iterable[str]
    rotated_at: Optional[str] = None


class APIKeyManager:
    """In-memory API key store backed by configuration or Vault."""

    def __init__(self, *, hashing_algorithm: str = "sha256") -> None:
        self.hashing_algorithm = hashing_algorithm
        self._records: Dict[str, APIKeyRecord] = {}

    def load(self, records: Dict[str, APIKeyRecord]) -> None:
        self._records = dict(records)

    def generate(self, *, key_id: Optional[str] = None, tenant_id: str, scopes: Iterable[str]) -> APIKey:
        key_id = key_id or f"key_{secrets.token_urlsafe(8)}"
        raw_secret = secrets.token_urlsafe(32)
        hashed = self._hash(raw_secret)
        record = APIKeyRecord(
            hashed_secret=hashed,
            tenant_id=tenant_id,
            scopes=list(scopes),
            rotated_at=datetime.utcnow().isoformat(),
        )
        self._records[key_id] = record
        return APIKey(key_id=key_id, raw_secret=raw_secret, hashed_secret=hashed, tenant_id=tenant_id, scopes=scopes)

    def rotate(self, key_id: str) -> APIKey:
        if key_id not in self._records:
            raise KeyError(f"Unknown API key {key_id}")
        record = self._records[key_id]
        rotated = self.generate(key_id=key_id, tenant_id=record.tenant_id, scopes=record.scopes)
        return rotated

    def authenticate(self, provided_key: str) -> tuple[str, APIKeyRecord]:
        hashed = self._hash(provided_key)
        for key_id, record in self._records.items():
            if record.hashed_secret == hashed:
                return key_id, record
        raise PermissionError("Invalid API key")

    def _hash(self, value: str) -> str:
        algorithm = getattr(hashlib, self.hashing_algorithm, None)
        if not algorithm:
            raise ValueError(f"Unsupported hashing algorithm {self.hashing_algorithm}")
        return algorithm(value.encode("utf-8")).hexdigest()


def build_api_key_manager(settings: Optional[AppSettings] = None) -> APIKeyManager:
    settings = settings or get_settings()
    cfg = settings.security.api_keys
    manager = APIKeyManager(hashing_algorithm=cfg.hashing_algorithm)
    if cfg.enabled:
        records = {}
        for key_id, record in cfg.keys.items():
            scopes = []
            for scope in record.scopes:
                if isinstance(scope, str) and scope.startswith("["):
                    try:
                        scopes.extend(json.loads(scope))
                        continue
                    except json.JSONDecodeError:
                        pass
                scopes.append(scope)
            records[key_id] = APIKeyRecord(
                hashed_secret=record.hashed_secret,
                tenant_id=record.tenant_id,
                scopes=scopes,
                rotated_at=record.rotated_at,
            )
        if cfg.secret_store_path:
            resolver = SecretResolver(settings)
            try:
                secret_payload = resolver.get_secret(cfg.secret_store_path)
            except KeyError:
                secret_payload = {}
            keys_payload = secret_payload.get("keys", {}) if isinstance(secret_payload, dict) else {}
            for key_id, payload in keys_payload.items():
                if isinstance(payload, dict) and "hashed_secret" in payload:
                    records[key_id] = APIKeyRecord.model_validate(payload)
        manager.load(records)
    return manager
