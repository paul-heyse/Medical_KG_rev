"""API key management utilities.

Utilities for managing API keys used by tenant integrations.

This module provides the canonical API key management flow for the platform.
It is responsible for generating secure secrets, validating provided keys, and
bootstrapping in-memory key stores from configuration or secret stores.

Key Responsibilities:
    - Generate cryptographically random API keys for tenants.
    - Authenticate incoming requests using hashed secrets.
    - Load key material from static configuration and secret stores.
    - Provide rotation helpers that preserve key identifiers.

Collaborators:
    - Upstream: FastAPI dependencies defined in ``dependencies.py``.
    - Downstream: ``SecurityContext`` construction and request auditing.
    - External: ``SecretResolver`` for retrieving secrets from Vault.

Side Effects:
    - None; state is stored in-memory for process lifetime only.

Thread Safety:
    - ``APIKeyManager`` is not thread-safe for concurrent writers. Each worker
      process should manage its own instance or guard mutations with locks.

Performance Characteristics:
    - Key generation and hashing are O(1) operations dominated by hashing cost.
    - Loading from configuration scales with number of configured keys.

Example:
    >>> manager = APIKeyManager()
    >>> key = manager.generate(tenant_id="demo", scopes=["ingest:write"])
    >>> manager.authenticate(key.raw_secret)
    ('key_...', APIKeyRecord(...))

"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================
import hashlib
import json
import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from ..config.settings import APIKeyRecord, AppSettings, SecretResolver, get_settings

# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class APIKey:
    """Runtime representation of an issued API key.

    Attributes:
        key_id: Stable identifier for the key used for rotation and auditing.
        raw_secret: Plaintext secret returned to the caller exactly once.
        hashed_secret: Hash of the secret stored in configuration or Vault.
        tenant_id: Tenant identifier that owns the key.
        scopes: Iterable of scopes granted to the key.
        rotated_at: ISO-8601 timestamp string for the most recent rotation.

    """

    key_id: str
    raw_secret: str
    hashed_secret: str
    tenant_id: str
    scopes: Iterable[str]
    rotated_at: str | None = None


# ============================================================================
# MANAGER IMPLEMENTATION
# ============================================================================


class APIKeyManager:
    """Manage API key lifecycle for tenants.

    The manager stores key metadata in-memory and exposes helpers to generate,
    rotate, and authenticate keys. Generated secrets are hashed using a
    configurable algorithm before being persisted in ``APIKeyRecord`` models.

    Attributes:
        hashing_algorithm: Name of ``hashlib`` hashing algorithm used for
            storing secrets.

    Invariants:
        - ``self._records`` contains only ``APIKeyRecord`` instances.
        - Keys are addressed by ``key_id`` and must be unique.

    """

    def __init__(self, *, hashing_algorithm: str = "sha256") -> None:
        """Initialize the manager with the desired hashing algorithm.

        Args:
            hashing_algorithm: Name of the ``hashlib`` algorithm used to hash
                API secrets.

        Raises:
            ValueError: If the provided algorithm is not supported by
                :mod:`hashlib` when hashing is attempted.

        """
        self.hashing_algorithm = hashing_algorithm
        self._records: dict[str, APIKeyRecord] = {}

    def load(self, records: dict[str, APIKeyRecord]) -> None:
        """Load API key records from configuration or external storage.

        Args:
            records: Mapping of key identifier to ``APIKeyRecord`` metadata.

        """
        self._records = dict(records)

    def generate(
        self, *, key_id: str | None = None, tenant_id: str, scopes: Iterable[str]
    ) -> APIKey:
        """Create a new API key for the provided tenant.

        Args:
            key_id: Optional identifier. A random identifier is created when not
                provided.
            tenant_id: Tenant identifier that owns the generated key.
            scopes: Iterable of scopes granted to the key.

        Returns:
            ``APIKey`` containing the identifier, plaintext secret, and hashed
            secret. The caller must persist the plaintext secret.

        """
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
        return APIKey(
            key_id=key_id,
            raw_secret=raw_secret,
            hashed_secret=hashed,
            tenant_id=tenant_id,
            scopes=scopes,
        )

    def rotate(self, key_id: str) -> APIKey:
        """Rotate an existing key while preserving its identifier.

        Args:
            key_id: Identifier of the key to rotate.

        Returns:
            Newly generated ``APIKey`` representing the rotated secret.

        Raises:
            KeyError: If the key identifier is unknown.

        """
        if key_id not in self._records:
            raise KeyError(f"Unknown API key {key_id}")
        record = self._records[key_id]
        rotated = self.generate(key_id=key_id, tenant_id=record.tenant_id, scopes=record.scopes)
        return rotated

    def authenticate(self, provided_key: str) -> tuple[str, APIKeyRecord]:
        """Validate a caller-provided key and return the canonical record.

        Args:
            provided_key: Plaintext secret presented by the caller.

        Returns:
            Tuple containing the key identifier and matching record.

        Raises:
            PermissionError: If the provided secret does not match any record.

        """
        hashed = self._hash(provided_key)
        for key_id, record in self._records.items():
            if record.hashed_secret == hashed:
                return key_id, record
        raise PermissionError("Invalid API key")

    def _hash(self, value: str) -> str:
        """Hash the provided secret using the configured algorithm.

        Args:
            value: Plaintext value to hash.

        Returns:
            Hexadecimal digest of the hashed secret.

        Raises:
            ValueError: If the configured algorithm is unsupported.

        """
        algorithm = getattr(hashlib, self.hashing_algorithm, None)
        if not algorithm:
            raise ValueError(f"Unsupported hashing algorithm {self.hashing_algorithm}")
        return algorithm(value.encode("utf-8")).hexdigest()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def build_api_key_manager(settings: AppSettings | None = None) -> APIKeyManager:
    """Construct an ``APIKeyManager`` initialized from application settings.

    Args:
        settings: Optional settings override. Defaults to global settings.

    Returns:
        A fully populated ``APIKeyManager`` ready for authentication checks.

    Notes:
        When the API key feature is disabled the returned manager will be empty
        but still functional for generating ad-hoc keys (useful in tests).

    """
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
            keys_payload = (
                secret_payload.get("keys", {}) if isinstance(secret_payload, dict) else {}
            )
            for key_id, payload in keys_payload.items():
                if isinstance(payload, dict) and "hashed_secret" in payload:
                    records[key_id] = APIKeyRecord.model_validate(payload)
        manager.load(records)
    return manager


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["APIKey", "APIKeyManager", "build_api_key_manager"]
