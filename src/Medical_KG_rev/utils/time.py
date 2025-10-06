"""Timestamp helpers with strict UTC enforcement."""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return current UTC time with tzinfo."""

    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    """Ensure datetime is timezone aware and converted to UTC."""

    if value.tzinfo is None:
        raise ValueError("Datetime must include timezone information")
    return value.astimezone(timezone.utc)
