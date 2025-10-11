"""Timestamp helpers with strict UTC enforcement for analytics and logging.

Key Responsibilities:
    - Provide a canonical way to fetch timezone-aware UTC timestamps
    - Validate and normalise user-provided datetimes to UTC

Side Effects:
    - None; functions operate on provided datetime values

Thread Safety:
    - Thread-safe; uses stdlib datetime utilities
"""

from __future__ import annotations

from datetime import UTC, datetime

# ==============================================================================
# PUBLIC HELPERS
# ==============================================================================


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Ensure that ``value`` is timezone aware and converted to UTC.

    Args:
        value: Datetime instance to normalise.

    Returns:
        Datetime converted to UTC.

    Raises:
        ValueError: If ``value`` is naive and lacks timezone information.
    """
    if value.tzinfo is None:
        raise ValueError("Datetime must include timezone information")
    return value.astimezone(UTC)
