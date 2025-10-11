"""Tests for time utilities."""

from datetime import UTC, datetime, timedelta, timezone

from Medical_KG_rev.utils.time import ensure_utc, utc_now


def test_utc_now_returns_timezone_aware() -> None:
    """`utc_now` should include timezone information in UTC."""
    now = utc_now()
    assert isinstance(now, datetime)
    assert now.tzinfo is UTC


def test_ensure_utc_converts_to_utc() -> None:
    """`ensure_utc` converts aware datetimes to UTC."""
    paris = timezone(timedelta(hours=2))
    dt = datetime(2024, 6, 1, 12, 0, tzinfo=paris)
    converted = ensure_utc(dt)
    assert converted.tzinfo is UTC
    assert converted.hour == 10  # two hours behind Paris


def test_ensure_utc_rejects_naive_datetime() -> None:
    """Naive datetimes should raise a ValueError."""
    naive = datetime(2024, 6, 1, 12, 0)
    try:
        ensure_utc(naive)
    except ValueError as exc:
        assert "timezone" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for naive datetime")
