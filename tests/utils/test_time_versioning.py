from datetime import UTC, datetime

import pytest

from Medical_KG_rev.utils.time import ensure_utc, utc_now
from Medical_KG_rev.utils.versioning import Version


def test_utc_now_returns_timezone_aware():
    now = utc_now()
    assert now.tzinfo == UTC


def test_ensure_utc_requires_timezone():
    aware = ensure_utc(datetime.now(UTC))
    assert aware.tzinfo == UTC
    with pytest.raises(ValueError):
        ensure_utc(datetime.now())


def test_version_helpers():
    version = Version.parse("v1.2")
    assert str(version.bump_patch()) == "v1.2.1"
    assert str(version.bump_minor()) == "v1.3.0"
