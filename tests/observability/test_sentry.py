from __future__ import annotations

from Medical_KG_rev.config.settings import AppSettings
from Medical_KG_rev.observability import sentry as sentry_module


def test_initialise_sentry_invokes_sdk(monkeypatch) -> None:
    sentry_module._SENTRY_INITIALISED = False
    calls: list[dict] = []

    class DummySDK:
        @staticmethod
        def init(**kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)

    monkeypatch.setattr(sentry_module, "sentry_sdk", DummySDK)

    settings = AppSettings()
    settings.observability.sentry.dsn = "https://public@example.ingest.sentry.io/1"

    sentry_module.initialise_sentry(settings)

    assert calls
    assert calls[0]["dsn"] == "https://public@example.ingest.sentry.io/1"
    assert calls[0]["environment"] == settings.environment.value
