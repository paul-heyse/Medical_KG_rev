"""Tests covering the high-level application settings schema metadata."""

from Medical_KG_rev.config.settings import ENVIRONMENT_DEFAULTS, Environment


def test_environment_defaults_provide_expected_overrides() -> None:
    """Environment presets should include telemetry and security overrides."""
    staging_defaults = ENVIRONMENT_DEFAULTS[Environment.STAGING]
    assert staging_defaults["telemetry"]["exporter"] == "otlp"
    assert staging_defaults["redis_cache"]["use_tls"] is True

    prod_defaults = ENVIRONMENT_DEFAULTS[Environment.PROD]
    assert prod_defaults["telemetry"]["sample_ratio"] == 0.05
    assert prod_defaults["object_storage"]["use_tls"] is True


def test_environment_enum_covers_supported_values() -> None:
    """`Environment` enum should expose the supported deployment tiers."""
    assert {env.value for env in Environment} == {"dev", "staging", "prod"}
