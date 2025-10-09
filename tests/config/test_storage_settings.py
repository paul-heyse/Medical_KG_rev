"""Unit tests for storage configuration settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from Medical_KG_rev.config.settings import ObjectStorageSettings, RedisCacheSettings


class TestObjectStorageSettings:
    """Test object storage configuration settings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = ObjectStorageSettings()

        assert settings.bucket == "medical-kg-pdf"
        assert settings.region == "us-east-1"
        assert settings.endpoint_url is None
        assert settings.access_key_id is None
        assert settings.secret_access_key is None
        assert settings.session_token is None
        assert settings.use_tls is True
        assert settings.tls_cert_path is None
        assert settings.max_file_size == 100 * 1024 * 1024  # 100MB
        assert settings.key_prefix == "pdf"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        settings = ObjectStorageSettings(
            bucket="custom-bucket",
            region="eu-west-1",
            endpoint_url="http://minio:9000",
            access_key_id="minioadmin",
            secret_access_key="minioadmin",
            use_tls=False,
            max_file_size=50 * 1024 * 1024,  # 50MB
            key_prefix="custom-prefix",
        )

        assert settings.bucket == "custom-bucket"
        assert settings.region == "eu-west-1"
        assert settings.endpoint_url == "http://minio:9000"
        assert settings.access_key_id == "minioadmin"
        assert settings.secret_access_key == "minioadmin"
        assert settings.use_tls is False
        assert settings.max_file_size == 50 * 1024 * 1024
        assert settings.key_prefix == "custom-prefix"

    def test_secret_str_fields(self) -> None:
        """Test that secret fields are properly handled as SecretStr."""
        settings = ObjectStorageSettings(
            secret_access_key="secret-key",
            session_token="session-token",
        )

        assert settings.secret_access_key.get_secret_value() == "secret-key"
        assert settings.session_token.get_secret_value() == "session-token"

    def test_validation_errors(self) -> None:
        """Test validation errors for invalid values."""
        # Test negative max_file_size
        with pytest.raises(ValidationError):
            ObjectStorageSettings(max_file_size=-1)

        # Test empty bucket name
        with pytest.raises(ValidationError):
            ObjectStorageSettings(bucket="")

    def test_field_descriptions(self) -> None:
        """Test that fields have proper descriptions."""
        field_info = ObjectStorageSettings.model_fields

        assert "S3 bucket name for PDF storage" in field_info["bucket"].description
        assert "AWS region for S3 operations" in field_info["region"].description
        assert "Custom S3 endpoint (e.g., MinIO)" in field_info["endpoint_url"].description
        assert "Maximum file size in bytes (100MB)" in field_info["max_file_size"].description


class TestRedisCacheSettings:
    """Test Redis cache configuration settings."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = RedisCacheSettings()

        assert settings.url == "redis://redis:6379/0"
        assert settings.password is None
        assert settings.use_tls is False
        assert settings.tls_cert_path is None
        assert settings.db_index == 0
        assert settings.key_prefix == "medical-kg"
        assert settings.default_ttl == 3600
        assert settings.max_connections == 10

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        settings = RedisCacheSettings(
            url="redis://localhost:6379/1",
            password="redis-password",
            use_tls=True,
            tls_cert_path="/path/to/cert.pem",
            db_index=5,
            key_prefix="custom-prefix",
            default_ttl=7200,
            max_connections=20,
        )

        assert settings.url == "redis://localhost:6379/1"
        assert settings.password.get_secret_value() == "redis-password"
        assert settings.use_tls is True
        assert settings.tls_cert_path == "/path/to/cert.pem"
        assert settings.db_index == 5
        assert settings.key_prefix == "custom-prefix"
        assert settings.default_ttl == 7200
        assert settings.max_connections == 20

    def test_secret_str_password(self) -> None:
        """Test that password field is properly handled as SecretStr."""
        settings = RedisCacheSettings(password="secret-password")

        assert settings.password.get_secret_value() == "secret-password"

    def test_validation_errors(self) -> None:
        """Test validation errors for invalid values."""
        # Test invalid db_index (out of range)
        with pytest.raises(ValidationError):
            RedisCacheSettings(db_index=16)  # Max is 15

        with pytest.raises(ValidationError):
            RedisCacheSettings(db_index=-1)  # Min is 0

        # Test negative default_ttl
        with pytest.raises(ValidationError):
            RedisCacheSettings(default_ttl=-1)

        # Test invalid max_connections
        with pytest.raises(ValidationError):
            RedisCacheSettings(max_connections=0)

    def test_field_descriptions(self) -> None:
        """Test that fields have proper descriptions."""
        field_info = RedisCacheSettings.model_fields

        assert "Redis connection URL" in field_info["url"].description
        assert "Redis password" in field_info["password"].description
        assert "Redis database index" in field_info["db_index"].description
        assert "Key prefix for cache entries" in field_info["key_prefix"].description
        assert "Default TTL in seconds" in field_info["default_ttl"].description
        assert "Maximum connection pool size" in field_info["max_connections"].description


class TestStorageSettingsIntegration:
    """Test storage settings integration with environment defaults."""

    def test_environment_defaults_dev(self) -> None:
        """Test development environment defaults."""
        from Medical_KG_rev.config.settings import ENVIRONMENT_DEFAULTS, Environment

        dev_defaults = ENVIRONMENT_DEFAULTS[Environment.DEV]

        # Check object storage defaults
        assert "object_storage" in dev_defaults
        obj_storage_defaults = dev_defaults["object_storage"]
        assert obj_storage_defaults["endpoint_url"] == "http://minio:9000"
        assert obj_storage_defaults["bucket"] == "medical-kg-pdf"
        assert obj_storage_defaults["use_tls"] is False

        # Check Redis cache defaults
        assert "redis_cache" in dev_defaults
        redis_defaults = dev_defaults["redis_cache"]
        assert redis_defaults["url"] == "redis://redis:6379/0"
        assert redis_defaults["use_tls"] is False

    def test_environment_defaults_staging(self) -> None:
        """Test staging environment defaults."""
        from Medical_KG_rev.config.settings import ENVIRONMENT_DEFAULTS, Environment

        staging_defaults = ENVIRONMENT_DEFAULTS[Environment.STAGING]

        # Check object storage defaults
        assert "object_storage" in staging_defaults
        obj_storage_defaults = staging_defaults["object_storage"]
        assert obj_storage_defaults["use_tls"] is True

        # Check Redis cache defaults
        assert "redis_cache" in staging_defaults
        redis_defaults = staging_defaults["redis_cache"]
        assert redis_defaults["use_tls"] is True

    def test_environment_defaults_prod(self) -> None:
        """Test production environment defaults."""
        from Medical_KG_rev.config.settings import ENVIRONMENT_DEFAULTS, Environment

        prod_defaults = ENVIRONMENT_DEFAULTS[Environment.PROD]

        # Check object storage defaults
        assert "object_storage" in prod_defaults
        obj_storage_defaults = prod_defaults["object_storage"]
        assert obj_storage_defaults["use_tls"] is True

        # Check Redis cache defaults
        assert "redis_cache" in prod_defaults
        redis_defaults = prod_defaults["redis_cache"]
        assert redis_defaults["use_tls"] is True
