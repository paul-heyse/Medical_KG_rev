"""mTLS configuration management for service-to-service authentication.

This module provides configuration classes and utilities for managing
mutual TLS certificates and authentication settings.
"""

from pathlib import Path
import os

from pydantic import BaseModel, Field, validator



class mTLSConfig(BaseModel):
    """Configuration for mTLS authentication."""

    enabled: bool = Field(default=False, description="Enable mTLS authentication")
    ca_cert_path: str = Field(default="certs/ca.crt", description="Path to CA certificate")
    ca_key_path: str = Field(default="certs/ca.key", description="Path to CA private key")
    cert_duration_days: int = Field(
        default=365, description="Certificate validity duration in days"
    )
    key_size: int = Field(default=2048, description="RSA key size in bits")
    auto_generate_certs: bool = Field(
        default=True, description="Auto-generate certificates if missing"
    )

    @validator("ca_cert_path", "ca_key_path")
    def validate_cert_paths(cls, v):
        """Validate certificate paths."""
        if not v:
            raise ValueError("Certificate path cannot be empty")
        return v

    @validator("cert_duration_days")
    def validate_cert_duration(cls, v):
        """Validate certificate duration."""
        if v < 1 or v > 3650:  # 1 day to 10 years
            raise ValueError("Certificate duration must be between 1 and 3650 days")
        return v

    @validator("key_size")
    def validate_key_size(cls, v):
        """Validate RSA key size."""
        if v not in [1024, 2048, 3072, 4096]:
            raise ValueError("Key size must be 1024, 2048, 3072, or 4096 bits")
        return v


class ServiceCertificateConfig(BaseModel):
    """Configuration for individual service certificates."""

    service_name: str = Field(..., description="Name of the service")
    common_name: str = Field(..., description="Certificate common name")
    san_dns_names: list[str] = Field(
        default_factory=list, description="DNS subject alternative names"
    )
    san_ip_addresses: list[str] = Field(
        default_factory=list, description="IP subject alternative names"
    )
    cert_path: str = Field(..., description="Path to certificate file")
    key_path: str = Field(..., description="Path to private key file")

    @validator("service_name", "common_name")
    def validate_names(cls, v):
        """Validate service and common names."""
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


class mTLSManagerConfig(BaseModel):
    """Configuration for mTLS manager."""

    mtls: mTLSConfig = Field(default_factory=mTLSConfig, description="mTLS configuration")
    services: dict[str, ServiceCertificateConfig] = Field(
        default_factory=dict, description="Service certificate configurations"
    )

    @classmethod
    def from_env(cls) -> "mTLSManagerConfig":
        """Create configuration from environment variables."""
        mtls_config = mTLSConfig(
            enabled=os.getenv("MTLS_ENABLED", "false").lower() == "true",
            ca_cert_path=os.getenv("MTLS_CA_CERT_PATH", "certs/ca.crt"),
            ca_key_path=os.getenv("MTLS_CA_KEY_PATH", "certs/ca.key"),
            cert_duration_days=int(os.getenv("MTLS_CERT_DURATION_DAYS", "365")),
            key_size=int(os.getenv("MTLS_KEY_SIZE", "2048")),
            auto_generate_certs=os.getenv("MTLS_AUTO_GENERATE_CERTS", "true").lower() == "true",
        )

        # Default service configurations
        services = {
            "gateway": ServiceCertificateConfig(
                service_name="gateway",
                common_name="gateway.medical-kg.local",
                san_dns_names=["gateway.medical-kg.local", "localhost"],
                san_ip_addresses=["127.0.0.1"],
                cert_path="certs/gateway.crt",
                key_path="certs/gateway.key",
            ),
            "gpu-management": ServiceCertificateConfig(
                service_name="gpu-management",
                common_name="gpu-management.medical-kg.local",
                san_dns_names=["gpu-management.medical-kg.local", "localhost"],
                san_ip_addresses=["127.0.0.1"],
                cert_path="certs/gpu-management.crt",
                key_path="certs/gpu-management.key",
            ),
            "embedding": ServiceCertificateConfig(
                service_name="embedding",
                common_name="embedding.medical-kg.local",
                san_dns_names=["embedding.medical-kg.local", "localhost"],
                san_ip_addresses=["127.0.0.1"],
                cert_path="certs/embedding.crt",
                key_path="certs/embedding.key",
            ),
            "reranking": ServiceCertificateConfig(
                service_name="reranking",
                common_name="reranking.medical-kg.local",
                san_dns_names=["reranking.medical-kg.local", "localhost"],
                san_ip_addresses=["127.0.0.1"],
                cert_path="certs/reranking.crt",
                key_path="certs/reranking.key",
            ),
            "docling-vlm": ServiceCertificateConfig(
                service_name="docling-vlm",
                common_name="docling-vlm.medical-kg.local",
                san_dns_names=["docling-vlm.medical-kg.local", "localhost"],
                san_ip_addresses=["127.0.0.1"],
                cert_path="certs/docling-vlm.crt",
                key_path="certs/docling-vlm.key",
            ),
        }

        return cls(mtls=mtls_config, services=services)

    def get_service_config(self, service_name: str) -> ServiceCertificateConfig | None:
        """Get configuration for a specific service."""
        return self.services.get(service_name)

    def list_services(self) -> list[str]:
        """List all configured services."""
        return list(self.services.keys())

    def add_service(self, service_config: ServiceCertificateConfig) -> None:
        """Add a new service configuration."""
        self.services[service_config.service_name] = service_config

    def remove_service(self, service_name: str) -> bool:
        """Remove a service configuration."""
        if service_name in self.services:
            del self.services[service_name]
            return True
        return False


def create_default_mtls_config() -> mTLSManagerConfig:
    """Create default mTLS configuration."""
    return mTLSManagerConfig.from_env()


def validate_certificate_files(config: mTLSManagerConfig) -> dict[str, bool]:
    """Validate that certificate files exist."""
    results = {}

    # Check CA certificate files
    results["ca_cert"] = Path(config.mtls.ca_cert_path).exists()
    results["ca_key"] = Path(config.mtls.ca_key_path).exists()

    # Check service certificate files
    for service_name, service_config in config.services.items():
        results[f"{service_name}_cert"] = Path(service_config.cert_path).exists()
        results[f"{service_name}_key"] = Path(service_config.key_path).exists()

    return results


def ensure_certificate_directories(config: mTLSManagerConfig) -> None:
    """Ensure certificate directories exist."""
    # Ensure CA certificate directory
    Path(config.mtls.ca_cert_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.mtls.ca_key_path).parent.mkdir(parents=True, exist_ok=True)

    # Ensure service certificate directories
    for service_config in config.services.values():
        Path(service_config.cert_path).parent.mkdir(parents=True, exist_ok=True)
        Path(service_config.key_path).parent.mkdir(parents=True, exist_ok=True)


def get_service_endpoint_with_mtls(
    service_name: str, host: str = "localhost", port: int = 50051
) -> str:
    """Get service endpoint with mTLS configuration."""
    # For mTLS, we use the service name as the target name override
    return f"{host}:{port}"


def get_mtls_options(service_name: str) -> list[tuple]:
    """Get mTLS options for gRPC channel/server."""
    return [
        ("grpc.ssl_target_name_override", service_name),
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 5000),
        ("grpc.keepalive_permit_without_calls", True),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 10000),
        ("grpc.http2.min_ping_interval_without_data_ms", 300000),
    ]
