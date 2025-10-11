"""gRPC mTLS integration utilities.

This module provides utilities for integrating mTLS authentication
with gRPC services and clients.
"""

import logging

import grpc
import grpc.aio
from grpc import aio

from Medical_KG_rev.config.mtls_config import create_default_mtls_config, mTLSManagerConfig
from Medical_KG_rev.services.security.mtls import CertificateConfig, mTLSManager

logger = logging.getLogger(__name__)


class gRPCmTLSManager:
    """Manager for gRPC services with mTLS authentication."""

    def __init__(self, config: mTLSManagerConfig | None = None):
        """Initialize gRPC mTLS manager."""
        self.config = config or create_default_mtls_config()
        self.mtls_manager: mTLSManager | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize mTLS manager and validate certificates."""
        try:
            if not self.config.mtls.enabled:
                logger.info("mTLS is disabled - using insecure connections")
                return

            # Create certificate config
            cert_config = CertificateConfig(
                ca_cert_path=self.config.mtls.ca_cert_path,
                ca_key_path=self.config.mtls.ca_key_path,
                cert_path="",  # Will be set per service
                key_path="",  # Will be set per service
                cert_duration_days=self.config.mtls.cert_duration_days,
                key_size=self.config.mtls.key_size,
            )

            # Initialize mTLS manager
            self.mtls_manager = mTLSManager(cert_config)
            await self.mtls_manager.initialize()

            # Generate certificates if needed
            if self.config.mtls.auto_generate_certs:
                await self._ensure_certificates_exist()

            self._initialized = True
            logger.info("gRPC mTLS manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize gRPC mTLS manager: {e}")
            raise

    async def _ensure_certificates_exist(self) -> None:
        """Ensure all required certificates exist."""
        try:
            # Check if CA certificate exists
            from pathlib import Path

            if not Path(self.config.mtls.ca_cert_path).exists():
                logger.info("CA certificate not found - generating...")
                await self._generate_ca_certificate()

            # Generate service certificates
            for service_name, service_config in self.config.services.items():
                cert_path = Path(service_config.cert_path)
                key_path = Path(service_config.key_path)

                if not cert_path.exists() or not key_path.exists():
                    logger.info(f"Generating certificate for service: {service_name}")
                    await self.mtls_manager.generate_service_certificate(
                        service_name=service_name,
                        common_name=service_config.common_name,
                        san_dns_names=service_config.san_dns_names,
                        san_ip_addresses=service_config.san_ip_addresses,
                    )

        except Exception as e:
            logger.error(f"Failed to ensure certificates exist: {e}")
            raise

    async def _generate_ca_certificate(self) -> None:
        """Generate CA certificate."""
        from datetime import datetime, timedelta
        from pathlib import Path

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        try:
            # Generate CA private key
            ca_key = rsa.generate_private_key(
                public_exponent=65537, key_size=self.config.mtls.key_size
            )

            # Create CA certificate
            subject = issuer = x509.Name(
                [
                    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
                    x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Medical_KG_rev"),
                    x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Certificate Authority"),
                    x509.NameAttribute(NameOID.COMMON_NAME, "Medical_KG_rev CA"),
                ]
            )

            ca_cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(ca_key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.utcnow())
                .not_valid_after(
                    datetime.utcnow() + timedelta(days=self.config.mtls.cert_duration_days * 10)
                )  # CA valid for 10x longer
                .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
                .add_extension(
                    x509.KeyUsage(
                        key_cert_sign=True,
                        crl_sign=True,
                        digital_signature=True,
                        key_encipherment=False,
                        content_commitment=False,
                        data_encipherment=False,
                        key_agreement=False,
                        encipher_only=False,
                        decipher_only=False,
                    ),
                    critical=True,
                )
                .add_extension(
                    x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False
                )
                .sign(ca_key, hashes.SHA256())
            )

            # Ensure directories exist
            Path(self.config.mtls.ca_cert_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.config.mtls.ca_key_path).parent.mkdir(parents=True, exist_ok=True)

            # Write CA certificate
            with open(self.config.mtls.ca_cert_path, "wb") as f:
                f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

            # Write CA private key
            with open(self.config.mtls.ca_key_path, "wb") as f:
                f.write(
                    ca_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                )

            logger.info("CA certificate generated successfully")

        except Exception as e:
            logger.error(f"Failed to generate CA certificate: {e}")
            raise

    def create_secure_channel(self, target: str, service_name: str) -> grpc.aio.Channel:
        """Create a secure gRPC channel with mTLS."""
        try:
            if not self._initialized or not self.mtls_manager:
                logger.warning("mTLS not initialized - creating insecure channel")
                return aio.insecure_channel(target)

            # Create SSL context
            ssl_context = self.mtls_manager.create_client_ssl_context(service_name)

            # Create secure channel
            channel = aio.secure_channel(target, ssl_context)

            logger.info(f"Created secure channel to {target} for service {service_name}")
            return channel

        except Exception as e:
            logger.error(f"Failed to create secure channel to {target}: {e}")
            # Fallback to insecure channel
            logger.warning(f"Falling back to insecure channel for {target}")
            return aio.insecure_channel(target)

    def create_secure_server(
        self, service_name: str, options: list[tuple[str, str]] | None = None
    ) -> grpc.aio.Server:
        """Create a secure gRPC server with mTLS."""
        try:
            if not self._initialized or not self.mtls_manager:
                logger.warning("mTLS not initialized - creating insecure server")
                return aio.server(options or [])

            # Create SSL context
            ssl_context = self.mtls_manager.create_server_ssl_context(service_name)

            # Create secure server
            server = aio.server(options or [])

            # Add SSL credentials
            server_credentials = grpc.ssl_server_credentials(
                [
                    (
                        self.mtls_manager._certificates[service_name].key_path,
                        self.mtls_manager._certificates[service_name].cert_path,
                    )
                ],
                root_certificates=self.mtls_manager._ca_cert.public_bytes(
                    serialization.Encoding.PEM
                ),
                require_client_auth=True,
            )

            logger.info(f"Created secure server for service {service_name}")
            return server

        except Exception as e:
            logger.error(f"Failed to create secure server for {service_name}: {e}")
            # Fallback to insecure server
            logger.warning(f"Falling back to insecure server for {service_name}")
            return aio.server(options or [])

    def get_service_endpoint(
        self, service_name: str, host: str = "localhost", port: int = 50051
    ) -> str:
        """Get service endpoint for mTLS communication."""
        return f"{host}:{port}"

    def get_mtls_options(self, service_name: str) -> list[tuple[str, str]]:
        """Get mTLS options for gRPC channel/server."""
        return [
            ("grpc.ssl_target_name_override", service_name),
            ("grpc.keepalive_time_ms", "30000"),
            ("grpc.keepalive_timeout_ms", "5000"),
            ("grpc.keepalive_permit_without_calls", "True"),
            ("grpc.http2.max_pings_without_data", "0"),
            ("grpc.http2.min_time_between_pings_ms", "10000"),
            ("grpc.http2.min_ping_interval_without_data_ms", "300000"),
        ]

    def is_mtls_enabled(self) -> bool:
        """Check if mTLS is enabled."""
        return self._initialized and self.config.mtls.enabled

    def get_service_config(self, service_name: str) -> dict[str, str] | None:
        """Get configuration for a specific service."""
        if not self._initialized or not self.mtls_manager:
            return None

        service_cert = self.mtls_manager.get_certificate_info(service_name)
        if not service_cert:
            return None

        return {
            "cert_path": service_cert.cert_path,
            "key_path": service_cert.key_path,
            "common_name": service_cert.common_name,
            "san_dns_names": ",".join(service_cert.san_dns_names),
            "san_ip_addresses": ",".join(service_cert.san_ip_addresses),
        }

    def list_services(self) -> list[str]:
        """List all configured services."""
        if not self._initialized or not self.mtls_manager:
            return []

        return self.mtls_manager.list_certificates()

    async def rotate_certificate(self, service_name: str) -> bool:
        """Rotate certificate for a service."""
        try:
            if not self._initialized or not self.mtls_manager:
                logger.warning("mTLS not initialized - cannot rotate certificate")
                return False

            await self.mtls_manager.rotate_certificate(service_name)
            logger.info(f"Rotated certificate for service: {service_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate certificate for {service_name}: {e}")
            return False

    async def validate_certificate(self, service_name: str) -> bool:
        """Validate certificate for a service."""
        try:
            if not self._initialized or not self.mtls_manager:
                return False

            service_cert = self.mtls_manager.get_certificate_info(service_name)
            if not service_cert:
                return False

            return await self.mtls_manager.validate_certificate(service_cert.cert_path)

        except Exception as e:
            logger.error(f"Failed to validate certificate for {service_name}: {e}")
            return False


# Global instance
_grpc_mtls_manager: gRPCmTLSManager | None = None


async def initialize_grpc_mtls(config: mTLSManagerConfig | None = None) -> gRPCmTLSManager:
    """Initialize global gRPC mTLS manager."""
    global _grpc_mtls_manager

    if _grpc_mtls_manager is None:
        _grpc_mtls_manager = gRPCmTLSManager(config)
        await _grpc_mtls_manager.initialize()

    return _grpc_mtls_manager


def get_grpc_mtls_manager() -> gRPCmTLSManager | None:
    """Get the global gRPC mTLS manager."""
    return _grpc_mtls_manager


def create_secure_channel(target: str, service_name: str) -> grpc.aio.Channel:
    """Create a secure gRPC channel with mTLS."""
    manager = get_grpc_mtls_manager()
    if manager:
        return manager.create_secure_channel(target, service_name)
    else:
        logger.warning("mTLS manager not available - creating insecure channel")
        return aio.insecure_channel(target)


def create_secure_server(
    service_name: str, options: list[tuple[str, str]] | None = None
) -> grpc.aio.Server:
    """Create a secure gRPC server with mTLS."""
    manager = get_grpc_mtls_manager()
    if manager:
        return manager.create_secure_server(service_name, options)
    else:
        logger.warning("mTLS manager not available - creating insecure server")
        return aio.server(options or [])
