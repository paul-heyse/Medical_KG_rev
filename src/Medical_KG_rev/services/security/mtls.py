"""Mutual TLS (mTLS) implementation for service-to-service authentication.

This module provides mTLS certificate management, validation, and gRPC
interceptor configuration for secure service-to-service communication.
"""

import logging
import ssl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import grpc
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CertificateConfig(BaseModel):
    """Configuration for mTLS certificates."""

    ca_cert_path: str = Field(..., description="Path to CA certificate")
    ca_key_path: str = Field(..., description="Path to CA private key")
    cert_path: str = Field(..., description="Path to service certificate")
    key_path: str = Field(..., description="Path to service private key")
    cert_duration_days: int = Field(
        default=365, description="Certificate validity duration in days"
    )
    key_size: int = Field(default=2048, description="RSA key size in bits")


class ServiceCertificate(BaseModel):
    """Service certificate information."""

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


class mTLSManager:
    """Manages mTLS certificates and SSL contexts for service-to-service authentication."""

    def __init__(self, config: CertificateConfig):
        """Initialize mTLS manager with configuration."""
        self.config = config
        self._ca_cert: x509.Certificate | None = None
        self._ca_key: rsa.RSAPrivateKey | None = None
        self._certificates: dict[str, ServiceCertificate] = {}

    async def initialize(self) -> None:
        """Initialize mTLS manager by loading CA certificate and key."""
        try:
            # Load CA certificate and key
            await self._load_ca_certificate()
            logger.info("mTLS manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize mTLS manager: {e}")
            raise

    async def _load_ca_certificate(self) -> None:
        """Load CA certificate and private key."""
        try:
            # Load CA certificate
            with open(self.config.ca_cert_path, "rb") as f:
                ca_cert_data = f.read()
            self._ca_cert = x509.load_pem_x509_certificate(ca_cert_data)

            # Load CA private key
            with open(self.config.ca_key_path, "rb") as f:
                ca_key_data = f.read()
            self._ca_key = serialization.load_pem_private_key(ca_key_data, password=None)

            logger.info(f"Loaded CA certificate: {self._ca_cert.subject}")
        except Exception as e:
            logger.error(f"Failed to load CA certificate: {e}")
            raise

    async def generate_service_certificate(
        self,
        service_name: str,
        common_name: str,
        san_dns_names: list[str] | None = None,
        san_ip_addresses: list[str] | None = None,
    ) -> ServiceCertificate:
        """Generate a new service certificate signed by the CA."""
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=self.config.key_size
            )

            # Create certificate builder
            builder = x509.CertificateBuilder()

            # Set certificate subject
            subject = x509.Name(
                [
                    x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Medical_KG_rev"),
                    x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "GPU Services"),
                ]
            )
            builder = builder.subject_name(subject)

            # Set issuer (CA)
            builder = builder.issuer_name(self._ca_cert.subject)

            # Set validity period
            now = datetime.utcnow()
            builder = builder.not_valid_before(now)
            builder = builder.not_valid_after(now + timedelta(days=self.config.cert_duration_days))

            # Set serial number
            builder = builder.serial_number(x509.random_serial_number())

            # Set public key
            builder = builder.public_key(private_key.public_key())

            # Add subject alternative names
            san_list = []
            if san_dns_names:
                san_list.extend([x509.DNSName(name) for name in san_dns_names])
            if san_ip_addresses:
                from ipaddress import ip_address

                san_list.extend([x509.IPAddress(ip_address(ip)) for ip in san_ip_addresses])

            if san_list:
                builder = builder.add_extension(
                    x509.SubjectAlternativeName(san_list), critical=False
                )

            # Add key usage extensions
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )

            # Add extended key usage
            builder = builder.add_extension(
                x509.ExtendedKeyUsage(
                    [
                        x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                        x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    ]
                ),
                critical=True,
            )

            # Sign certificate with CA
            if self._ca_key is None:
                raise ValueError("CA key not loaded")
            certificate = builder.sign(private_key=self._ca_key, algorithm=hashes.SHA256())

            # Create certificate paths
            cert_path = f"certs/{service_name}.crt"
            key_path = f"certs/{service_name}.key"

            # Ensure certs directory exists
            Path("certs").mkdir(exist_ok=True)

            # Write certificate
            Path(cert_path).write_bytes(certificate.public_bytes(serialization.Encoding.PEM))

            # Write private key
            Path(key_path).write_bytes(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

            # Create service certificate info
            service_cert = ServiceCertificate(
                service_name=service_name,
                common_name=common_name,
                san_dns_names=san_dns_names or [],
                san_ip_addresses=san_ip_addresses or [],
                cert_path=cert_path,
                key_path=key_path,
            )

            self._certificates[service_name] = service_cert

            logger.info(f"Generated certificate for service: {service_name}")
            return service_cert

        except Exception as e:
            logger.error(f"Failed to generate certificate for {service_name}: {e}")
            raise

    def create_server_ssl_context(self, service_name: str) -> ssl.SSLContext:
        """Create SSL context for gRPC server with mTLS."""
        try:
            if service_name not in self._certificates:
                raise ValueError(f"No certificate found for service: {service_name}")

            service_cert = self._certificates[service_name]

            # Create SSL context
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(service_cert.cert_path, service_cert.key_path)
            context.load_verify_locations(self.config.ca_cert_path)
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = False  # We verify via certificate CN/SAN

            logger.info(f"Created server SSL context for service: {service_name}")
            return context

        except Exception as e:
            logger.error(f"Failed to create server SSL context for {service_name}: {e}")
            raise

    def create_client_ssl_context(self, service_name: str) -> ssl.SSLContext:
        """Create SSL context for gRPC client with mTLS."""
        try:
            if service_name not in self._certificates:
                raise ValueError(f"No certificate found for service: {service_name}")

            service_cert = self._certificates[service_name]

            # Create SSL context
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            context.load_cert_chain(service_cert.cert_path, service_cert.key_path)
            context.load_verify_locations(self.config.ca_cert_path)
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = False  # We verify via certificate CN/SAN

            logger.info(f"Created client SSL context for service: {service_name}")
            return context

        except Exception as e:
            logger.error(f"Failed to create client SSL context for {service_name}: {e}")
            raise

    def create_grpc_server_options(self, service_name: str) -> list[tuple[str, Any]]:
        """Create gRPC server options for mTLS."""
        try:
            ssl_context = self.create_server_ssl_context(service_name)

            # Convert SSL context to gRPC options
            options = [
                ("grpc.ssl_target_name_override", service_name),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.min_ping_interval_without_data_ms", 300000),
            ]

            logger.info(f"Created gRPC server options for service: {service_name}")
            return options

        except Exception as e:
            logger.error(f"Failed to create gRPC server options for {service_name}: {e}")
            raise

    def create_grpc_client_options(self, service_name: str) -> list[tuple[str, Any]]:
        """Create gRPC client options for mTLS."""
        try:
            ssl_context = self.create_client_ssl_context(service_name)

            # Convert SSL context to gRPC options
            options = [
                ("grpc.ssl_target_name_override", service_name),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.min_ping_interval_without_data_ms", 300000),
            ]

            logger.info(f"Created gRPC client options for service: {service_name}")
            return options

        except Exception as e:
            logger.error(f"Failed to create gRPC client options for {service_name}: {e}")
            raise

    async def validate_certificate(self, cert_path: str) -> bool:
        """Validate a certificate against the CA."""
        try:
            cert_data = Path(cert_path).read_bytes()

            certificate = x509.load_pem_x509_certificate(cert_data)

            # Check if certificate is signed by our CA
            try:
                if self._ca_cert is None:
                    return False
                # Verify certificate signature
                self._ca_cert.public_key().verify(
                    certificate.signature,
                    certificate.tbs_certificate_bytes,
                    certificate.signature_algorithm_oid,
                )
            except Exception:
                return False

            # Check validity period
            now = datetime.utcnow()
            if now < certificate.not_valid_before or now > certificate.not_valid_after:
                return False

            logger.info(f"Certificate validation successful: {cert_path}")
            return True

        except Exception as e:
            logger.error(f"Certificate validation failed for {cert_path}: {e}")
            return False

    async def rotate_certificate(self, service_name: str) -> ServiceCertificate:
        """Rotate certificate for a service."""
        try:
            if service_name not in self._certificates:
                raise ValueError(f"No certificate found for service: {service_name}")

            old_cert = self._certificates[service_name]

            # Generate new certificate
            new_cert = await self.generate_service_certificate(
                service_name=service_name,
                common_name=old_cert.common_name,
                san_dns_names=old_cert.san_dns_names,
                san_ip_addresses=old_cert.san_ip_addresses,
            )

            logger.info(f"Rotated certificate for service: {service_name}")
            return new_cert

        except Exception as e:
            logger.error(f"Failed to rotate certificate for {service_name}: {e}")
            raise

    def get_certificate_info(self, service_name: str) -> Optional[ServiceCertificate]:
        """Get certificate information for a service."""
        return self._certificates.get(service_name)

    def list_certificates(self) -> list[str]:
        """List all managed certificates."""
        return list(self._certificates.keys())


class mTLSInterceptor(grpc.aio.ServerInterceptor):
    """gRPC server interceptor for mTLS authentication."""

    def __init__(self, mtls_manager: mTLSManager, service_name: str):
        """Initialize mTLS interceptor."""
        self.mtls_manager = mtls_manager
        self.service_name = service_name

    async def intercept_service(
        self, continuation: Any, handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """Intercept gRPC service calls for mTLS authentication."""
        try:
            # Extract peer certificate information
            context = handler_call_details.invocation_metadata
            peer_cert = None

            for key, value in context:
                if key == "peer_certificate":
                    peer_cert = value
                    break

            if not peer_cert:
                logger.warning(f"Missing peer certificate in request to {self.service_name}")
                return grpc.aio.Abort(grpc.StatusCode.UNAUTHENTICATED, "Missing certificate")

            # Validate peer certificate
            is_valid = await self.mtls_manager.validate_certificate(peer_cert)
            if not is_valid:
                logger.warning(f"Invalid peer certificate in request to {self.service_name}")
                return grpc.aio.Abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid certificate")

            # Continue with the request
            return await continuation(handler_call_details)

        except Exception as e:
            logger.error(f"mTLS authentication error in {self.service_name}: {e}")
            return grpc.aio.Abort(grpc.StatusCode.INTERNAL, "Authentication error")


def create_mtls_channel(
    target: str, mtls_manager: mTLSManager, service_name: str
) -> grpc.aio.Channel:
    """Create a gRPC channel with mTLS authentication."""
    try:
        ssl_context = mtls_manager.create_client_ssl_context(service_name)
        options = mtls_manager.create_grpc_client_options(service_name)

        # Create secure channel
        channel = grpc.aio.secure_channel(target, ssl_context, options)

        logger.info(f"Created mTLS channel to {target} for service {service_name}")
        return channel

    except Exception as e:
        logger.error(f"Failed to create mTLS channel to {target}: {e}")
        raise


def create_mtls_server(mtls_manager: mTLSManager, service_name: str) -> grpc.aio.Server:
    """Create a gRPC server with mTLS authentication."""
    try:
        ssl_context = mtls_manager.create_server_ssl_context(service_name)
        options = mtls_manager.create_grpc_server_options(service_name)

        # Create secure server
        server = grpc.aio.server(options)

        # Add mTLS interceptor
        mtls_interceptor = mTLSInterceptor(mtls_manager, service_name)
        server = grpc.aio.intercept(server, mtls_interceptor)

        logger.info(f"Created mTLS server for service {service_name}")
        return server

    except Exception as e:
        logger.error(f"Failed to create mTLS server for {service_name}: {e}")
        raise
