#!/usr/bin/env python3
"""Service security validation script.

This script validates the security implementation of gRPC services
including mTLS authentication, certificate validation, and service
communication security.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from cryptography import x509

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Medical_KG_rev.config.mtls_config import create_default_mtls_config, mTLSManagerConfig
from Medical_KG_rev.services.security.mtls import CertificateConfig, mTLSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityValidator:
    """Validates security implementation of gRPC services."""

    def __init__(self, config: mTLSManagerConfig):
        """Initialize security validator."""
        self.config = config
        self.mtls_manager: mTLSManager | None = None
        self.validation_results: dict[str, bool] = {}

    async def initialize(self) -> None:
        """Initialize the security validator."""
        try:
            if not self.config.mtls.enabled:
                logger.warning("mTLS is disabled - security validation limited")
                return

            # Create certificate config
            cert_config = CertificateConfig(
                ca_cert_path=self.config.mtls.ca_cert_path,
                ca_key_path=self.config.mtls.ca_key_path,
                cert_path="",
                key_path="",
                cert_duration_days=self.config.mtls.cert_duration_days,
                key_size=self.config.mtls.key_size,
            )

            # Initialize mTLS manager
            self.mtls_manager = mTLSManager(cert_config)
            await self.mtls_manager.initialize()

            logger.info("Security validator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize security validator: {e}")
            raise

    async def validate_certificates(self) -> bool:
        """Validate all service certificates."""
        try:
            logger.info("Validating service certificates...")

            if not self.mtls_manager:
                logger.error("mTLS manager not initialized")
                return False

            # Validate CA certificate
            ca_valid = await self.mtls_manager.validate_certificate(self.config.mtls.ca_cert_path)
            self.validation_results["ca_certificate"] = ca_valid

            if not ca_valid:
                logger.error("CA certificate validation failed")
                return False

            # Validate service certificates
            all_valid = True
            for service_name in self.config.services.keys():
                service_config = self.config.services[service_name]
                cert_valid = await self.mtls_manager.validate_certificate(service_config.cert_path)
                self.validation_results[f"{service_name}_certificate"] = cert_valid

                if not cert_valid:
                    logger.error(f"Certificate validation failed for service: {service_name}")
                    all_valid = False
                else:
                    logger.info(f"Certificate validation passed for service: {service_name}")

            logger.info("Certificate validation completed")
            return all_valid

        except Exception as e:
            logger.error(f"Certificate validation failed: {e}")
            return False

    async def validate_ssl_contexts(self) -> bool:
        """Validate SSL contexts for all services."""
        try:
            logger.info("Validating SSL contexts...")

            if not self.mtls_manager:
                logger.error("mTLS manager not initialized")
                return False

            all_valid = True
            for service_name in self.config.services.keys():
                try:
                    # Test server SSL context
                    self.mtls_manager.create_server_ssl_context(service_name)
                    self.validation_results[f"{service_name}_server_ssl"] = True

                    # Test client SSL context
                    self.mtls_manager.create_client_ssl_context(service_name)
                    self.validation_results[f"{service_name}_client_ssl"] = True

                    logger.info(f"SSL context validation passed for service: {service_name}")

                except Exception as e:
                    logger.error(f"SSL context validation failed for service {service_name}: {e}")
                    self.validation_results[f"{service_name}_server_ssl"] = False
                    self.validation_results[f"{service_name}_client_ssl"] = False
                    all_valid = False

            logger.info("SSL context validation completed")
            return all_valid

        except Exception as e:
            logger.error(f"SSL context validation failed: {e}")
            return False

    async def validate_certificate_security(self) -> bool:
        """Validate certificate security properties."""
        try:
            logger.info("Validating certificate security properties...")

            if not self.mtls_manager:
                logger.error("mTLS manager not initialized")
                return False

            all_valid = True

            # Validate CA certificate security
            ca_cert = self.mtls_manager._ca_cert
            if ca_cert:
                # Check CA certificate properties
                if ca_cert.not_valid_after < ca_cert.not_valid_before:
                    logger.error("CA certificate has invalid validity period")
                    all_valid = False

                # Check key usage
                try:
                    key_usage = ca_cert.extensions.get_extension_for_oid(
                        x509.oid.ExtensionOID.KEY_USAGE
                    )
                    if not key_usage.value.key_cert_sign:
                        logger.error("CA certificate missing keyCertSign usage")
                        all_valid = False
                except x509.ExtensionNotFound:
                    logger.error("CA certificate missing key usage extension")
                    all_valid = False

            # Validate service certificates
            for service_name in self.config.services.keys():
                service_config = self.config.services[service_name]
                try:
                    cert_data = Path(service_config.cert_path).read_bytes()
                    certificate = x509.load_pem_x509_certificate(cert_data)

                    # Check certificate validity period
                    if certificate.not_valid_after < certificate.not_valid_before:
                        logger.error(
                            f"Service certificate {service_name} has invalid validity period"
                        )
                        all_valid = False

                    # Check extended key usage
                    try:
                        ext_key_usage = certificate.extensions.get_extension_for_oid(
                            x509.oid.ExtensionOID.EXTENDED_KEY_USAGE
                        )
                        if (
                            x509.oid.ExtendedKeyUsageOID.SERVER_AUTH not in ext_key_usage.value
                            or x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH not in ext_key_usage.value
                        ):
                            logger.error(
                                f"Service certificate {service_name} missing required key usage"
                            )
                            all_valid = False
                    except x509.ExtensionNotFound:
                        logger.error(
                            f"Service certificate {service_name} missing extended key usage"
                        )
                        all_valid = False

                    # Check subject alternative names
                    try:
                        san = certificate.extensions.get_extension_for_oid(
                            x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
                        )
                        if not san.value:
                            logger.error(
                                f"Service certificate {service_name} missing subject alternative names"
                            )
                            all_valid = False
                    except x509.ExtensionNotFound:
                        logger.error(
                            f"Service certificate {service_name} missing subject alternative names"
                        )
                        all_valid = False

                    logger.info(
                        f"Certificate security validation passed for service: {service_name}"
                    )

                except Exception as e:
                    logger.error(
                        f"Certificate security validation failed for service {service_name}: {e}"
                    )
                    all_valid = False

            logger.info("Certificate security validation completed")
            return all_valid

        except Exception as e:
            logger.error(f"Certificate security validation failed: {e}")
            return False

    async def validate_network_security(self) -> bool:
        """Validate network security configuration."""
        try:
            logger.info("Validating network security configuration...")

            # Check if certificates are properly configured
            cert_files_exist = True
            for service_name in self.config.services.keys():
                service_config = self.config.services[service_name]
                if not Path(service_config.cert_path).exists():
                    logger.error(f"Certificate file not found: {service_config.cert_path}")
                    cert_files_exist = False

                if not Path(service_config.key_path).exists():
                    logger.error(f"Key file not found: {service_config.key_path}")
                    cert_files_exist = False

            self.validation_results["network_security"] = cert_files_exist

            if not cert_files_exist:
                logger.error("Network security validation failed - missing certificate files")
                return False

            logger.info("Network security validation completed")
            return True

        except Exception as e:
            logger.error(f"Network security validation failed: {e}")
            return False

    async def validate_service_communication(self) -> bool:
        """Validate service communication security."""
        try:
            logger.info("Validating service communication security...")

            if not self.mtls_manager:
                logger.error("mTLS manager not initialized")
                return False

            all_valid = True

            # Test gRPC server options
            for service_name in self.config.services.keys():
                try:
                    server_options = self.mtls_manager.create_grpc_server_options(service_name)
                    client_options = self.mtls_manager.create_grpc_client_options(service_name)

                    # Validate options format
                    if not isinstance(server_options, list):
                        logger.error(f"Invalid server options format for service: {service_name}")
                        all_valid = False

                    if not isinstance(client_options, list):
                        logger.error(f"Invalid client options format for service: {service_name}")
                        all_valid = False

                    logger.info(
                        f"Service communication validation passed for service: {service_name}"
                    )

                except Exception as e:
                    logger.error(
                        f"Service communication validation failed for service {service_name}: {e}"
                    )
                    all_valid = False

            self.validation_results["service_communication"] = all_valid

            logger.info("Service communication validation completed")
            return all_valid

        except Exception as e:
            logger.error(f"Service communication validation failed: {e}")
            return False

    async def run_full_validation(self) -> bool:
        """Run full security validation."""
        try:
            logger.info("Starting full security validation...")

            # Initialize validator
            await self.initialize()

            # Run all validation checks
            validations = [
                ("Certificate Validation", self.validate_certificates()),
                ("SSL Context Validation", self.validate_ssl_contexts()),
                ("Certificate Security Validation", self.validate_certificate_security()),
                ("Network Security Validation", self.validate_network_security()),
                ("Service Communication Validation", self.validate_service_communication()),
            ]

            all_passed = True
            for name, validation_coro in validations:
                try:
                    result = await validation_coro
                    if not result:
                        all_passed = False
                        logger.error(f"Validation failed: {name}")
                    else:
                        logger.info(f"Validation passed: {name}")
                except Exception as e:
                    logger.error(f"Validation error in {name}: {e}")
                    all_passed = False

            # Print summary
            self.print_validation_summary()

            logger.info("Full security validation completed")
            return all_passed

        except Exception as e:
            logger.error(f"Full security validation failed: {e}")
            return False

    def print_validation_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("SECURITY VALIDATION SUMMARY")
        print("=" * 60)

        total_checks = len(self.validation_results)
        passed_checks = sum(1 for result in self.validation_results.values() if result)
        failed_checks = total_checks - passed_checks

        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {failed_checks}")
        print(f"Success Rate: {(passed_checks/total_checks)*100:.1f}%")

        print("\nDetailed Results:")
        print("-" * 40)
        for check_name, result in self.validation_results.items():
            status = "PASS" if result else "FAIL"
            print(f"{check_name:<30} {status}")

        print("=" * 60)

        if failed_checks > 0:
            print("❌ SECURITY VALIDATION FAILED")
            print("Some security checks failed. Please review the errors above.")
        else:
            print("✅ SECURITY VALIDATION PASSED")
            print("All security checks passed successfully.")


async def main() -> None:
    """Main function to run security validation."""
    parser = argparse.ArgumentParser(description="Validate service security")
    parser.add_argument("--config", help="Path to mTLS configuration file")
    parser.add_argument("--ca-cert", default="certs/ca.crt", help="Path to CA certificate")
    parser.add_argument("--ca-key", default="certs/ca.key", help="Path to CA private key")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        config = create_default_mtls_config()

        # Override paths if provided
        if args.ca_cert:
            config.mtls.ca_cert_path = args.ca_cert
        if args.ca_key:
            config.mtls.ca_key_path = args.ca_key

        # Create validator
        validator = SecurityValidator(config)

        # Run validation
        success = await validator.run_full_validation()

        if success:
            logger.info("Security validation completed successfully")
            sys.exit(0)
        else:
            logger.error("Security validation failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Security validation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
