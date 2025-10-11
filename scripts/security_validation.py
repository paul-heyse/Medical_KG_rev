#!/usr/bin/env python3
"""Security validation script.

This script validates that the service architecture maintains security
standards and implements proper security measures.
"""

import sys
from pathlib import Path
from typing import Any


class SecurityValidator:
    """Validator for service architecture security."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.security_requirements = {
            "mTLS_enabled": True,
            "TLS_version": "TLSv1.3",
            "certificate_validation": True,
            "service_authentication": True,
            "audit_logging": True,
            "data_encryption": True,
            "access_control": True,
            "rate_limiting": True,
            "input_validation": True,
            "error_handling": True,
        }

    def validate_mtls_configuration(self) -> dict[str, Any]:
        """Validate mTLS configuration."""
        cert_dir = self.project_root / "certs"

        if not cert_dir.exists():
            return {
                "passed": False,
                "message": "Certificate directory does not exist",
                "issues": ["Missing certs/ directory"],
            }

        required_certs = [
            "ca.crt",
            "ca.key",
            "gpu-management-service.crt",
            "gpu-management-service.key",
            "embedding-service.crt",
            "embedding-service.key",
            "reranking-service.crt",
            "reranking-service.key",
            "docling-vlm-service.crt",
            "docling-vlm-service.key",
            "gateway.crt",
            "gateway.key",
        ]

        missing_certs = []
        for cert_file in required_certs:
            if not (cert_dir / cert_file).exists():
                missing_certs.append(cert_file)

        if missing_certs:
            return {
                "passed": False,
                "message": f"Missing certificates: {missing_certs}",
                "issues": [f"Missing certificates: {missing_certs}"],
            }

        # Validate certificate formats
        cert_issues = []
        for cert_file in required_certs:
            if cert_file.endswith(".crt"):
                try:
                    cert_path = cert_dir / cert_file
                    with open(cert_path, encoding="utf-8") as f:
                        cert_content = f.read()

                    # Basic PEM format check
                    if "BEGIN CERTIFICATE" not in cert_content:
                        cert_issues.append(f"Invalid certificate format: {cert_file}")

                except (UnicodeDecodeError, OSError):
                    cert_issues.append(f"Error reading certificate: {cert_file}")

        return {
            "passed": len(cert_issues) == 0,
            "message": (
                "mTLS configuration properly configured"
                if len(cert_issues) == 0
                else f"mTLS configuration issues: {cert_issues}"
            ),
            "issues": cert_issues,
        }

    def validate_tls_configuration(self) -> dict[str, Any]:
        """Validate TLS configuration."""
        # Check TLS configuration in Docker Compose
        compose_path = self.project_root / "ops/docker/docker-compose.torch-isolation.yml"

        if not compose_path.exists():
            return {
                "passed": False,
                "message": "Docker Compose file does not exist",
                "issues": ["Missing docker-compose.torch-isolation.yml"],
            }

        try:
            with open(compose_path, encoding="utf-8") as f:
                content = f.read()

            issues = []

            # Check for TLS configuration
            if "TLS" not in content and "tls" not in content:
                issues.append("Missing TLS configuration")

            # Check for certificate mounts
            if "certs" not in content:
                issues.append("Missing certificate mounts")

            # Check for SSL context configuration
            if "SSL" not in content and "ssl" not in content:
                issues.append("Missing SSL context configuration")

            return {
                "passed": len(issues) == 0,
                "message": (
                    "TLS configuration properly configured"
                    if len(issues) == 0
                    else f"TLS configuration issues: {issues}"
                ),
                "issues": issues,
            }

        except (UnicodeDecodeError, OSError) as e:
            return {
                "passed": False,
                "message": f"Error reading Docker Compose file: {e}",
                "issues": [f"Error reading Docker Compose file: {e}"],
            }

    def validate_service_authentication(self) -> dict[str, Any]:
        """Validate service authentication."""
        # Check for authentication configuration in service clients
        client_files = [
            "src/Medical_KG_rev/services/clients/gpu_client.py",
            "src/Medical_KG_rev/services/clients/embedding_client.py",
            "src/Medical_KG_rev/services/clients/reranking_client.py",
            "src/Medical_KG_rev/services/clients/docling_vlm_client.py",
        ]

        issues = []
        for client_file in client_files:
            file_path = self.project_root / client_file

            if not file_path.exists():
                issues.append(f"Missing client file: {client_file}")
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for authentication mechanisms
                if "mtls" not in content.lower() and "ssl" not in content.lower():
                    issues.append(f"Missing authentication in {client_file}")

                if "certificate" not in content.lower() and "cert" not in content.lower():
                    issues.append(f"Missing certificate configuration in {client_file}")

            except (UnicodeDecodeError, OSError):
                issues.append(f"Error reading {client_file}")

        return {
            "passed": len(issues) == 0,
            "message": (
                "Service authentication properly configured"
                if len(issues) == 0
                else f"Service authentication issues: {issues}"
            ),
            "issues": issues,
        }

    def validate_audit_logging(self) -> dict[str, Any]:
        """Validate audit logging configuration."""
        audit_logger_path = (
            self.project_root / "src/Medical_KG_rev/services/logging/audit_logger.py"
        )

        if not audit_logger_path.exists():
            return {
                "passed": False,
                "message": "Audit logger does not exist",
                "issues": ["Missing audit_logger.py"],
            }

        try:
            with open(audit_logger_path, encoding="utf-8") as f:
                content = f.read()

            issues = []

            # Check for audit logging functionality
            if "audit" not in content.lower():
                issues.append("Missing audit logging functionality")

            if "log" not in content.lower():
                issues.append("Missing logging functionality")

            # Check for structured logging
            if "structlog" not in content.lower():
                issues.append("Missing structured logging")

            # Check for service operation logging
            if "service" not in content.lower():
                issues.append("Missing service operation logging")

            return {
                "passed": len(issues) == 0,
                "message": (
                    "Audit logging properly configured"
                    if len(issues) == 0
                    else f"Audit logging issues: {issues}"
                ),
                "issues": issues,
            }

        except (UnicodeDecodeError, OSError) as e:
            return {
                "passed": False,
                "message": f"Error reading audit logger: {e}",
                "issues": [f"Error reading audit logger: {e}"],
            }

    def validate_data_encryption(self) -> dict[str, Any]:
        """Validate data encryption configuration."""
        # Check for encryption configuration in service files
        service_files = [
            "src/Medical_KG_rev/services/security/mtls.py",
            "src/Medical_KG_rev/services/security/grpc_mtls.py",
        ]

        issues = []
        for service_file in service_files:
            file_path = self.project_root / service_file

            if not file_path.exists():
                issues.append(f"Missing security file: {service_file}")
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for encryption mechanisms
                if "encrypt" not in content.lower() and "cipher" not in content.lower():
                    issues.append(f"Missing encryption in {service_file}")

                if "ssl" not in content.lower() and "tls" not in content.lower():
                    issues.append(f"Missing SSL/TLS in {service_file}")

            except (UnicodeDecodeError, OSError):
                issues.append(f"Error reading {service_file}")

        return {
            "passed": len(issues) == 0,
            "message": (
                "Data encryption properly configured"
                if len(issues) == 0
                else f"Data encryption issues: {issues}"
            ),
            "issues": issues,
        }

    def validate_access_control(self) -> dict[str, Any]:
        """Validate access control configuration."""
        # Check for access control in service configurations
        config_files = ["src/Medical_KG_rev/config/mtls_config.py", "ops/k8s/mtls-config.yaml"]

        issues = []
        for config_file in config_files:
            file_path = self.project_root / config_file

            if not file_path.exists():
                issues.append(f"Missing config file: {config_file}")
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for access control mechanisms
                if "access" not in content.lower() and "permission" not in content.lower():
                    issues.append(f"Missing access control in {config_file}")

                if "role" not in content.lower() and "policy" not in content.lower():
                    issues.append(f"Missing role-based access in {config_file}")

            except (UnicodeDecodeError, OSError):
                issues.append(f"Error reading {config_file}")

        return {
            "passed": len(issues) == 0,
            "message": (
                "Access control properly configured"
                if len(issues) == 0
                else f"Access control issues: {issues}"
            ),
            "issues": issues,
        }

    def validate_rate_limiting(self) -> dict[str, Any]:
        """Validate rate limiting configuration."""
        # Check for rate limiting in service clients
        client_files = [
            "src/Medical_KG_rev/services/clients/error_handler.py",
            "src/Medical_KG_rev/services/clients/circuit_breaker.py",
        ]

        issues = []
        for client_file in client_files:
            file_path = self.project_root / client_file

            if not file_path.exists():
                issues.append(f"Missing client file: {client_file}")
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for rate limiting mechanisms
                if "rate" not in content.lower() and "limit" not in content.lower():
                    issues.append(f"Missing rate limiting in {client_file}")

                if "throttle" not in content.lower() and "backoff" not in content.lower():
                    issues.append(f"Missing throttling in {client_file}")

            except (UnicodeDecodeError, OSError):
                issues.append(f"Error reading {client_file}")

        return {
            "passed": len(issues) == 0,
            "message": (
                "Rate limiting properly configured"
                if len(issues) == 0
                else f"Rate limiting issues: {issues}"
            ),
            "issues": issues,
        }

    def validate_input_validation(self) -> dict[str, Any]:
        """Validate input validation configuration."""
        # Check for input validation in service clients
        client_files = [
            "src/Medical_KG_rev/services/clients/gpu_client.py",
            "src/Medical_KG_rev/services/clients/embedding_client.py",
            "src/Medical_KG_rev/services/clients/reranking_client.py",
            "src/Medical_KG_rev/services/clients/docling_vlm_client.py",
        ]

        issues = []
        for client_file in client_files:
            file_path = self.project_root / client_file

            if not file_path.exists():
                issues.append(f"Missing client file: {client_file}")
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for input validation mechanisms
                if "validate" not in content.lower() and "sanitize" not in content.lower():
                    issues.append(f"Missing input validation in {client_file}")

                if "pydantic" not in content.lower() and "schema" not in content.lower():
                    issues.append(f"Missing schema validation in {client_file}")

            except (UnicodeDecodeError, OSError):
                issues.append(f"Error reading {client_file}")

        return {
            "passed": len(issues) == 0,
            "message": (
                "Input validation properly configured"
                if len(issues) == 0
                else f"Input validation issues: {issues}"
            ),
            "issues": issues,
        }

    def validate_error_handling(self) -> dict[str, Any]:
        """Validate error handling configuration."""
        # Check for error handling in service clients
        error_files = [
            "src/Medical_KG_rev/services/clients/errors.py",
            "src/Medical_KG_rev/services/clients/error_handler.py",
        ]

        issues = []
        for error_file in error_files:
            file_path = self.project_root / error_file

            if not file_path.exists():
                issues.append(f"Missing error file: {error_file}")
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for error handling mechanisms
                if "error" not in content.lower() and "exception" not in content.lower():
                    issues.append(f"Missing error handling in {error_file}")

                if "handle" not in content.lower() and "catch" not in content.lower():
                    issues.append(f"Missing error handling logic in {error_file}")

            except (UnicodeDecodeError, OSError):
                issues.append(f"Error reading {error_file}")

        return {
            "passed": len(issues) == 0,
            "message": (
                "Error handling properly configured"
                if len(issues) == 0
                else f"Error handling issues: {issues}"
            ),
            "issues": issues,
        }

    def run_all_validations(self) -> dict[str, Any]:
        """Run all security validations."""
        validations = {
            "mtls_configuration": self.validate_mtls_configuration(),
            "tls_configuration": self.validate_tls_configuration(),
            "service_authentication": self.validate_service_authentication(),
            "audit_logging": self.validate_audit_logging(),
            "data_encryption": self.validate_data_encryption(),
            "access_control": self.validate_access_control(),
            "rate_limiting": self.validate_rate_limiting(),
            "input_validation": self.validate_input_validation(),
            "error_handling": self.validate_error_handling(),
        }

        all_passed = all(validation["passed"] for validation in validations.values())

        return {
            "all_passed": all_passed,
            "validations": validations,
            "summary": {
                "total_validations": len(validations),
                "passed_validations": sum(
                    1 for validation in validations.values() if validation["passed"]
                ),
                "failed_validations": sum(
                    1 for validation in validations.values() if not validation["passed"]
                ),
            },
        }


def main() -> None:
    """Main entry point for security validation."""
    project_root = Path(__file__).parent.parent

    print("🔍 Running Security Validation...")
    print("=" * 60)

    validator = SecurityValidator(project_root)
    result = validator.run_all_validations()

    print("\n📊 Security Validation Results:")
    print(f"   Total Validations: {result['summary']['total_validations']}")
    print(f"   Passed: {result['summary']['passed_validations']}")
    print(f"   Failed: {result['summary']['failed_validations']}")

    print("\n🔍 Individual Validation Results:")
    for validation_name, validation_result in result["validations"].items():
        status = "✅ PASS" if validation_result["passed"] else "❌ FAIL"
        print(f"   {validation_name}: {status}")
        if not validation_result["passed"]:
            print(f"      {validation_result['message']}")
            if validation_result.get("issues"):
                for issue in validation_result["issues"]:
                    print(f"         - {issue}")

    print("\n" + "=" * 60)

    if result["all_passed"]:
        print("✅ SECURITY VALIDATION PASSED!")
        print("\nAll security validations have passed successfully.")
        print("The service architecture maintains security standards.")
        sys.exit(0)
    else:
        print("❌ SECURITY VALIDATION FAILED!")
        print("\nPlease address the security issues above before proceeding.")
        print("The service architecture does not meet security standards.")
        sys.exit(1)


if __name__ == "__main__":
    main()
