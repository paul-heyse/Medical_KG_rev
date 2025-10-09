# Secret Management Guide

This document provides comprehensive guidance on secret management, security practices, and best practices for handling sensitive information in the Medical_KG_rev system.

## Overview

Secret management is critical for the security of the Medical_KG_rev system. This guide covers how to securely store, access, rotate, and audit secrets across different environments.

## Secret Categories

### Authentication Secrets

#### JWT Secrets

- **Purpose**: Signing and verifying JWT tokens
- **Sensitivity**: Critical
- **Rotation**: Every 90 days
- **Storage**: Environment variables, HashiCorp Vault

```bash
# JWT Configuration
JWT_SECRET_KEY=your-very-secure-jwt-secret-key-minimum-32-characters
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600
JWT_REFRESH_EXPIRATION=86400
```

#### OAuth Secrets

- **Purpose**: OAuth 2.0 client authentication
- **Sensitivity**: Critical
- **Rotation**: Every 180 days
- **Storage**: Environment variables, HashiCorp Vault

```bash
# OAuth Configuration
OAUTH_CLIENT_ID=your-oauth-client-id
OAUTH_CLIENT_SECRET=your-oauth-client-secret
OAUTH_AUTHORIZATION_URL=https://auth.example.com/oauth/authorize
OAUTH_TOKEN_URL=https://auth.example.com/oauth/token
```

### Database Secrets

#### Database Credentials

- **Purpose**: Database authentication
- **Sensitivity**: Critical
- **Rotation**: Every 90 days
- **Storage**: Environment variables, HashiCorp Vault

```bash
# Database Credentials
DATABASE_URL=postgresql://username:password@host:port/database
NEO4J_PASSWORD=neo4j-secure-password
REDIS_PASSWORD=redis-secure-password
```

#### Connection Strings

- **Purpose**: Service connectivity
- **Sensitivity**: High
- **Rotation**: Every 90 days
- **Storage**: Environment variables, HashiCorp Vault

```bash
# Connection Strings
DATABASE_URL=postgresql://user:pass@host:port/db
NEO4J_URI=bolt://neo4j:password@host:7687
REDIS_URL=redis://user:pass@host:6379/0
VECTOR_STORE_URL=http://user:pass@host:6333
```

### API Keys

#### External API Keys

- **Purpose**: External service authentication
- **Sensitivity**: High
- **Rotation**: Every 180 days
- **Storage**: Environment variables, HashiCorp Vault

```bash
# External API Keys
OPENALEX_API_KEY=your-openalex-api-key
UNPAYWALL_API_KEY=your-unpaywall-api-key
CLINICALTRIALS_API_KEY=your-clinicaltrials-api-key
FHIR_API_KEY=your-fhir-api-key
RXNORM_API_KEY=your-rxnorm-api-key
ICD11_API_KEY=your-icd11-api-key
```

#### Service API Keys

- **Purpose**: Internal service authentication
- **Sensitivity**: High
- **Rotation**: Every 90 days
- **Storage**: Environment variables, HashiCorp Vault

```bash
# Service API Keys
MINERU_API_KEY=your-mineru-api-key
VLLM_API_KEY=your-vllm-api-key
EMBEDDING_API_KEY=your-embedding-api-key
```

### Encryption Keys

#### Data Encryption Keys

- **Purpose**: Data encryption at rest
- **Sensitivity**: Critical
- **Rotation**: Every 365 days
- **Storage**: HashiCorp Vault, AWS KMS

```bash
# Encryption Keys
DATA_ENCRYPTION_KEY=your-data-encryption-key
BACKUP_ENCRYPTION_KEY=your-backup-encryption-key
CACHE_ENCRYPTION_KEY=your-cache-encryption-key
```

## Secret Storage Strategies

### Environment Variables

#### Development Environment

Use `.env` files for local development:

```bash
# .env file for local development
# DO NOT COMMIT TO VERSION CONTROL

# JWT Secrets
JWT_SECRET_KEY=dev-jwt-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# Database Credentials
DATABASE_URL=postgresql://dev:dev@localhost:5432/medical_kg_dev
NEO4J_PASSWORD=dev_neo4j_password
REDIS_PASSWORD=dev_redis_password

# API Keys
OPENALEX_API_KEY=your_openalex_key_here
UNPAYWALL_API_KEY=your_unpaywall_key_here
CLINICALTRIALS_API_KEY=your_clinicaltrials_key_here

# OAuth Secrets
OAUTH_CLIENT_ID=dev_client_id
OAUTH_CLIENT_SECRET=dev_client_secret
```

#### Production Environment

Set environment variables at the system level:

```bash
# Production environment variables
export JWT_SECRET_KEY="very-secure-jwt-secret-key-for-production"
export DATABASE_URL="postgresql://prod_user:secure_password@prod-db:5432/medical_kg_prod"
export NEO4J_PASSWORD="secure_neo4j_password_for_production"
export REDIS_PASSWORD="secure_redis_password_for_production"
export OPENALEX_API_KEY="production_openalex_api_key"
export UNPAYWALL_API_KEY="production_unpaywall_api_key"
export CLINICALTRIALS_API_KEY="production_clinicaltrials_api_key"
export OAUTH_CLIENT_SECRET="secure_oauth_client_secret_for_production"
```

### HashiCorp Vault Integration

#### Vault Configuration

Configure HashiCorp Vault for secret management:

```python
# Example: HashiCorp Vault integration
import hvac
import os
from typing import Optional

class VaultSecretManager:
    """HashiCorp Vault secret manager."""

    def __init__(self, vault_url: str, vault_token: str):
        self.client = hvac.Client(url=vault_url, token=vault_token)
        self.client.is_authenticated()

    def get_secret(self, path: str, key: str) -> Optional[str]:
        """Get secret from Vault."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data'].get(key)
        except Exception as e:
            logger.error(f"Failed to get secret from Vault: {e}")
            return None

    def set_secret(self, path: str, key: str, value: str) -> bool:
        """Set secret in Vault."""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret={key: value}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set secret in Vault: {e}")
            return False

    def rotate_secret(self, path: str, key: str) -> bool:
        """Rotate secret in Vault."""
        try:
            # Generate new secret
            new_value = self._generate_secret()

            # Update secret
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret={key: new_value}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to rotate secret in Vault: {e}")
            return False

    def _generate_secret(self) -> str:
        """Generate secure random secret."""
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(32))
```

#### Vault Usage Examples

```python
# Example: Using Vault for secret management
vault_manager = VaultSecretManager(
    vault_url="https://vault.example.com",
    vault_token=os.getenv("VAULT_TOKEN")
)

# Get JWT secret
jwt_secret = vault_manager.get_secret("medical-kg-rev/jwt", "secret_key")

# Get database password
db_password = vault_manager.get_secret("medical-kg-rev/database", "password")

# Get API key
api_key = vault_manager.get_secret("medical-kg-rev/api", "openalex_key")
```

### AWS KMS Integration

#### KMS Configuration

Configure AWS KMS for encryption key management:

```python
# Example: AWS KMS integration
import boto3
from botocore.exceptions import ClientError
import base64
import os

class KMSSecretManager:
    """AWS KMS secret manager."""

    def __init__(self, region: str = 'us-east-1'):
        self.kms_client = boto3.client('kms', region_name=region)
        self.key_id = os.getenv('KMS_KEY_ID')

    def encrypt_secret(self, plaintext: str) -> str:
        """Encrypt secret using KMS."""
        try:
            response = self.kms_client.encrypt(
                KeyId=self.key_id,
                Plaintext=plaintext
            )
            return base64.b64encode(response['CiphertextBlob']).decode('utf-8')
        except ClientError as e:
            logger.error(f"Failed to encrypt secret: {e}")
            raise

    def decrypt_secret(self, ciphertext: str) -> str:
        """Decrypt secret using KMS."""
        try:
            ciphertext_blob = base64.b64decode(ciphertext)
            response = self.kms_client.decrypt(
                CiphertextBlob=ciphertext_blob
            )
            return response['Plaintext'].decode('utf-8')
        except ClientError as e:
            logger.error(f"Failed to decrypt secret: {e}")
            raise
```

## Secret Access Patterns

### Lazy Loading

Load secrets only when needed:

```python
# Example: Lazy loading of secrets
class SecretLoader:
    """Lazy secret loader."""

    def __init__(self):
        self._secrets = {}
        self._vault_manager = None

    def get_jwt_secret(self) -> str:
        """Get JWT secret with lazy loading."""
        if 'jwt_secret' not in self._secrets:
            self._secrets['jwt_secret'] = self._load_jwt_secret()
        return self._secrets['jwt_secret']

    def _load_jwt_secret(self) -> str:
        """Load JWT secret from storage."""
        # Try environment variable first
        jwt_secret = os.getenv('JWT_SECRET_KEY')
        if jwt_secret:
            return jwt_secret

        # Try Vault
        if self._vault_manager:
            jwt_secret = self._vault_manager.get_secret("medical-kg-rev/jwt", "secret_key")
            if jwt_secret:
                return jwt_secret

        # Fallback to default (development only)
        if os.getenv('ENVIRONMENT') == 'development':
            return 'dev-jwt-secret-key-change-in-production'

        raise ValueError("JWT secret not found")
```

### Secret Caching

Cache secrets with expiration:

```python
# Example: Secret caching with expiration
import time
from typing import Dict, Optional

class SecretCache:
    """Secret cache with expiration."""

    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, tuple] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[str]:
        """Get secret from cache."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._default_ttl:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set secret in cache."""
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
```

## Secret Rotation

### Automated Rotation

Implement automated secret rotation:

```python
# Example: Automated secret rotation
import schedule
import time
from datetime import datetime, timedelta

class SecretRotationManager:
    """Secret rotation manager."""

    def __init__(self, vault_manager: VaultSecretManager):
        self.vault_manager = vault_manager
        self.rotation_schedule = {
            'jwt_secret': 90,  # days
            'oauth_secret': 180,  # days
            'database_password': 90,  # days
            'api_keys': 180,  # days
        }

    def schedule_rotation(self) -> None:
        """Schedule secret rotation."""
        # Daily check for rotation needs
        schedule.every().day.at("02:00").do(self.check_rotation_needs)

        # Weekly rotation check
        schedule.every().monday.at("03:00").do(self.rotate_expired_secrets)

    def check_rotation_needs(self) -> None:
        """Check if secrets need rotation."""
        for secret_name, rotation_days in self.rotation_schedule.items():
            last_rotation = self._get_last_rotation_date(secret_name)
            if last_rotation:
                days_since_rotation = (datetime.now() - last_rotation).days
                if days_since_rotation >= rotation_days:
                    logger.info(f"Secret {secret_name} needs rotation")
                    self._schedule_secret_rotation(secret_name)

    def rotate_secret(self, secret_name: str) -> bool:
        """Rotate a specific secret."""
        try:
            # Generate new secret
            new_secret = self._generate_secret()

            # Update in Vault
            path = f"medical-kg-rev/{secret_name}"
            success = self.vault_manager.set_secret(path, "value", new_secret)

            if success:
                # Update rotation date
                self._update_rotation_date(secret_name, datetime.now())
                logger.info(f"Successfully rotated secret: {secret_name}")
                return True
            else:
                logger.error(f"Failed to rotate secret: {secret_name}")
                return False

        except Exception as e:
            logger.error(f"Error rotating secret {secret_name}: {e}")
            return False

    def _generate_secret(self) -> str:
        """Generate secure random secret."""
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(32))

    def _get_last_rotation_date(self, secret_name: str) -> Optional[datetime]:
        """Get last rotation date for secret."""
        # Implementation depends on storage backend
        pass

    def _update_rotation_date(self, secret_name: str, date: datetime) -> None:
        """Update rotation date for secret."""
        # Implementation depends on storage backend
        pass
```

### Manual Rotation

Provide manual rotation procedures:

```bash
#!/bin/bash
# Manual secret rotation script

set -e

echo "Starting manual secret rotation..."

# Rotate JWT secret
echo "Rotating JWT secret..."
NEW_JWT_SECRET=$(openssl rand -base64 32)
export JWT_SECRET_KEY="$NEW_JWT_SECRET"
echo "New JWT secret generated"

# Rotate database password
echo "Rotating database password..."
NEW_DB_PASSWORD=$(openssl rand -base64 16)
# Update database password in database
# Update environment variable
export DATABASE_URL="postgresql://user:$NEW_DB_PASSWORD@host:port/db"
echo "New database password generated"

# Rotate API keys
echo "Rotating API keys..."
# Generate new API keys for external services
# Update in respective services
echo "New API keys generated"

# Restart services
echo "Restarting services..."
docker-compose restart
echo "Services restarted"

echo "Secret rotation completed successfully!"
```

## Secret Auditing

### Access Logging

Log all secret access:

```python
# Example: Secret access logging
import logging
from datetime import datetime
from typing import Optional

class SecretAuditor:
    """Secret access auditor."""

    def __init__(self):
        self.audit_logger = logging.getLogger('secret_audit')
        self.audit_logger.setLevel(logging.INFO)

        # Create audit log handler
        handler = logging.FileHandler('/var/log/medical-kg-rev/secret_audit.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)

    def log_secret_access(self, secret_name: str, action: str, user: Optional[str] = None) -> None:
        """Log secret access."""
        self.audit_logger.info(
            f"Secret access - Name: {secret_name}, Action: {action}, User: {user}, "
            f"Timestamp: {datetime.utcnow().isoformat()}"
        )

    def log_secret_rotation(self, secret_name: str, success: bool) -> None:
        """Log secret rotation."""
        status = "SUCCESS" if success else "FAILURE"
        self.audit_logger.info(
            f"Secret rotation - Name: {secret_name}, Status: {status}, "
            f"Timestamp: {datetime.utcnow().isoformat()}"
        )
```

### Compliance Reporting

Generate compliance reports:

```python
# Example: Compliance reporting
from datetime import datetime, timedelta
from typing import List, Dict

class ComplianceReporter:
    """Compliance reporter for secrets."""

    def __init__(self, audit_logger):
        self.audit_logger = audit_logger

    def generate_secret_report(self, days: int = 30) -> Dict:
        """Generate secret compliance report."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        report = {
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'secrets': {},
            'compliance_status': 'COMPLIANT',
            'issues': []
        }

        # Check secret rotation compliance
        for secret_name in ['jwt_secret', 'oauth_secret', 'database_password']:
            last_rotation = self._get_last_rotation_date(secret_name)
            if last_rotation:
                days_since_rotation = (end_date - last_rotation).days
                rotation_threshold = self._get_rotation_threshold(secret_name)

                if days_since_rotation > rotation_threshold:
                    report['compliance_status'] = 'NON_COMPLIANT'
                    report['issues'].append(
                        f"Secret {secret_name} has not been rotated for {days_since_rotation} days "
                        f"(threshold: {rotation_threshold} days)"
                    )

                report['secrets'][secret_name] = {
                    'last_rotation': last_rotation.isoformat(),
                    'days_since_rotation': days_since_rotation,
                    'rotation_threshold': rotation_threshold,
                    'status': 'COMPLIANT' if days_since_rotation <= rotation_threshold else 'NON_COMPLIANT'
                }

        return report

    def _get_last_rotation_date(self, secret_name: str) -> Optional[datetime]:
        """Get last rotation date for secret."""
        # Implementation depends on storage backend
        pass

    def _get_rotation_threshold(self, secret_name: str) -> int:
        """Get rotation threshold for secret."""
        thresholds = {
            'jwt_secret': 90,
            'oauth_secret': 180,
            'database_password': 90,
            'api_keys': 180
        }
        return thresholds.get(secret_name, 365)
```

## Security Best Practices

### Secret Generation

Use cryptographically secure random generation:

```python
# Example: Secure secret generation
import secrets
import string
import hashlib
import base64

def generate_secure_secret(length: int = 32) -> str:
    """Generate cryptographically secure secret."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_jwt_secret() -> str:
    """Generate JWT secret."""
    return generate_secure_secret(64)

def generate_api_key() -> str:
    """Generate API key."""
    return generate_secure_secret(32)

def generate_database_password() -> str:
    """Generate database password."""
    return generate_secure_secret(16)
```

### Secret Validation

Validate secret strength:

```python
# Example: Secret validation
import re
from typing import List

class SecretValidator:
    """Secret validator."""

    def __init__(self):
        self.min_length = 16
        self.max_length = 128
        self.required_chars = {
            'lowercase': r'[a-z]',
            'uppercase': r'[A-Z]',
            'digits': r'[0-9]',
            'special': r'[!@#$%^&*()_+\-=\[\]{};:,.<>?]'
        }

    def validate_secret(self, secret: str) -> List[str]:
        """Validate secret strength."""
        errors = []

        # Length validation
        if len(secret) < self.min_length:
            errors.append(f"Secret must be at least {self.min_length} characters long")

        if len(secret) > self.max_length:
            errors.append(f"Secret must be at most {self.max_length} characters long")

        # Character type validation
        for char_type, pattern in self.required_chars.items():
            if not re.search(pattern, secret):
                errors.append(f"Secret must contain at least one {char_type} character")

        # Common weak secrets
        weak_secrets = [
            'password', '123456', 'secret', 'admin', 'root',
            'test', 'demo', 'example', 'default', 'changeme'
        ]

        if secret.lower() in weak_secrets:
            errors.append("Secret is too weak (common password)")

        return errors

    def is_secret_strong(self, secret: str) -> bool:
        """Check if secret is strong."""
        return len(self.validate_secret(secret)) == 0
```

### Secret Encryption

Encrypt secrets at rest:

```python
# Example: Secret encryption
from cryptography.fernet import Fernet
import base64
import os

class SecretEncryptor:
    """Secret encryptor."""

    def __init__(self, key: Optional[str] = None):
        if key:
            self.key = key.encode()
        else:
            self.key = Fernet.generate_key()

        self.cipher = Fernet(self.key)

    def encrypt_secret(self, secret: str) -> str:
        """Encrypt secret."""
        encrypted = self.cipher.encrypt(secret.encode())
        return base64.b64encode(encrypted).decode('utf-8')

    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt secret."""
        encrypted_bytes = base64.b64decode(encrypted_secret.encode('utf-8'))
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')

    def get_key(self) -> str:
        """Get encryption key."""
        return base64.b64encode(self.key).decode('utf-8')
```

## Troubleshooting

### Common Issues

1. **Missing Secrets**
   - Error: `JWT_SECRET_KEY not found`
   - Solution: Set environment variable or configure Vault

2. **Weak Secrets**
   - Error: `Secret is too weak`
   - Solution: Use stronger secret generation

3. **Rotation Failures**
   - Error: `Failed to rotate secret`
   - Solution: Check Vault connectivity and permissions

4. **Access Denied**
   - Error: `Access denied to secret`
   - Solution: Check Vault policies and authentication

### Debug Commands

```bash
# Check environment variables
env | grep -E "(JWT|OAUTH|DATABASE|REDIS|NEO4J)" | sed 's/=.*/=***/'

# Test Vault connectivity
vault status

# List Vault secrets
vault kv list medical-kg-rev/

# Get secret from Vault
vault kv get medical-kg-rev/jwt

# Test secret validation
python -c "from Medical_KG_rev.security.secrets import SecretValidator; v = SecretValidator(); print(v.validate_secret('test123'))"

# Check secret rotation status
python -c "from Medical_KG_rev.security.secrets import ComplianceReporter; r = ComplianceReporter(); print(r.generate_secret_report())"
```

## Best Practices

1. **Never Commit Secrets**: Keep secrets out of version control
2. **Use Strong Secrets**: Generate cryptographically secure secrets
3. **Rotate Regularly**: Implement automated secret rotation
4. **Audit Access**: Log all secret access and changes
5. **Encrypt at Rest**: Encrypt secrets when stored
6. **Limit Access**: Use principle of least privilege
7. **Monitor Changes**: Alert on secret modifications
8. **Test Rotation**: Regularly test secret rotation procedures
9. **Backup Secrets**: Maintain secure backups of secrets
10. **Document Procedures**: Keep secret management procedures documented

## Related Documentation

- [Configuration Reference](configuration_reference.md)
- [Environment Variables](environment_variables.md)
- [Security Configuration](security.md)
- [Deployment Guide](deployment.md)
- [Troubleshooting Guide](troubleshooting.md)
