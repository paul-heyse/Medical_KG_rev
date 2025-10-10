# Environment Variables Reference

This document provides a comprehensive reference for all environment variables used in the Medical_KG_rev system, including their purpose, default values, and precedence rules.

## Overview

The Medical_KG_rev system uses environment variables for configuration management with the following precedence order (highest to lowest):

1. **System Environment Variables** - Set at the OS level
2. **`.env` File** - Local development overrides
3. **Configuration Files** - Default values from YAML files
4. **Code Defaults** - Fallback values in application code

## Environment Variable Categories

### Core System Variables

#### Gateway Configuration

```bash
# API Gateway Settings
GATEWAY_HOST=0.0.0.0                    # Gateway host address
GATEWAY_PORT=8000                       # Gateway port number
GATEWAY_WORKERS=4                       # Number of worker processes
GATEWAY_LOG_LEVEL=INFO                  # Logging level (DEBUG, INFO, WARNING, ERROR)
GATEWAY_CORS_ORIGINS=*                  # CORS allowed origins (comma-separated)
GATEWAY_CORS_METHODS=GET,POST,PUT,DELETE # CORS allowed methods
GATEWAY_CORS_HEADERS=Authorization,Content-Type # CORS allowed headers
```

#### Database Configuration

```bash
# Primary Database
DATABASE_URL=postgresql://user:pass@host:port/dbname  # Primary database connection string
DATABASE_POOL_SIZE=20                   # Connection pool size
DATABASE_MAX_OVERFLOW=30                # Maximum pool overflow
DATABASE_TIMEOUT=30                    # Connection timeout in seconds
DATABASE_ECHO=false                     # Enable SQL query logging

# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687        # Neo4j connection URI
NEO4J_USER=neo4j                       # Neo4j username
NEO4J_PASSWORD=password                 # Neo4j password
NEO4J_MAX_CONNECTIONS=50               # Maximum Neo4j connections

# Redis Cache
REDIS_URL=redis://localhost:6379/0     # Redis connection URL
REDIS_PASSWORD=                         # Redis password (if required)
REDIS_MAX_CONNECTIONS=100              # Maximum Redis connections
REDIS_SOCKET_TIMEOUT=5                 # Redis socket timeout
REDIS_SOCKET_CONNECT_TIMEOUT=5         # Redis connection timeout
```

#### Vector Store Configuration

```bash
# Vector Store Settings
VECTOR_STORE_TYPE=qdrant               # Vector store type (qdrant, weaviate, chroma)
VECTOR_STORE_URL=http://localhost:6333 # Vector store connection URL
VECTOR_DIMENSION=768                   # Vector dimension size
VECTOR_COLLECTION_NAME=medical_embeddings # Collection name
VECTOR_DISTANCE_METRIC=cosine          # Distance metric (cosine, euclidean, dot)
```

### Service Configuration

#### Embedding Service

```bash
# Embedding Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Embedding model name
EMBEDDING_BATCH_SIZE=32                # Batch size for embedding generation
EMBEDDING_DEVICE=cpu                   # Device for computation (cpu, cuda)
EMBEDDING_MAX_LENGTH=512                # Maximum input length
EMBEDDING_POOLING_MODE=mean            # Pooling mode (mean, max, cls)

# vLLM Configuration
VLLM_ENABLED=false                     # Enable vLLM for embeddings
VLLM_MODEL_PATH=/models/embedding      # Path to vLLM model
VLLM_TENSOR_PARALLEL_SIZE=1           # Tensor parallel size
VLLM_MAX_MODEL_LEN=4096               # Maximum model length
VLLM_GPU_MEMORY_UTILIZATION=0.8       # GPU memory utilization
```

#### Chunking Service

```bash
# Chunking Configuration
CHUNKING_STRATEGY=semantic             # Chunking strategy (semantic, fixed, adaptive)
CHUNK_SIZE=512                         # Default chunk size
CHUNK_OVERLAP=50                       # Chunk overlap size
CHUNK_MIN_SIZE=200                     # Minimum chunk size
CHUNK_MAX_SIZE=2048                    # Maximum chunk size
CHUNK_SEMANTIC_THRESHOLD=0.8           # Semantic similarity threshold
```

#### GPU Services

```bash
# GPU Configuration
GPU_ENABLED=false                      # Enable GPU services
CUDA_VISIBLE_DEVICES=0                 # CUDA device IDs (comma-separated)
GPU_MEMORY_FRACTION=0.8                # GPU memory fraction
GPU_MEMORY_GROWTH=true                 # Enable memory growth

# MinerU Configuration
MINERU_MODEL_PATH=/models/mineru       # MinerU model path
MINERU_BATCH_SIZE=1                    # MinerU batch size
MINERU_MAX_LENGTH=4096                 # MinerU maximum length
MINERU_DEVICE=cuda                     # MinerU device

# Docling Gemma3 VLM Configuration
DOCLING_VLM_MODEL_PATH=/models/gemma3-12b   # Location of the Gemma3 checkpoint
DOCLING_VLM_BATCH_SIZE=8                    # Number of pages processed per batch
DOCLING_VLM_TIMEOUT_SECONDS=300             # Processing timeout per PDF
DOCLING_VLM_RETRY_ATTEMPTS=3                # Retry attempts for transient failures
DOCLING_VLM_GPU_MEMORY_FRACTION=0.95        # Fraction of GPU memory reserved for Docling
PDF_PROCESSING_BACKEND=docling_vlm          # Active PDF backend (mineru or docling_vlm)
```

##### Example `.env` snippet

```bash
# Gemma3 checkpoint location and runtime behaviour
DOCLING_VLM_MODEL_PATH=/models/gemma3-12b
DOCLING_VLM_RETRY_ATTEMPTS=5
DOCLING_VLM_TIMEOUT_SECONDS=420
DOCLING_VLM_GPU_MEMORY_FRACTION=0.95
PDF_PROCESSING_BACKEND=docling_vlm

# Optional Hugging Face access token for model downloads
HUGGINGFACE_HUB_TOKEN=hf_xxx
```

##### Validation tips

1. Download the checkpoint once before bootstrapping services:

   ```bash
   python scripts/download_gemma3.py --target "$DOCLING_VLM_MODEL_PATH"
   ```

2. Confirm the readiness probe reports the Docling backend as healthy:

   ```bash
   curl -s localhost:8000/ready | jq '.checks.docling_vlm'
   ```

3. If the health check reports insufficient memory, lower
   `DOCLING_VLM_BATCH_SIZE` or provision a 24GB+ GPU as described in
   `config/gpu.yaml`.

### Orchestration Configuration

#### Kafka Configuration

```bash
# Kafka Settings
KAFKA_BOOTSTRAP_SERVERS=localhost:9092  # Kafka bootstrap servers (comma-separated)
KAFKA_GROUP_ID=medical-kg-rev          # Kafka consumer group ID
KAFKA_AUTO_OFFSET_RESET=earliest       # Auto offset reset policy
KAFKA_ENABLE_AUTO_COMMIT=true          # Enable auto commit
KAFKA_SESSION_TIMEOUT_MS=30000         # Session timeout in milliseconds
KAFKA_HEARTBEAT_INTERVAL_MS=10000      # Heartbeat interval in milliseconds
KAFKA_MAX_POLL_RECORDS=500             # Maximum poll records
KAFKA_CONSUMER_TIMEOUT_MS=1000         # Consumer timeout in milliseconds
```

#### Dagster Configuration

```bash
# Dagster Settings
DAGSTER_HOME=/opt/dagster/dagster_home # Dagster home directory
DAGSTER_STORAGE_DIR=/opt/dagster/storage # Dagster storage directory
DAGSTER_LOG_LEVEL=INFO                 # Dagster log level
DAGSTER_WORKSPACE_FILE=workspace.yaml  # Dagster workspace file
DAGSTER_SCHEDULER_ENABLED=true         # Enable Dagster scheduler
DAGSTER_DAEMON_ENABLED=true            # Enable Dagster daemon
```

### Security Configuration

#### Authentication & Authorization

```bash
# JWT Configuration
JWT_SECRET_KEY=your-secret-key         # JWT signing secret key
JWT_ALGORITHM=HS256                    # JWT algorithm (HS256, RS256)
JWT_EXPIRATION=3600                    # JWT expiration time in seconds
JWT_REFRESH_EXPIRATION=86400           # JWT refresh token expiration
JWT_ISSUER=medical-kg-rev              # JWT issuer
JWT_AUDIENCE=medical-kg-rev-api        # JWT audience

# OAuth Configuration
OAUTH_CLIENT_ID=your-client-id         # OAuth client ID
OAUTH_CLIENT_SECRET=your-client-secret  # OAuth client secret
OAUTH_AUTHORIZATION_URL=https://auth.example.com/oauth/authorize  # OAuth authorization URL
OAUTH_TOKEN_URL=https://auth.example.com/oauth/token              # OAuth token URL
OAUTH_REDIRECT_URI=https://app.example.com/callback               # OAuth redirect URI
OAUTH_SCOPE=read write                 # OAuth scope (space-separated)
```

#### Rate Limiting

```bash
# Rate Limiting Configuration
RATE_LIMIT_ENABLED=true                # Enable rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100     # Requests per minute limit
RATE_LIMIT_BURST_SIZE=200               # Burst size limit
RATE_LIMIT_STORAGE_URL=redis://localhost:6379/1  # Rate limit storage URL
RATE_LIMIT_KEY_PREFIX=rate_limit:      # Rate limit key prefix
```

### Monitoring & Observability

#### Prometheus Configuration

```bash
# Prometheus Settings
PROMETHEUS_ENABLED=true                # Enable Prometheus metrics
PROMETHEUS_PORT=9090                   # Prometheus metrics port
PROMETHEUS_PATH=/metrics               # Prometheus metrics path
PROMETHEUS_NAMESPACE=medical_kg_rev    # Prometheus namespace
PROMETHEUS_SUBSYSTEM=api               # Prometheus subsystem
```

#### Jaeger Configuration

```bash
# Jaeger Tracing
JAEGER_ENABLED=true                    # Enable Jaeger tracing
JAEGER_ENDPOINT=http://localhost:14268/api/traces  # Jaeger endpoint
JAEGER_SERVICE_NAME=medical-kg-rev     # Jaeger service name
JAEGER_SAMPLING_RATE=0.1               # Jaeger sampling rate (0.0-1.0)
JAEGER_TAGS=environment=production     # Jaeger tags (comma-separated)
```

#### Logging Configuration

```bash
# Logging Settings
LOG_LEVEL=INFO                         # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_FORMAT=json                        # Log format (json, text)
LOG_OUTPUT=stdout                      # Log output (stdout, stderr, file)
LOG_FILE_PATH=/var/log/medical-kg-rev/app.log  # Log file path
LOG_MAX_SIZE=100MB                     # Maximum log file size
LOG_BACKUP_COUNT=5                     # Number of backup log files
LOG_ROTATION_INTERVAL=1d               # Log rotation interval
```

### External API Configuration

#### Biomedical APIs

```bash
# OpenAlex API
OPENALEX_API_KEY=your-openalex-key     # OpenAlex API key
OPENALEX_BASE_URL=https://api.openalex.org  # OpenAlex base URL
OPENALEX_RATE_LIMIT=100                # OpenAlex rate limit per minute
OPENALEX_TIMEOUT=30                    # OpenAlex timeout in seconds

# Unpaywall API
UNPAYWALL_API_KEY=your-unpaywall-key   # Unpaywall API key
UNPAYWALL_BASE_URL=https://api.unpaywall.org  # Unpaywall base URL
UNPAYWALL_RATE_LIMIT=200               # Unpaywall rate limit per minute
UNPAYWALL_TIMEOUT=30                   # Unpaywall timeout in seconds

# ClinicalTrials.gov API
CLINICALTRIALS_API_KEY=your-ct-key      # ClinicalTrials.gov API key
CLINICALTRIALS_BASE_URL=https://clinicaltrials.gov/api/v2  # ClinicalTrials.gov base URL
CLINICALTRIALS_RATE_LIMIT=1000         # ClinicalTrials.gov rate limit per minute
CLINICALTRIALS_TIMEOUT=60               # ClinicalTrials.gov timeout in seconds
```

#### FHIR & Medical Standards

```bash
# FHIR Configuration
FHIR_BASE_URL=https://hapi.fhir.org/baseR4  # FHIR base URL
FHIR_API_KEY=your-fhir-key             # FHIR API key
FHIR_TIMEOUT=30                        # FHIR timeout in seconds
FHIR_RATE_LIMIT=100                    # FHIR rate limit per minute

# RxNorm Configuration
RXNORM_BASE_URL=https://rxnav.nlm.nih.gov/REST  # RxNorm base URL
RXNORM_API_KEY=your-rxnorm-key         # RxNorm API key
RXNORM_TIMEOUT=30                      # RxNorm timeout in seconds
RXNORM_RATE_LIMIT=100                  # RxNorm rate limit per minute

# ICD-11 Configuration
ICD11_BASE_URL=https://id.who.int/icd/release/11  # ICD-11 base URL
ICD11_API_KEY=your-icd11-key           # ICD-11 API key
ICD11_TIMEOUT=30                       # ICD-11 timeout in seconds
ICD11_RATE_LIMIT=100                   # ICD-11 rate limit per minute
```

## Environment Variable Precedence

### Precedence Order

1. **System Environment Variables** (Highest Priority)
   - Set at the OS level
   - Cannot be overridden by application code
   - Example: `export DATABASE_URL="postgresql://prod:pass@prod-db:5432/medical_kg"`

2. **`.env` File** (Development Override)
   - Local development overrides
   - Not committed to version control
   - Example: `DATABASE_URL=postgresql://dev:pass@localhost:5432/medical_kg_dev`

3. **Configuration Files** (Default Values)
   - YAML configuration files
   - Committed to version control
   - Example: `database.url: "postgresql://default:pass@localhost:5432/medical_kg"`

4. **Code Defaults** (Lowest Priority)
   - Fallback values in application code
   - Used when no other value is provided
   - Example: `DEFAULT_DATABASE_URL = "postgresql://fallback:pass@localhost:5432/fallback"`

### Precedence Examples

#### Example 1: Database URL Override

```bash
# System environment variable (highest priority)
export DATABASE_URL="postgresql://prod:pass@prod-db:5432/medical_kg"

# .env file (development override)
DATABASE_URL=postgresql://dev:pass@localhost:5432/medical_kg_dev

# config/database.yaml (default)
database:
  url: "postgresql://default:pass@localhost:5432/medical_kg"

# Code default (lowest priority)
DEFAULT_DATABASE_URL = "postgresql://fallback:pass@localhost:5432/fallback"

# Result: Uses "postgresql://prod:pass@prod-db:5432/medical_kg"
```

#### Example 2: Log Level Override

```bash
# System environment variable
export GATEWAY_LOG_LEVEL="DEBUG"

# .env file
GATEWAY_LOG_LEVEL=INFO

# config/gateway.yaml
gateway:
  log_level: "WARNING"

# Code default
DEFAULT_LOG_LEVEL = "ERROR"

# Result: Uses "DEBUG"
```

## Environment Variable Validation

### Validation Rules

1. **Required Variables**: Must be present and non-empty
2. **Type Validation**: Values must match expected types
3. **Format Validation**: String values must match expected formats
4. **Range Validation**: Numeric values must be within valid ranges

### Validation Examples

```python
# Example validation for database URL
def validate_database_url(url: str) -> bool:
    if not url:
        raise ValueError("DATABASE_URL is required")

    if not url.startswith(('postgresql://', 'postgres://')):
        raise ValueError("DATABASE_URL must start with postgresql:// or postgres://")

    return True

# Example validation for port number
def validate_port(port: str) -> bool:
    try:
        port_num = int(port)
        if not (1 <= port_num <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return True
    except ValueError as e:
        raise ValueError(f"Invalid port number: {e}")

# Example validation for log level
def validate_log_level(level: str) -> bool:
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level.upper() not in valid_levels:
        raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
    return True
```

## Secret Management

### Environment Variables for Secrets

Never commit secrets to version control. Use environment variables:

```bash
# Database credentials
export DATABASE_URL="postgresql://user:password@host:port/db"
export NEO4J_PASSWORD="neo4j_password"
export REDIS_PASSWORD="redis_password"

# API keys
export OPENALEX_API_KEY="your_openalex_key"
export UNPAYWALL_API_KEY="your_unpaywall_key"
export CLINICALTRIALS_API_KEY="your_clinicaltrials_key"

# JWT secrets
export JWT_SECRET_KEY="your_jwt_secret_key"
export OAUTH_CLIENT_SECRET="your_oauth_secret"

# Vector store credentials
export VECTOR_STORE_URL="http://user:password@host:port"
```

### Secret Rotation

1. **Automated Rotation**: Use tools like HashiCorp Vault
2. **Manual Rotation**: Update environment variables
3. **Graceful Degradation**: Support multiple secrets during rotation
4. **Audit Logging**: Log all secret access and changes

### Secret Validation

```python
# Example secret validation
def validate_secret(secret: str, min_length: int = 32) -> bool:
    if not secret:
        raise ValueError("Secret is required")

    if len(secret) < min_length:
        raise ValueError(f"Secret must be at least {min_length} characters long")

    # Check for common weak secrets
    weak_secrets = ['password', '123456', 'secret', 'admin']
    if secret.lower() in weak_secrets:
        raise ValueError("Secret is too weak")

    return True
```

## Development Environment Setup

### `.env` File Template

Create a `.env` file in the project root for local development:

```bash
# .env file for local development
# Copy this file to .env and update values as needed

# Gateway Configuration
GATEWAY_HOST=localhost
GATEWAY_PORT=8000
GATEWAY_LOG_LEVEL=DEBUG

# Database Configuration
DATABASE_URL=postgresql://dev:dev@localhost:5432/medical_kg_dev
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=dev_password
REDIS_URL=redis://localhost:6379/0

# Vector Store Configuration
VECTOR_STORE_TYPE=qdrant
VECTOR_STORE_URL=http://localhost:6333
VECTOR_DIMENSION=768

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=cpu

# Chunking Configuration
CHUNKING_STRATEGY=semantic
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# GPU Configuration
GPU_ENABLED=false
CUDA_VISIBLE_DEVICES=0

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_GROUP_ID=medical-kg-rev-dev

# Security Configuration
JWT_SECRET_KEY=dev_jwt_secret_key_change_in_production
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# Monitoring Configuration
PROMETHEUS_ENABLED=false
JAEGER_ENABLED=false
LOG_LEVEL=DEBUG
LOG_FORMAT=text

# External API Configuration
OPENALEX_API_KEY=your_openalex_key_here
UNPAYWALL_API_KEY=your_unpaywall_key_here
CLINICALTRIALS_API_KEY=your_clinicaltrials_key_here
```

### Production Environment Setup

For production, set environment variables at the system level:

```bash
# Production environment variables
export GATEWAY_HOST=0.0.0.0
export GATEWAY_PORT=8000
export GATEWAY_LOG_LEVEL=INFO

export DATABASE_URL="postgresql://prod_user:secure_password@prod-db:5432/medical_kg_prod"
export NEO4J_URI="bolt://prod-neo4j:7687"
export NEO4J_PASSWORD="secure_neo4j_password"
export REDIS_URL="redis://prod-redis:6379/0"

export VECTOR_STORE_TYPE=qdrant
export VECTOR_STORE_URL="http://prod-qdrant:6333"
export VECTOR_DIMENSION=768

export JWT_SECRET_KEY="very_secure_jwt_secret_key_for_production"
export OAUTH_CLIENT_SECRET="secure_oauth_client_secret"

export PROMETHEUS_ENABLED=true
export JAEGER_ENABLED=true
export LOG_LEVEL=INFO
export LOG_FORMAT=json
```

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**: Check `.env` file and system environment
2. **Invalid Values**: Validate environment variable values
3. **Precedence Conflicts**: Understand override order
4. **Secret Exposure**: Never log sensitive environment variables
5. **Type Mismatches**: Ensure values match expected types

### Debug Commands

```bash
# Check environment variables
env | grep -E "(DATABASE|REDIS|NEO4J|JWT|OAUTH)"

# Check specific variable
echo $DATABASE_URL

# Validate environment variable
python -c "import os; print('DATABASE_URL:', os.getenv('DATABASE_URL', 'NOT_SET'))"

# Test configuration loading
python -c "from Medical_KG_rev.config import Config; config = Config(); print('Database URL:', config.database.url)"

# Check environment variable precedence
python -c "
import os
from Medical_KG_rev.config import Config
config = Config()
print('System env:', os.getenv('DATABASE_URL'))
print('Config value:', config.database.url)
print('Final value:', config.get_database_url())
"
```

### Validation Script

Create a validation script to check all environment variables:

```python
#!/usr/bin/env python3
"""Environment variable validation script."""

import os
import sys
from typing import Dict, List, Optional

def validate_required_vars(required_vars: List[str]) -> Dict[str, bool]:
    """Validate required environment variables."""
    results = {}
    for var in required_vars:
        value = os.getenv(var)
        results[var] = value is not None and value.strip() != ""
    return results

def validate_numeric_vars(numeric_vars: Dict[str, tuple]) -> Dict[str, bool]:
    """Validate numeric environment variables."""
    results = {}
    for var, (min_val, max_val) in numeric_vars.items():
        value = os.getenv(var)
        if value is None:
            results[var] = False
            continue

        try:
            num_val = int(value)
            results[var] = min_val <= num_val <= max_val
        except ValueError:
            results[var] = False
    return results

def main():
    """Main validation function."""
    # Required environment variables
    required_vars = [
        'DATABASE_URL',
        'NEO4J_URI',
        'NEO4J_PASSWORD',
        'REDIS_URL',
        'JWT_SECRET_KEY',
    ]

    # Numeric environment variables (var_name: (min_value, max_value))
    numeric_vars = {
        'GATEWAY_PORT': (1, 65535),
        'DATABASE_POOL_SIZE': (1, 100),
        'DATABASE_MAX_OVERFLOW': (0, 100),
        'DATABASE_TIMEOUT': (1, 300),
        'JWT_EXPIRATION': (60, 86400),
    }

    # Validate required variables
    required_results = validate_required_vars(required_vars)
    numeric_results = validate_numeric_vars(numeric_vars)

    # Print results
    print("Environment Variable Validation Results:")
    print("=" * 50)

    all_valid = True

    print("\nRequired Variables:")
    for var, valid in required_results.items():
        status = "✓" if valid else "✗"
        print(f"  {status} {var}")
        if not valid:
            all_valid = False

    print("\nNumeric Variables:")
    for var, valid in numeric_results.items():
        status = "✓" if valid else "✗"
        print(f"  {status} {var}")
        if not valid:
            all_valid = False

    if all_valid:
        print("\n✓ All environment variables are valid!")
        sys.exit(0)
    else:
        print("\n✗ Some environment variables are invalid!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Best Practices

1. **Use Environment Variables**: For all configuration, especially secrets
2. **Validate Early**: Check environment variables at startup
3. **Fail Fast**: Exit immediately on validation errors
4. **Document Changes**: Update this reference when adding new variables
5. **Test Configurations**: Validate in development before production
6. **Version Control**: Keep `.env` files out of version control
7. **Backup Configurations**: Maintain backups of production configurations
8. **Monitor Changes**: Log environment variable changes for audit purposes
9. **Use Secure Defaults**: Never use insecure default values
10. **Rotate Secrets**: Regularly rotate sensitive environment variables

## Related Documentation

- [Configuration Reference](configuration_reference.md)
- [Environment Setup Guide](environment_setup.md)
- [Deployment Guide](deployment.md)
- [Security Configuration](security.md)
- [Troubleshooting Guide](troubleshooting.md)
