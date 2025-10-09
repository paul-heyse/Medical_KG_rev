# Configuration Reference

This document provides a comprehensive reference for all configuration files, environment variables, and settings used in the Medical_KG_rev system.

## Overview

The Medical_KG_rev system uses a hierarchical configuration system with the following precedence (highest to lowest):

1. Environment variables
2. Configuration files in `/config`
3. Default values in code

## Configuration Directory Structure

```
config/
├── chunking/
│   ├── profiles/           # Chunking strategy profiles
│   └── chunking.yaml      # Main chunking configuration
├── embedding/
│   ├── namespaces/        # Embedding namespace configurations
│   ├── namespaces.yaml    # Namespace definitions
│   ├── pyserini.yaml      # Pyserini configuration
│   └── vllm.yaml         # vLLM configuration
├── mineru.yaml           # MinerU PDF processing configuration
├── embeddings.yaml       # Embedding service configuration
├── orchestration/
│   ├── pipelines/         # Pipeline configurations
│   ├── pipelines.yaml     # Pipeline definitions
│   ├── resilience.yaml    # Resilience and retry policies
│   └── versions/         # Version-specific configurations
├── retrieval/
│   ├── components.yaml    # Retrieval component configuration
│   ├── reranking_models.yaml  # Reranking model configurations
│   └── reranking.yaml    # Reranking service configuration
├── vector_store.yaml     # Vector store configuration
├── monitoring/
│   └── rollback_triggers.yaml  # Monitoring rollback triggers
└── dagster.yaml         # Dagster orchestration configuration
```

## Core Configuration Files

### Gateway Configuration

#### Environment Variables

- `GATEWAY_HOST`: API gateway host (default: `0.0.0.0`)
- `GATEWAY_PORT`: API gateway port (default: `8000`)
- `GATEWAY_WORKERS`: Number of worker processes (default: `4`)
- `GATEWAY_LOG_LEVEL`: Logging level (default: `INFO`)

#### Configuration File: `config/gateway.yaml`

```yaml
gateway:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  log_level: "INFO"
  cors:
    origins: ["*"]
    methods: ["GET", "POST", "PUT", "DELETE"]
    headers: ["Authorization", "Content-Type"]

  protocols:
    rest:
      enabled: true
      openapi_version: "3.0.3"
      json_api: true
      odata: true
    graphql:
      enabled: true
      introspection: true
      playground: true
    grpc:
      enabled: true
      max_message_size: 4194304
    soap:
      enabled: false
    asyncapi:
      enabled: true
      sse: true
```

### Database Configuration

#### Environment Variables

- `DATABASE_URL`: Primary database connection string
- `DATABASE_POOL_SIZE`: Connection pool size (default: `20`)
- `DATABASE_MAX_OVERFLOW`: Maximum pool overflow (default: `30`)
- `DATABASE_TIMEOUT`: Connection timeout in seconds (default: `30`)

#### Configuration File: `config/database.yaml`

```yaml
database:
  primary:
    url: "${DATABASE_URL}"
    pool_size: 20
    max_overflow: 30
    timeout: 30
    echo: false

  neo4j:
    uri: "${NEO4J_URI}"
    user: "${NEO4J_USER}"
    password: "${NEO4J_PASSWORD}"
    max_connections: 50

  redis:
    url: "${REDIS_URL}"
    max_connections: 100
    socket_timeout: 5
    socket_connect_timeout: 5
```

### Vector Store Configuration

#### Environment Variables

- `VECTOR_STORE_TYPE`: Vector store type (`qdrant`, `weaviate`, `chroma`)
- `VECTOR_STORE_URL`: Vector store connection URL
- `VECTOR_DIMENSION`: Vector dimension (default: `768`)

#### Configuration File: `config/vector_store.yaml`

```yaml
vector_store:
  type: "qdrant"
  url: "${VECTOR_STORE_URL}"
  dimension: 768

  qdrant:
    collection_name: "medical_embeddings"
    distance_metric: "cosine"
    hnsw_config:
      m: 16
      ef_construct: 200
      ef: 100

  weaviate:
    class_name: "MedicalDocument"
    vectorizer: "none"

  chroma:
    collection_name: "medical_docs"
    distance_function: "cosine"
```

### Embedding Configuration

#### Environment Variables

- `EMBEDDING_MODEL`: Embedding model name (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `EMBEDDING_BATCH_SIZE`: Batch size for embedding generation (default: `32`)
- `EMBEDDING_DEVICE`: Device for embedding computation (`cpu`, `cuda`)

#### Configuration File: `config/embeddings.yaml`

```yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  device: "cpu"

  vllm:
    enabled: false
    model_path: "/models/embedding"
    tensor_parallel_size: 1
    max_model_len: 4096

  pyserini:
    index_path: "/indices/pyserini"
    k1: 0.9
    b: 0.4
```

### Chunking Configuration

#### Environment Variables

- `CHUNKING_STRATEGY`: Chunking strategy (`semantic`, `fixed`, `adaptive`)
- `CHUNK_SIZE`: Default chunk size (default: `512`)
- `CHUNK_OVERLAP`: Chunk overlap size (default: `50`)

#### Configuration File: `config/chunking.yaml`

```yaml
chunking:
  strategy: "semantic"
  default_chunk_size: 512
  default_overlap: 50

  profiles:
    medical_papers:
      strategy: "semantic"
      chunk_size: 1024
      overlap: 100
      min_chunk_size: 200
      max_chunk_size: 2048

    clinical_trials:
      strategy: "fixed"
      chunk_size: 512
      overlap: 50

    regulatory_docs:
      strategy: "adaptive"
      chunk_size: 768
      overlap: 75
      semantic_threshold: 0.8
```

### Orchestration Configuration

#### Environment Variables

- `KAFKA_BOOTSTRAP_SERVERS`: Kafka bootstrap servers
- `KAFKA_GROUP_ID`: Kafka consumer group ID
- `DAGSTER_HOME`: Dagster home directory

#### Configuration File: `config/orchestration/pipelines.yaml`

```yaml
pipelines:
  auto_pipeline:
    enabled: true
    max_concurrent_jobs: 10
    timeout: 3600

  manual_pipeline:
    enabled: true
    max_concurrent_jobs: 5
    timeout: 7200
    gpu_required: true

  resilience:
    max_retries: 3
    backoff_factor: 2
    retry_delay: 60
    circuit_breaker_threshold: 5
```

### GPU Services Configuration

#### Environment Variables

- `GPU_ENABLED`: Enable GPU services (default: `false`)
- `CUDA_VISIBLE_DEVICES`: CUDA device IDs
- `GPU_MEMORY_FRACTION`: GPU memory fraction (default: `0.8`)

#### Configuration File: `config/gpu.yaml`

```yaml
gpu:
  enabled: false
  cuda_visible_devices: "0"
  memory_fraction: 0.8

  mineru:
    model_path: "/models/mineru"
    batch_size: 1
    max_length: 4096

  vllm:
    model_path: "/models/vllm"
    tensor_parallel_size: 1
    max_model_len: 4096
    gpu_memory_utilization: 0.8
```

### Security Configuration

#### Environment Variables

- `JWT_SECRET_KEY`: JWT signing secret key
- `JWT_ALGORITHM`: JWT algorithm (default: `HS256`)
- `JWT_EXPIRATION`: JWT expiration time (default: `3600`)
- `OAUTH_CLIENT_ID`: OAuth client ID
- `OAUTH_CLIENT_SECRET`: OAuth client secret

#### Configuration File: `config/security.yaml`

```yaml
security:
  jwt:
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "HS256"
    expiration: 3600

  oauth:
    client_id: "${OAUTH_CLIENT_ID}"
    client_secret: "${OAUTH_CLIENT_SECRET}"
    authorization_url: "https://auth.example.com/oauth/authorize"
    token_url: "https://auth.example.com/oauth/token"

  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_size: 200

  cors:
    origins: ["https://app.example.com"]
    methods: ["GET", "POST", "PUT", "DELETE"]
    headers: ["Authorization", "Content-Type"]
```

### Monitoring Configuration

#### Environment Variables

- `PROMETHEUS_PORT`: Prometheus metrics port (default: `9090`)
- `JAEGER_ENDPOINT`: Jaeger tracing endpoint
- `LOG_LEVEL`: Logging level (default: `INFO`)

#### Configuration File: `config/monitoring.yaml`

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
    path: "/metrics"

  jaeger:
    enabled: true
    endpoint: "${JAEGER_ENDPOINT}"
    service_name: "medical-kg-rev"

  logging:
    level: "INFO"
    format: "json"
    output: "stdout"

  health_checks:
    enabled: true
    interval: 30
    timeout: 10
```

## Configuration Validation

### Validation Rules

1. **Required Fields**: All marked fields must be present
2. **Type Validation**: Values must match expected types
3. **Range Validation**: Numeric values must be within valid ranges
4. **Format Validation**: String values must match expected formats

### Validation Examples

```python
# Example validation for database configuration
def validate_database_config(config: dict) -> bool:
    required_fields = ['url', 'pool_size', 'max_overflow']

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(config['pool_size'], int) or config['pool_size'] <= 0:
        raise ValueError("pool_size must be a positive integer")

    if config['max_overflow'] < 0:
        raise ValueError("max_overflow must be non-negative")

    return True
```

## Environment Variable Precedence

### Override Order

1. **System Environment Variables**: Highest priority
2. **`.env` file**: Local development overrides
3. **Configuration Files**: Default values from YAML files
4. **Code Defaults**: Fallback values in application code

### Example Override Chain

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

# JWT secrets
export JWT_SECRET_KEY="your_jwt_secret_key"
export OAUTH_CLIENT_SECRET="your_oauth_secret"
```

### Secret Rotation

1. **Automated Rotation**: Use tools like HashiCorp Vault
2. **Manual Rotation**: Update environment variables
3. **Graceful Degradation**: Support multiple secrets during rotation
4. **Audit Logging**: Log all secret access and changes

## Configuration Templates

### Development Template

```yaml
# config/development.yaml
gateway:
  host: "localhost"
  port: 8000
  log_level: "DEBUG"

database:
  url: "postgresql://dev:dev@localhost:5432/medical_kg_dev"
  echo: true

monitoring:
  prometheus:
    enabled: false
  jaeger:
    enabled: false
```

### Production Template

```yaml
# config/production.yaml
gateway:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"

database:
  url: "${DATABASE_URL}"
  echo: false

monitoring:
  prometheus:
    enabled: true
  jaeger:
    enabled: true
```

## Troubleshooting

### Common Configuration Issues

1. **Missing Environment Variables**: Check `.env` file and system environment
2. **Invalid YAML Syntax**: Use YAML linter to validate syntax
3. **Type Mismatches**: Ensure values match expected types
4. **Network Connectivity**: Verify database and service URLs
5. **Permission Issues**: Check file and directory permissions

### Debug Commands

```bash
# Check environment variables
env | grep -E "(DATABASE|REDIS|NEO4J)"

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/database.yaml'))"

# Test database connection
python -c "from Medical_KG_rev.storage.database import DatabaseManager; db = DatabaseManager(); print(db.test_connection())"

# Check configuration loading
python -c "from Medical_KG_rev.config import Config; config = Config(); print(config.database.url)"
```

## Best Practices

1. **Use Environment Variables**: For all sensitive configuration
2. **Validate Early**: Check configuration at startup
3. **Fail Fast**: Exit immediately on configuration errors
4. **Document Changes**: Update this reference when adding new options
5. **Test Configurations**: Validate in development before production
6. **Version Control**: Keep configuration files in version control (excluding secrets)
7. **Backup Configurations**: Maintain backups of production configurations
8. **Monitor Changes**: Log configuration changes for audit purposes

## Related Documentation

- [Environment Setup Guide](environment_setup.md)
- [Deployment Guide](deployment.md)
- [Security Configuration](security.md)
- [Monitoring Configuration](monitoring.md)
- [Troubleshooting Guide](troubleshooting.md)
