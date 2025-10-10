## 1. Torch-Free Chunking Implementation

- [ ] 1.1 Remove torch-based semantic checks from chunking
      - Remove GPU semantic validation from `src/Medical_KG_rev/chunking/base.py`
      - Remove torch imports from chunking modules
      - Update chunking interfaces to use Docling's built-in chunking capabilities
      - Remove `gpu_semantic_checks` parameter from chunking functions
      - Update chunking tests to not require GPU validation

- [ ] 1.2 Integrate Docling chunking capabilities
      - Update `src/Medical_KG_rev/chunking/` to use Docling's chunking output
      - Implement Docling document structure parsing for chunking
      - Remove torch-based token counting fallbacks
      - Update chunking configuration to leverage Docling capabilities
      - Add chunking validation for Docling output

- [ ] 1.3 Update chunking tests for torch-free operation
      - Modify `tests/chunking/` to work without torch dependencies
      - Update test fixtures to use Docling chunking output
      - Remove GPU-related test assertions
      - Add tests for Docling integration
      - Ensure all chunking tests pass without torch

- [ ] 1.4 Implement chunking interface compatibility
      - Ensure chunking interface remains compatible with existing systems
      - Update chunking configuration to support both torch-free and legacy modes
      - Add fallback mechanisms for torch-based chunking if needed
      - Implement chunking result validation and quality checks
      - Add chunking performance monitoring without torch dependencies

## 2. GPU Services Docker Container Creation

- [ ] 2.1 Create GPU services base Docker image
      - Create `ops/docker/gpu-services/Dockerfile` with torch ecosystem
      - Install torch, torchvision, transformers, accelerate
      - Configure CUDA 12.1+ support for GPU processing
      - Set up model caching directories
      - Include health check endpoints

- [ ] 2.2 Move GPU manager to Docker service
      - Extract `src/Medical_KG_rev/services/gpu/` to Docker service
      - Create gRPC service definition (proto/gpu.proto) for GPU management
      - Implement gRPC service methods: GetStatus, ListDevices, AllocateGPU
      - Add GPU memory monitoring and alerting
      - Include circuit breaker for network failures (not GPU unavailability)

- [ ] 2.3 Move embedding GPU utilities to Docker service
      - Extract `src/Medical_KG_rev/embeddings/utils/gpu.py` to Docker service
      - Extend gRPC service definition for embedding GPU operations
      - Implement embedding model loading and caching
      - Add GPU memory management for embedding operations
      - Include gRPC health check protocol for service monitoring

- [ ] 2.4 Move vector store GPU functionality to Docker service
      - Extract `src/Medical_KG_rev/services/vector_store/gpu.py` to Docker service
      - Extend gRPC service definition for vector store GPU operations
      - Implement GPU-accelerated similarity search
      - Add vector store GPU statistics and monitoring
      - Include gRPC health check protocol

- [ ] 2.5 Create GPU services gRPC server entry point
      - Create `src/Medical_KG_rev/services/gpu/grpc_server.py` for GPU service
      - Implement gRPC server with GPU service implementation
      - Add interceptors for error handling and OpenTelemetry tracing
      - Include GPU service configuration and initialization with fail-fast GPU checks
      - Add graceful shutdown handling for GPU service
      - Configure gRPC health check service

## 3. Embedding Services Docker Container

- [ ] 3.1 Create embedding services Docker image
      - Create `ops/docker/embedding-services/Dockerfile` with torch and transformers
      - Install torch, transformers, sentence-transformers for embedding generation
      - Configure GPU memory allocation for embedding models
      - Set up model download and caching
      - Include embedding service health monitoring

- [ ] 3.2 Implement embedding service gRPC API
      - Create `src/Medical_KG_rev/services/embedding/grpc_service.py`
      - Define proto for embedding service (proto/embedding_service.proto)
      - Implement GenerateEmbeddings RPC method for batch embedding
      - Add ListModels RPC method for available models
      - Implement gRPC health check protocol
      - Add request/response validation and error handling with proper error codes

- [ ] 3.3 Update embedding service configuration
      - Create `src/Medical_KG_rev/config/embedding_docker_config.py`
      - Add service URL, timeout, retry configuration
      - Include model selection and GPU allocation settings
      - Add circuit breaker configuration for service resilience
      - Implement configuration validation and hot-reloading

- [ ] 3.4 Create embedding services gRPC server entry point
      - Create `src/Medical_KG_rev/services/embedding/grpc_server.py` for embedding service
      - Implement gRPC server with embedding service implementation
      - Add interceptors for OpenTelemetry tracing and error handling
      - Include embedding service configuration and initialization
      - Add graceful shutdown handling for embedding service
      - Configure gRPC health check service

## 4. Reranking Services Docker Container

- [ ] 4.1 Create reranking services Docker image
      - Create `ops/docker/reranking-services/Dockerfile` with torch ecosystem
      - Install torch, transformers for cross-encoder reranking
      - Configure GPU memory for reranking models
      - Set up model loading and caching
      - Include reranking service health monitoring

- [ ] 4.2 Implement reranking service gRPC API
      - Create `src/Medical_KG_rev/services/reranking/grpc_service.py`
      - Define proto for reranking service (proto/reranking_service.proto)
      - Implement RerankBatch RPC method for batch reranking
      - Add ListModels RPC method for available reranking models
      - Implement gRPC health check protocol
      - Add request/response validation and error handling with proper error codes

- [ ] 4.3 Update reranking service configuration
      - Create `src/Medical_KG_rev/config/reranking_docker_config.py`
      - Add service URL, timeout, batch size configuration
      - Include model selection and GPU allocation settings
      - Add circuit breaker configuration for service resilience
      - Implement configuration validation and hot-reloading

- [ ] 4.4 Create reranking services gRPC server entry point
      - Create `src/Medical_KG_rev/services/reranking/grpc_server.py` for reranking service
      - Implement gRPC server with reranking service implementation
      - Add interceptors for OpenTelemetry tracing and error handling
      - Include reranking service configuration and initialization
      - Add graceful shutdown handling for reranking service
      - Configure gRPC health check service

## 5. Main Gateway Torch-Free Implementation

- [ ] 5.1 Remove all torch imports from main gateway
      - Remove torch imports from `src/Medical_KG_rev/gateway/`
      - Remove torch imports from `src/Medical_KG_rev/services/`
      - Remove torch imports from `src/Medical_KG_rev/observability/`
      - Update all code to use HTTP API calls instead of direct torch usage
      - Remove torch-based GPU checks and fallbacks

- [ ] 5.2 Implement gRPC service client architecture
      - Create `src/Medical_KG_rev/services/clients/` for gRPC service clients
      - Implement `GPUClient` using gRPC stubs for GPU service communication
      - Implement `EmbeddingClient` using gRPC stubs for embedding service communication
      - Implement `RerankingClient` using gRPC stubs for reranking service communication
      - Add service discovery with gRPC health check protocol
      - Include OpenTelemetry trace context propagation in gRPC metadata

- [ ] 5.3 Update configuration for service endpoints
      - Create `src/Medical_KG_rev/config/service_clients_config.py`
      - Add GPU service URL and configuration
      - Add embedding service URL and configuration
      - Add reranking service URL and configuration
      - Include circuit breaker and retry settings

- [ ] 5.4 Implement circuit breaker patterns for network failures only
      - Create `src/Medical_KG_rev/services/clients/circuit_breaker.py`
      - Implement circuit breaker for GPU service network failures (NOT GPU unavailability)
      - Implement circuit breaker for embedding service network failures
      - Implement circuit breaker for reranking service network failures
      - Distinguish gRPC UNAVAILABLE (circuit breaker) from FAILED_PRECONDITION (fail-fast)
      - Add fallback mechanisms for network issues, fail-fast for GPU issues

- [ ] 5.5 Update gateway to use gRPC service clients
      - Replace direct torch calls with gRPC service client calls in gateway
      - Update embedding generation to use embedding gRPC client
      - Update reranking to use reranking gRPC client
      - Update GPU operations to use GPU gRPC client
      - Add error handling distinguishing network failures from GPU unavailability
      - Propagate tenant_id in gRPC metadata for all service calls

## 6. Service Integration and Communication

- [ ] 6.1 Implement service registry and discovery
      - Create `src/Medical_KG_rev/services/registry.py`
      - Implement service discovery for GPU, embedding, reranking services
      - Add health checking for service availability
      - Include service load balancing and failover
      - Add service registration and deregistration

- [ ] 6.2 Update all torch usage to gRPC service calls
      - Update `src/Medical_KG_rev/gateway/` to use gRPC service clients
      - Update `src/Medical_KG_rev/services/` to use gRPC service clients
      - Update `src/Medical_KG_rev/observability/` to use gRPC service clients
      - Replace all direct torch calls with gRPC API calls
      - Add error handling for service communication failures with proper error classification

- [ ] 6.3 Implement service communication error handling
      - Create `src/Medical_KG_rev/services/clients/errors.py`
      - Implement service unavailable error handling
      - Add timeout error handling for service calls
      - Include retry logic for transient failures
      - Add comprehensive error logging and monitoring

- [ ] 6.4 Implement service client testing
      - Create `tests/services/clients/test_gpu_client.py`
      - Create `tests/services/clients/test_embedding_client.py`
      - Create `tests/services/clients/test_reranking_client.py`
      - Test service client communication and error handling
      - Test circuit breaker functionality and fallback mechanisms

## 7. Docker Configuration and Deployment

- [ ] 7.1 Create torch-free main gateway Docker image
      - Create `ops/docker/gateway/Dockerfile` without torch dependencies
      - Use multi-stage build for minimal image size
      - Include only necessary dependencies for torch-free operation
      - Add health check endpoint for gateway service
      - Configure environment variables for service URLs

- [ ] 7.2 Create docker-compose configuration for all services
      - Create `ops/docker/docker-compose.torch-isolation.yml`
      - Define GPU services, embedding services, reranking services with gRPC ports
      - Configure service networking and communication (Docker DNS, gRPC ports)
      - Add service dependencies and startup order
      - Include volume mounts for model caching and mTLS certificates

- [ ] 7.3 Update Kubernetes deployment configurations
      - Create `ops/k8s/gateway-deployment-torch-free.yaml`
      - Create `ops/k8s/gpu-services-deployment.yaml`
      - Create `ops/k8s/embedding-services-deployment.yaml`
      - Create `ops/k8s/reranking-services-deployment.yaml`
      - Add service mesh configuration for communication

- [ ] 7.4 Create Docker service health checks using gRPC health protocol
      - Implement gRPC health check service in all Docker services
      - Add health check validation for torch availability
      - Add GPU memory availability checks for GPU services (fail if unavailable)
      - Include model loading validation for embedding and reranking services
      - Use grpc_health_probe for Docker HEALTHCHECK commands
      - Add comprehensive health monitoring and alerting

## 8. Testing and Validation

- [ ] 8.1 Create service integration tests
      - Create `tests/services/integration/test_gpu_service.py`
      - Create `tests/services/integration/test_embedding_service.py`
      - Create `tests/services/integration/test_reranking_service.py`
      - Test end-to-end service communication
      - Validate service responses match direct torch usage

- [ ] 8.2 Update existing tests for torch-free operation
      - Update `tests/chunking/` to not require torch
      - Update `tests/embeddings/` to use Docker service
      - Update `tests/reranking/` to use Docker service
      - Update `tests/gateway/` to use service clients
      - Ensure all tests pass without torch dependencies

- [ ] 8.3 Create performance tests for service communication
      - Create `tests/performance/test_service_latency.py`
      - Test service response times vs direct torch usage
      - Validate performance requirements are met
      - Add load testing for concurrent service calls
      - Monitor service resource usage and scaling

- [ ] 8.4 Create integration tests for complete torch isolation
      - Create `tests/integration/test_torch_isolation.py`
      - Test end-to-end torch-free gateway operation
      - Validate GPU services provide equivalent functionality
      - Test service failover and circuit breaker behavior
      - Ensure no torch dependencies in main gateway

- [ ] 8.5 Create Docker service tests
      - Create `tests/docker/test_gpu_services.py`
      - Create `tests/docker/test_embedding_services.py`
      - Create `tests/docker/test_reranking_services.py`
      - Test Docker service functionality and health checks
      - Validate Docker service API endpoints and responses

## 9. Monitoring and Observability

- [ ] 9.1 Update monitoring for service architecture
      - Update `src/Medical_KG_rev/observability/metrics.py` for service metrics
      - Add service call latency and error rate metrics
      - Add service availability and health metrics
      - Include circuit breaker state monitoring
      - Add service resource utilization metrics

- [ ] 9.2 Update Grafana dashboards for service monitoring
      - Create `docs/guides/monitoring/service_architecture_dashboard.json`
      - Add service health and availability panels
      - Include service response time and error rate charts
      - Add circuit breaker status and service discovery metrics
      - Update existing dashboards for service architecture

- [ ] 9.3 Implement service health monitoring
      - Create `src/Medical_KG_rev/services/monitoring/service_health.py`
      - Monitor GPU service health and availability
      - Monitor embedding service health and model loading
      - Monitor reranking service health and GPU utilization
      - Add alerting for service failures and performance issues
      - Include service dependency health checks

- [ ] 9.4 Implement service performance monitoring
      - Create `src/Medical_KG_rev/services/monitoring/service_performance.py`
      - Monitor service response times and throughput
      - Track service resource utilization (CPU, memory, GPU)
      - Add performance degradation detection and alerting
      - Include service performance optimization recommendations
      - Add service performance benchmarking and comparison

## 10. Documentation and Migration

- [ ] 10.1 Update architecture documentation
      - Update `docs/architecture/overview.md` with torch isolation architecture
      - Add service architecture diagrams and communication flows
      - Document Docker service interfaces and APIs
      - Update deployment guides for torch-free operation
      - Add troubleshooting guides for service communication

- [ ] 10.2 Create migration guide for torch isolation
      - Create `docs/guides/torch_isolation_migration.md`
      - Document step-by-step migration process
      - Include before/after configuration examples
      - Add troubleshooting section for common issues
      - Provide rollback procedures if needed

- [ ] 10.3 Update operational runbooks
      - Update `docs/operational-runbook.md` for service architecture
      - Add service deployment and scaling procedures
      - Include service health check and troubleshooting
      - Add service update and rollback procedures
      - Update monitoring and alerting procedures

- [ ] 10.4 Update developer documentation
      - Update `docs/guides/developer_guide.md` with service architecture
      - Add service client usage examples
      - Include Docker service development setup
      - Update testing guidelines for service architecture
      - Add debugging guides for service communication issues

- [ ] 10.5 Create API documentation for service endpoints
      - Create `docs/api/service_architecture.md` for service APIs
      - Document GPU service API endpoints and usage
      - Document embedding service API endpoints and usage
      - Document reranking service API endpoints and usage
      - Include service client examples and best practices

## 11. Dagster Integration

- [ ] 11.1 Update Dagster assets to use gRPC service clients
      - Update `src/Medical_KG_rev/orchestration/dagster/assets/` to use gRPC clients
      - Replace direct torch usage with GPU service calls in Dagster assets
      - Add error handling for GPU unavailability (fail entire Dagster run)
      - Distinguish network failures (retry) from GPU failures (fail-fast)
      - Propagate tenant_id through Dagster asset execution context

- [ ] 11.2 Create Dagster resources for GPU services
      - Create `src/Medical_KG_rev/orchestration/dagster/resources/gpu_services.py`
      - Implement Dagster resource providing gRPC clients (GPU, embedding, reranking)
      - Add resource configuration for service addresses and authentication
      - Include circuit breaker configuration in Dagster resources
      - Add OpenTelemetry trace context propagation from Dagster to services

- [ ] 11.3 Update Dagster job configurations
      - Update `src/Medical_KG_rev/orchestration/dagster/jobs/` to use GPU service resources
      - Configure service endpoints for different environments (dev, staging, prod)
      - Add mTLS certificate configuration for Dagster workers
      - Update two-phase pipeline jobs to use gRPC service clients
      - Add job-level error handling for GPU service unavailability

- [ ] 11.4 Create Dagster integration tests
      - Create `tests/orchestration/dagster/test_gpu_service_integration.py`
      - Test Dagster assets calling GPU services via gRPC
      - Test fail-fast behavior when GPU unavailable in Dagster runs
      - Test circuit breaker behavior for network failures in Dagster
      - Validate OpenTelemetry tracing through Dagster to GPU services

- [ ] 11.5 Update Dagster monitoring and observability
      - Add Dagster sensors for GPU service health monitoring
      - Include GPU service metrics in Dagster run metadata
      - Add alerting for GPU service failures in Dagster runs
      - Update Dagster logging to include gRPC service call details
      - Add Dagster dashboard panels for GPU service utilization

## 12. Security and Compliance

- [ ] 12.1 Implement service-to-service authentication with mTLS
      - Generate CA certificate and key pairs for services
      - Create service certificates for GPU, embedding, reranking services
      - Create client certificates for gateway and Dagster workers
      - Configure gRPC servers to require client certificates (mTLS)
      - Configure gRPC clients with client certificates
      - Add certificate rotation procedures and documentation

- [ ] 12.2 Review service architecture for security implications
      - Conduct security assessment of gRPC service-to-service communication
      - Review GPU service data handling and memory management
      - Verify mTLS authentication implementation
      - Document security findings and mitigation strategies

- [ ] 12.3 Ensure service architecture maintains compliance
      - Review service processing for PHI data handling compliance
      - Verify data encryption in transit via mTLS for gRPC calls
      - Update compliance documentation for service architecture
      - Add HIPAA compliance checklist for service deployment
      - Document data retention policies for service artifacts

- [ ] 12.4 Update audit logging for service operations
      - Modify `src/Medical_KG_rev/auth/audit.py` for gRPC service call logging
      - Add audit events for service communication and failures
      - Include service health and performance in audit logs
      - Include tenant_id from gRPC metadata in all audit logs
      - Update audit log retention policies for service data
      - Add compliance reporting for service activities

- [ ] 12.5 Validate gRPC service security measures
      - Verify mTLS authentication for all service-to-service communication
      - Ensure gRPC connections use TLS 1.3+
      - Add request validation and sanitization for gRPC service APIs
      - Include rate limiting for gRPC service endpoints
      - Validate security error handling (avoid leaking sensitive info in error messages)

## 13. Performance Optimization

- [ ] 13.1 Optimize gRPC service communication performance
      - Implement connection pooling for service clients
      - Add request batching for efficient service calls
      - Implement caching for repeated service requests
      - Add service call timeout and retry optimization
      - Monitor and optimize service response times

- [ ] 13.2 Implement service auto-scaling
      - Add auto-scaling configuration for GPU services
      - Implement load-based scaling for embedding services
      - Add performance-based scaling for reranking services
      - Include scaling metrics and thresholds
      - Add scaling policy validation and monitoring

- [ ] 13.3 Optimize GPU resource utilization
      - Implement GPU memory management across services
      - Add model caching and sharing between services
      - Implement GPU resource allocation and scheduling
      - Add GPU utilization monitoring and alerting
      - Include GPU resource optimization procedures

- [ ] 13.4 Implement service caching strategies
      - Add Redis-based caching for service responses
      - Implement cache invalidation for service updates
      - Add cache warming for frequently used models
      - Include cache performance monitoring and optimization
      - Add cache size and TTL configuration

## 14. Deployment and Rollout

- [ ] 14.1 Create deployment automation for torch isolation
      - Create `scripts/deploy_torch_isolation.py` deployment script
      - Implement blue-green deployment for service architecture
      - Add automated service discovery and configuration
      - Include deployment validation and health checks
      - Add rollback procedures for deployment failures

- [ ] 14.2 Update CI/CD pipelines for service architecture
      - Modify `.github/workflows/ci-cd.yml` for service architecture
      - Add Docker service image building and testing
      - Include service integration testing in CI pipeline
      - Add deployment automation for service architecture
      - Configure artifact storage for service images

- [ ] 14.3 Implement service deployment validation
      - Create `scripts/validate_torch_isolation_deployment.py`
      - Validate torch-free main gateway operation
      - Verify GPU service functionality and performance
      - Test service communication and error handling
      - Include deployment success/failure reporting
      - Add automated rollback for deployment issues

- [ ] 14.4 Create deployment configuration management
      - Create `ops/config/torch_isolation_config.py` for deployment settings
      - Add environment-specific configuration for different deployment targets
      - Include service endpoint configuration for different environments
      - Add deployment validation and health check configuration
      - Include rollback configuration and procedures

## 15. Legacy Code Cleanup

- [ ] 15.1 Remove torch dependencies from main codebase
      - Remove torch imports from all main gateway files
      - Remove torch-based GPU checks and fallbacks
      - Remove torch-related configuration options
      - Update requirements.txt to exclude torch packages
      - Clean up torch-related test fixtures and mocks

- [ ] 15.2 Update documentation for torch-free architecture
      - Update all documentation to reflect torch-free main gateway
      - Remove references to torch dependencies in main codebase
      - Update architecture diagrams for service architecture
      - Add service architecture to API documentation
      - Update deployment guides for torch-free operation

- [ ] 15.3 Archive torch-dependent code
      - Move torch-dependent code to `src/Medical_KG_rev/legacy/` directory
      - Create migration notes for torch-dependent functionality
      - Add deprecation warnings for torch-dependent features
      - Document torch dependency removal in changelog
      - Update code comments to reflect service architecture

- [ ] 15.4 Update requirements and dependencies
      - Remove torch, torchvision, transformers from main requirements.txt
      - Verify gRPC dependencies already present (grpcio, grpcio-tools, grpcio-health-checking)
      - Add circuit breaker and service discovery dependencies
      - Update Docker requirements for GPU services
      - Add OpenTelemetry dependencies for distributed tracing

## 16. Validation and Quality Assurance

- [ ] 16.1 Create comprehensive validation tests
      - Create `tests/validation/test_torch_isolation_completeness.py`
      - Validate no torch imports in main gateway code
      - Test all torch functionality moved to Docker services
      - Validate service API compatibility and functionality
      - Ensure no torch dependencies in production Docker images

- [ ] 16.2 Implement quality gates for torch isolation
      - Add CI checks to prevent torch imports in main gateway
      - Implement automated validation of torch-free deployment
      - Add service integration validation in deployment pipeline
      - Include performance regression testing for service architecture
      - Add security validation for service communication

- [ ] 16.3 Create acceptance tests for torch isolation
      - Create `tests/acceptance/test_torch_free_gateway.py`
      - Test end-to-end torch-free gateway functionality
      - Validate GPU services provide equivalent functionality
      - Test service failover and resilience mechanisms
      - Include performance and security acceptance criteria

- [ ] 16.4 Implement monitoring validation
      - Create `tests/monitoring/test_service_monitoring.py`
      - Validate service health monitoring functionality
      - Test service performance monitoring and alerting
      - Validate circuit breaker state monitoring
      - Include monitoring data validation and reporting
