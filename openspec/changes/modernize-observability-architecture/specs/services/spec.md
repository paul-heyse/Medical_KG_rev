## ADDED Requirements

### Requirement: Torch-Isolated GPU Service Architecture

All GPU-dependent services SHALL communicate via gRPC to isolated GPU microservices instead of loading models in-process.

#### Scenario: Qwen3 Service gRPC Integration

- **WHEN** the Qwen3 service processes embedding requests
- **THEN** it SHALL use gRPC client to communicate with GPU microservices
- **AND** SHALL NOT load Hugging Face models in the main process
- **AND** SHALL handle gRPC service discovery and connection management

#### Scenario: GPU Microservice Interface

- **GIVEN** torch-isolated architecture requirements
- **WHEN** GPU microservices are deployed
- **THEN** they SHALL expose gRPC interfaces for embedding operations
- **AND** SHALL implement health checking and readiness probes
- **AND** SHALL support batch processing and resource management

#### Scenario: Circuit Breaker Integration for GPU Services

- **WHEN** GPU microservices experience failures
- **THEN** gRPC clients SHALL implement circuit breaker patterns
- **AND** SHALL provide fallback behavior for service unavailability
- **AND** SHALL report circuit breaker state for monitoring

### Requirement: GPU Service Resilience

GPU service clients SHALL implement comprehensive resilience patterns for production reliability.

#### Scenario: Service Discovery and Load Balancing

- **WHEN** multiple GPU microservice instances are available
- **THEN** clients SHALL implement service discovery and load balancing
- **AND** SHALL distribute requests across healthy instances
- **AND** SHALL handle instance failures gracefully

#### Scenario: Resource Management and Quotas

- **WHEN** GPU resources are constrained
- **THEN** service clients SHALL implement resource quotas and limits
- **AND** SHALL queue requests when resources are unavailable
- **AND** SHALL provide resource usage monitoring and reporting

## MODIFIED Requirements

### Requirement: Embedding Service Interface

Embedding services SHALL provide a consistent interface for text embedding operations.

#### Scenario: Service Implementation Flexibility

- **GIVEN** torch-isolated architecture requirements
- **WHEN** embedding services are implemented
- **THEN** they SHALL support both in-process and gRPC-based implementations
- **AND** SHALL maintain consistent API contracts across implementations
- **AND** SHALL provide configuration for service endpoint selection

### Requirement: Service Health Monitoring

All services SHALL implement comprehensive health checking and monitoring.

#### Scenario: GPU Service Health Validation

- **GIVEN** gRPC-based GPU service integration
- **WHEN** health checks are performed
- **THEN** services SHALL validate gRPC connectivity and service readiness
- **AND** SHALL report GPU resource availability and utilization
- **AND** SHALL include circuit breaker state in health reports

## REMOVED Requirements

### Requirement: In-Process Model Loading (Deprecated)

**Reason**: Violates torch isolation architecture by loading models in main process
**Migration**: Replace with gRPC calls to GPU microservices

### Requirement: Hugging Face Model Management (Deprecated)

**Reason**: In-process model management conflicts with torch isolation requirements
**Migration**: Move model management to GPU microservice containers
