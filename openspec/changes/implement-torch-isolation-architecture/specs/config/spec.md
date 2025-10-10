## ADDED Requirements

### Requirement: Torch-Free Gateway Configuration

The system SHALL provide configuration management for torch-free main API gateway with service endpoint configuration and circuit breaker settings.

#### Scenario: Torch-free gateway configuration initialization

- **WHEN** the torch-free gateway starts up
- **THEN** it SHALL load configuration without torch dependencies
- **AND** SHALL configure service client endpoints for GPU services
- **AND** SHALL set up circuit breaker configuration
- **AND** SHALL validate service availability before accepting requests

#### Scenario: Service endpoint configuration

- **WHEN** configuring service endpoints
- **THEN** the system SHALL support configuration of GPU service URLs
- **AND** SHALL support configuration of embedding service URLs
- **AND** SHALL support configuration of reranking service URLs
- **AND** SHALL provide service discovery and load balancing configuration

#### Scenario: Circuit breaker configuration

- **WHEN** configuring circuit breaker settings
- **THEN** the system SHALL support failure threshold configuration
- **AND** SHALL support reset timeout configuration
- **AND** SHALL support retry strategy configuration
- **AND** SHALL provide circuit breaker state persistence

### Requirement: GPU Services Configuration

The system SHALL provide configuration management for GPU services Docker containers with torch ecosystem settings.

#### Scenario: GPU services configuration initialization

- **WHEN** GPU services containers start
- **THEN** they SHALL initialize torch and CUDA environment
- **AND** SHALL configure GPU memory allocation and management
- **AND** SHALL set up model loading and caching configuration
- **AND** SHALL configure health check and monitoring settings

#### Scenario: GPU resource allocation configuration

- **WHEN** configuring GPU resources
- **THEN** the system SHALL support GPU memory allocation settings
- **AND** SHALL support GPU device selection configuration
- **AND** SHALL support GPU utilization monitoring configuration
- **AND** SHALL provide GPU resource optimization settings

#### Scenario: Model configuration for GPU services

- **WHEN** configuring models for GPU services
- **THEN** the system SHALL support model loading and caching settings
- **AND** SHALL support model version and update configuration
- **AND** SHALL support model performance optimization settings
- **AND** SHALL provide model health monitoring configuration

### Requirement: Service Client Configuration

The system SHALL provide configuration management for service client communication with circuit breaker and retry settings.

#### Scenario: Service client configuration initialization

- **WHEN** service clients initialize
- **THEN** they SHALL configure connection pooling settings
- **AND** SHALL configure timeout and retry settings
- **AND** SHALL configure circuit breaker settings
- **AND** SHALL validate service endpoint availability

#### Scenario: Circuit breaker configuration

- **WHEN** configuring circuit breaker settings
- **THEN** the system SHALL support failure threshold configuration
- **AND** SHALL support reset timeout configuration
- **AND** SHALL support retry strategy configuration
- **AND** SHALL provide circuit breaker state monitoring

#### Scenario: Service communication configuration

- **WHEN** configuring service communication
- **THEN** the system SHALL support timeout configuration
- **AND** SHALL support retry policy configuration
- **AND** SHALL support connection pooling configuration
- **AND** SHALL provide load balancing configuration

## MODIFIED Requirements

### Requirement: Main Application Configuration

The main application configuration SHALL support torch-free gateway operation with service endpoint configuration.

#### Scenario: Torch-free configuration loading

- **WHEN** the application loads configuration
- **THEN** it SHALL support torch-free gateway configuration
- **AND** SHALL configure service client endpoints
- **AND** SHALL set up circuit breaker configuration
- **AND** SHALL validate service availability

#### Scenario: Service endpoint configuration

- **WHEN** configuring service endpoints
- **THEN** the system SHALL support GPU service URL configuration
- **AND** SHALL support embedding service URL configuration
- **AND** SHALL support reranking service URL configuration
- **AND** SHALL provide service discovery configuration

### Requirement: GPU Configuration

The GPU configuration SHALL support Docker service deployment with torch ecosystem settings.

#### Scenario: GPU services configuration

- **WHEN** configuring GPU services
- **THEN** the system SHALL support GPU memory allocation
- **AND** SHALL support GPU device selection
- **AND** SHALL support GPU utilization monitoring
- **AND** SHALL provide GPU resource optimization

#### Scenario: Model configuration for GPU services

- **WHEN** configuring models for GPU services
- **THEN** the system SHALL support model loading configuration
- **AND** SHALL support model caching configuration
- **AND** SHALL support model performance configuration
- **AND** SHALL provide model health monitoring

## REMOVED Requirements

### Requirement: Torch Dependencies in Main Configuration

**Reason**: Eliminated torch dependencies from main application configuration
**Migration**: Torch configuration moved to GPU services Docker containers

The main application configuration no longer requires torch dependencies, instead configuring service endpoints for GPU operations.

### Requirement: Direct GPU Configuration in Main Application

**Reason**: Replaced by service-based GPU configuration for torch isolation
**Migration**: GPU configuration moved to dedicated GPU services Docker containers

The main application no longer configures direct GPU operations, instead configuring service endpoints for GPU functionality.
