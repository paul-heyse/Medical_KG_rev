## ADDED Requirements

### Requirement: Docling VLM Configuration Management
The system SHALL provide configuration management for Docling VLM processing including model settings, GPU resource allocation, and feature flag controls.

#### Scenario: Docling configuration initialization
- **WHEN** the application starts
- **THEN** it SHALL load Docling VLM configuration from environment variables and config files
- **AND** SHALL validate Gemma3 12B model availability and GPU requirements
- **AND** SHALL set up model caching and download directories
- **AND** SHALL initialize feature flag for Docling vs MinerU processing modes

#### Scenario: GPU resource configuration for VLM
- **WHEN** configuring Docling VLM processing
- **THEN** the system SHALL allow configuration of GPU memory allocation for Gemma3 12B
- **AND** SHALL support batch size configuration for optimal throughput
- **AND** SHALL provide model warm-up settings for consistent performance
- **AND** SHALL include timeout and retry configuration for VLM operations

#### Scenario: Feature flag management for migration
- **WHEN** managing the Docling VLM migration
- **THEN** the system SHALL provide feature flags to control processing backend selection
- **AND** SHALL support gradual rollout percentages (0-100%)
- **AND** SHALL allow backend comparison and performance monitoring
- **AND** SHALL enable rollback to MinerU processing if needed

### Requirement: Retrieval Configuration Management
The system SHALL provide configuration management for hybrid retrieval system including BM25, SPLADE, and Qwen3 settings.

#### Scenario: Retrieval backend configuration
- **WHEN** configuring retrieval strategies
- **THEN** the system SHALL allow selection of BM25, SPLADE, Qwen3, or hybrid modes
- **AND** SHALL support feature flag control for gradual migration
- **AND** SHALL provide method-specific configuration options
- **AND** SHALL validate tokenizer alignment and model compatibility

#### Scenario: Index configuration management
- **WHEN** configuring retrieval indexes
- **THEN** the system SHALL support BM25 field boosts and analyzer settings
- **AND** SHALL allow SPLADE sparsity thresholds and quantization parameters
- **AND** SHALL provide Qwen3 embedding dimensions and storage backend selection
- **AND** SHALL include manifest-based version tracking for indexes

#### Scenario: Performance configuration
- **WHEN** optimizing retrieval performance
- **THEN** the system SHALL support batch size configuration for each strategy
- **AND** SHALL allow caching configuration for repeated queries
- **AND** SHALL provide fusion ranking parameters for hybrid retrieval
- **AND** SHALL include query timeout and retry settings

## MODIFIED Requirements

### Requirement: PDF Processing Configuration
The PDF processing configuration SHALL support both MinerU and Docling VLM backends with unified configuration interface.

#### Scenario: Backend-agnostic configuration
- **WHEN** configuring PDF processing settings
- **THEN** the configuration system SHALL abstract backend-specific settings
- **AND** SHALL provide common interface for batch sizes, timeouts, and resource allocation
- **AND** SHALL maintain backward compatibility with existing MinerU configurations
- **AND** SHALL validate configuration changes before applying them

#### Scenario: Environment-based configuration
- **WHEN** deploying in different environments
- **THEN** the Docling VLM configuration SHALL support environment-specific overrides
- **AND** SHALL provide sensible defaults for development, staging, and production
- **AND** SHALL include security settings for model access and data handling
- **AND** SHALL support configuration validation and health checks

### Requirement: Retrieval Configuration
The retrieval configuration SHALL support hybrid retrieval combining multiple strategies with unified configuration interface.

#### Scenario: Strategy configuration
- **WHEN** configuring retrieval strategies
- **THEN** the configuration system SHALL support BM25, SPLADE, Qwen3, and hybrid modes
- **AND** SHALL provide feature flag control for gradual migration
- **AND** SHALL maintain backward compatibility with existing retrieval APIs
- **AND** SHALL validate configuration changes before applying them

#### Scenario: Performance configuration
- **WHEN** optimizing retrieval performance
- **THEN** the configuration system SHALL support batch sizes and caching for each strategy
- **AND** SHALL allow fusion ranking parameter configuration
- **AND** SHALL provide query timeout and retry settings
- **AND** SHALL include performance monitoring configuration

## REMOVED Requirements

### Requirement: MinerU-Specific Configuration
**Reason**: Replaced by Docling VLM configuration
**Migration**: Existing MinerU configurations remain valid; Docling configurations added alongside

The system no longer requires MinerU-specific configuration parameters when Docling VLM mode is active.

### Requirement: Single Retrieval Strategy Configuration
**Reason**: Replaced by hybrid retrieval configuration
**Migration**: Hybrid retrieval maintains backward compatibility while providing improved configuration options

The system no longer limits retrieval configuration to a single strategy but supports multiple complementary approaches.
