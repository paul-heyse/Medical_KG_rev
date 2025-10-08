## 1. Design & Planning

- [ ] 1.1 Analyze current biomedical adapter implementations and identify common patterns
- [ ] 1.2 Design shared mixin architecture for common behaviors
- [x] 1.3 Plan package structure for individual adapter modules
- [ ] 1.4 Design standardized interfaces for metadata extraction and PDF URLs
- [ ] 1.5 Plan integration strategy with existing plugin manager
- [ ] 1.6 Analyze adapter-specific requirements and data formats
- [ ] 1.7 Design adapter dependency management and chaining
- [ ] 1.8 Plan adapter performance characteristics and optimization
- [ ] 1.9 Design adapter error recovery and retry strategies
- [ ] 1.10 Plan adapter security model and access control

### Critical Library Integration Requirements

- [x] 1.11 **Integrate `pyalex>=0.1.0`**: Design OpenAlex adapter using official pyalex client for PDF retrieval
- [ ] 1.12 **Integrate `httpx>=0.28.1`**: Replace legacy HTTP clients with modern async httpx for all adapters
- [ ] 1.13 **Integrate `tenacity>=9.1.2`**: Add retry logic with exponential backoff for external API calls
- [ ] 1.14 **Integrate `pydantic>=2.11.10`**: Design typed adapter request/response models with validation
- [ ] 1.15 **Integrate `pluggy>=1.6.0`**: Ensure adapter decomposition works with existing plugin system
- [ ] 1.16 **Integrate `aiohttp>=3.12.15`**: Add async HTTP client support for high-throughput adapters
- [ ] 1.17 **Integrate `aiolimiter>=1.2.1`**: Implement rate limiting for external API compliance

## 2. Shared Infrastructure

- [ ] 2.1 Create HTTPAdapterMixin for common HTTP operations
- [ ] 2.2 Create PaginationMixin for standardized pagination handling
- [ ] 2.3 Create DOINormalizationMixin for DOI standardization
- [ ] 2.4 Create OALinkResolutionMixin for open access link handling
- [ ] 2.5 Create MetadataExtractionMixin for consistent metadata shaping
- [ ] 2.6 Create RateLimitMixin for API quota management
- [ ] 2.7 Create RetryMixin for resilient API interactions
- [ ] 2.8 Create ValidationMixin for data integrity checks
- [ ] 2.9 Create LoggingMixin for standardized adapter logging
- [ ] 2.10 Create MetricsMixin for adapter performance monitoring

## 3. Base Adapter Classes

- [ ] 3.1 Create BiomedicalAdapterBase with common functionality
- [ ] 3.2 Define standardized adapter interface methods
- [ ] 3.3 Add validation and error handling patterns
- [ ] 3.4 Create configuration schema for adapter settings
- [ ] 3.5 Add adapter capability reporting interface
- [ ] 3.6 Implement adapter lifecycle management (init, cleanup)
- [ ] 3.7 Add adapter health checking and monitoring
- [ ] 3.8 Create adapter caching strategies for performance
- [ ] 3.9 Implement adapter circuit breaker patterns
- [ ] 3.10 Add adapter configuration validation and defaults

## 4. Individual Adapter Modules

- [x] 4.1 Create adapters/openalex/ package structure
- [ ] 4.2 Create adapters/unpaywall/ package structure
- [ ] 4.3 Create adapters/crossref/ package structure
- [ ] 4.4 Create adapters/pmc/ package structure
- [ ] 4.5 Create remaining adapter packages as needed
- [x] 4.6 Implement OpenAlexAdapter with work metadata extraction and PDF asset retrieval
- [x] 4.7 Add PDF detection logic to identify works with available full-text PDFs
- [ ] 4.8 Implement PDF download functionality using pyalex client integration
- [x] 4.9 Add document_type="pdf" flagging for works with downloadable PDFs
- [ ] 4.10 Create PDF persistence logic to store downloaded assets for MinerU processing
- [ ] 4.11 Implement UnpaywallAdapter with OA status resolution
- [ ] 4.12 Implement CrossrefAdapter with DOI registration data
- [ ] 4.13 Implement PMCAdapter with full-text article access
- [ ] 4.14 Create adapter-specific configuration and validation

## 5. Plugin Manager Integration

- [ ] 5.1 Update plugin manager to work with new adapter structure
- [ ] 5.2 Create adapter discovery and registration mechanisms
- [ ] 5.3 Update adapter metadata and capability reporting
- [ ] 5.4 Add adapter configuration validation
- [ ] 5.5 Update plugin loading and initialization
- [ ] 5.6 Implement adapter dependency resolution and ordering
- [ ] 5.7 Add adapter health monitoring and failure detection
- [ ] 5.8 Create adapter performance tracking and optimization
- [ ] 5.9 Implement adapter security validation and access control
- [ ] 5.10 Add adapter configuration hot-reloading capabilities

## 6. Migration & Testing

- [ ] 6.1 Migrate existing adapter implementations to new structure
- [ ] 6.2 Create comprehensive unit tests for each adapter
- [ ] 6.3 Integration tests for adapter plugin manager
- [ ] 6.4 Performance tests for adapter overhead
- [ ] 6.5 Create migration utilities for existing configurations
- [ ] 6.6 Test adapter dependency chains and interaction patterns
- [ ] 6.7 Test adapter error recovery and circuit breaker behavior
- [ ] 6.8 Test adapter performance under load and rate limits
- [ ] 6.9 Test adapter security boundaries and access control
- [ ] 6.10 Test adapter configuration validation and error handling

## 7. Documentation & Developer Experience

- [ ] 7.1 Update adapter development documentation
- [ ] 7.2 Add examples for creating new biomedical adapters
- [ ] 7.3 Update configuration documentation for new adapter structure
- [ ] 7.4 Create migration guide for existing adapter code
- [ ] 7.5 Add troubleshooting guide for common adapter issues
- [ ] 7.6 Document adapter performance tuning and optimization
- [ ] 7.7 Create adapter security best practices guide
- [ ] 7.8 Add adapter debugging and monitoring documentation
- [ ] 7.9 Create adapter API integration testing guide
- [ ] 7.10 Add adapter deployment and configuration management guide

## 8. Legacy Code Decommissioning

### Phase 1: Remove Monolithic Adapter (Week 1)

- [ ] 8.1 **DECOMMISSION**: Remove `src/Medical_KG_rev/adapters/biomedical.py` monolithic file
- [ ] 8.2 **DECOMMISSION**: Delete hardcoded adapter implementations (OpenAlex, Unpaywall, Crossref, PMC)
- [ ] 8.3 **DECOMMISSION**: Remove legacy HTTP client usage (requests, urllib3)
- [ ] 8.4 **DECOMMISSION**: Delete unused adapter utility functions and helpers
- [ ] 8.5 **DECOMMISSION**: Remove legacy adapter configuration and validation code

### Phase 2: Clean Up Dependencies (Week 2)

- [ ] 8.6 **DECOMMISSION**: Remove unused HTTP client dependencies (requests, urllib3)
- [ ] 8.7 **DECOMMISSION**: Delete legacy adapter error handling and fallback mechanisms
- [ ] 8.8 **DECOMMISSION**: Remove unused adapter import statements and dependencies
- [ ] 8.9 **DECOMMISSION**: Clean up legacy adapter test fixtures and mocks
- [ ] 8.10 **DECOMMISSION**: Remove legacy adapter debugging and introspection tools

### Phase 3: Documentation and Cleanup (Week 3)

- [ ] 8.11 **DECOMMISSION**: Update documentation to remove references to monolithic adapter
- [ ] 8.12 **DECOMMISSION**: Remove legacy adapter examples and configuration templates
- [ ] 8.13 **DECOMMISSION**: Clean up unused adapter configuration files
- [ ] 8.14 **DECOMMISSION**: Remove legacy adapter performance monitoring code
- [ ] 8.15 **DECOMMISSION**: Final cleanup of unused files and directories
