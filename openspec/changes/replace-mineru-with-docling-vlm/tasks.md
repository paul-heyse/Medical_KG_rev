## 1. Infrastructure Setup

- [x] 1.1 Add Docling[vlm] and Gemma3 12B dependencies to requirements.in
      - Add `docling[vlm]>=2.0.0` to requirements.in
      - Add `transformers>=4.36.0` for model loading
      - Add `torch>=2.1.0` with CUDA support
      - Add `pillow>=10.0.0` for image processing
      - Run `pip-compile requirements.in` to update requirements.txt

- [x] 1.2 Update Docker configuration for Gemma3 model support
      - Modify `Dockerfile` to install Docling dependencies
      - Add CUDA 12.1+ support for Gemma3 12B requirements
      - Configure model cache directory `/models/gemma3-12b`
      - Update `docker-compose.yml` with Docling service if needed

- [x] 1.3 Configure GPU memory allocation for Gemma3 12B (requires ~24GB VRAM)
      - Update `config/gpu.yaml` with Gemma3 memory requirements
      - Set `gpu_memory_fraction: 0.95` for Gemma3 12B
      - Configure `max_model_len: 4096` for document processing
      - Add GPU health check that verifies 24GB+ available memory

- [x] 1.4 Set up model download and caching for Gemma3 12B
      - Create `/models/gemma3-12b/` directory structure
      - Add model download script in `scripts/download_gemma3.py`
      - Configure huggingface_hub for authenticated model downloads
      - Set up model validation after download

- [x] 1.5 Update health checks to verify Gemma3 model availability
      - Modify `src/Medical_KG_rev/services/gpu/manager.py` health check
      - Add Gemma3 model loading verification in health endpoint
      - Update `/health` endpoint to check VLM model availability
      - Add GPU memory availability check for 24GB requirement

## 2. Configuration Management

- [x] 2.1 Create Docling configuration class in `src/Medical_KG_rev/config/`
      - Create `src/Medical_KG_rev/config/docling_config.py` with DoclingVLMConfig class
      - Include model_path, batch_size, timeout, retry_attempts, gpu_memory_fraction
      - Add validation for model availability and GPU requirements
      - Implement from_dict/from_yaml class methods for configuration loading

- [x] 2.2 Add Docling settings to main application configuration
      - Update `src/Medical_KG_rev/config/settings.py` to include DoclingVLMSettings
      - Add docling_vlm: DoclingVLMConfig section to main Settings class
      - Set sensible defaults: batch_size=8, timeout=300, retry_attempts=3
      - Add environment variable mapping (DOCLING_VLM_* prefix)

- [x] 2.3 Update environment variable documentation for Docling settings
      - Add Docling section to `docs/guides/environment_variables.md`
      - Document DOCLING_VLM_MODEL_PATH, DOCLING_VLM_BATCH_SIZE, etc.
      - Include example .env file with Docling configuration
      - Add validation examples for configuration values

- [x] 2.4 Migrate vLLM configuration to support Docling model switching
      - Update `src/Medical_KG_rev/config/vllm_config.py` to support multiple model types
      - Add model_type field ("vllm" | "docling_vlm") to configuration
      - Create unified GPUConfig that works for both vLLM and Docling
      - Update config loading to handle both model types

- [x] 2.5 Add feature flag for Docling vs MinerU processing modes
      - Create feature flag in `src/Medical_KG_rev/config/settings.py`
      - Add `pdf_processing_backend: str = "minerv"  # "minerv" | "docling_vlm"`
      - Update environment variable: PDF_PROCESSING_BACKEND
      - Add validation to ensure only valid backend values are accepted

## 3. Core Docling Integration

- [x] 3.1 Create DoclingVLMService class in `src/Medical_KG_rev/services/parsing/`
      - Create `src/Medical_KG_rev/services/parsing/docling_vlm_service.py`
      - Implement DoclingVLMService class inheriting from same base as MineruProcessor
      - Add __init__(config: DoclingVLMConfig, gpu_manager: GPUServiceManager)
      - Implement process_pdf(pdf_path: str) -> DoclingVLMResult method
      - Add model loading with transformers.pipeline for vision-language model
      - Include GPU memory management and model warm-up logic

- [x] 3.2 Implement PDF processing interface compatible with existing pipeline
      - Ensure DoclingVLMService implements same MineruProcessor interface
      - Return DoclingVLMResult with same structure as MineruResult
      - Include text, tables, figures, and metadata extraction
      - Maintain backward compatibility with existing document formats
      - Add provenance tracking for VLM processing (model_version, processing_time)

- [x] 3.3 Add error handling for VLM model failures and fallbacks
      - Create custom exception classes in `src/Medical_KG_rev/services/parsing/exceptions.py`
      - Add DoclingVLMError, DoclingModelLoadError, DoclingProcessingError
      - Implement retry logic with exponential backoff for transient failures
      - Add circuit breaker pattern for persistent model failures
      - Include detailed error logging with model state and GPU memory info

- [x] 3.4 Implement batch processing for multiple PDFs
      - Add process_pdf_batch(pdf_paths: List[str]) -> List[DoclingVLMResult] method
      - Implement intelligent batching based on available GPU memory
      - Add progress tracking with estimated completion times
      - Handle partial batch failures gracefully (return successful results)
      - Optimize batch sizes based on document complexity and available memory

- [x] 3.5 Add performance monitoring and metrics for VLM processing
      - Create VLM-specific metrics class in `src/Medical_KG_rev/services/parsing/metrics.py`
      - Track processing_time_seconds, gpu_memory_usage_mb, model_load_time
      - Add counters for successful/failed processing, retry attempts
      - Implement Prometheus metrics integration
      - Add detailed timing breakdowns (model_loading, pdf_rendering, inference)

## 4. Pipeline Stage Updates

- [ ] 4.1 Update PDF download stage to work with Docling (remove MinerU dependency)
      - Modify `src/Medical_KG_rev/orchestration/stages/pdf_download.py`
      - Remove MinerU-specific metadata and processing flags
      - Update artifact creation to include Docling compatibility flags
      - Ensure download artifacts work with both MinerU and Docling processing
      - Update stage documentation to reflect Docling support

- [ ] 4.2 Modify PDF gate stage to check Docling readiness instead of MinerU
      - Update `src/Medical_KG_rev/orchestration/stages/pdf_gate.py`
      - Change readiness check from `pdf_ir_ready` to `vlm_processing_ready`
      - Add DoclingVLMService health check integration
      - Update error messages to reference Docling instead of MinerU
      - Ensure backward compatibility with existing MinerU gates

- [ ] 4.3 Update orchestration plugins to support Docling-based stages
      - Modify `src/Medical_KG_rev/orchestration/stages/plugins/builtin.py`
      - Update CoreStagePlugin to support DoclingVLMService injection
      - Add Docling-specific stage creation logic
      - Update plugin health checks to include Docling service verification
      - Add feature flag awareness for plugin stage selection

- [ ] 4.4 Remove MinerU-specific stage implementations
      - Archive `src/Medical_KG_rev/services/mineru/` directory
      - Remove MinerU imports from orchestration plugins
      - Update any remaining references to MineruProcessor
      - Move to `src/Medical_KG_rev/services/mineru/archive/` for historical reference
      - Update import statements in affected files

- [ ] 4.5 Add Docling-specific pipeline stages for VLM processing
      - Create `src/Medical_KG_rev/orchestration/stages/docling_vlm_stage.py`
      - Implement DoclingVLMProcessingStage class
      - Add VLM-specific configuration and error handling
      - Include progress tracking for VLM batch processing
      - Add integration with DoclingVLMService for pipeline execution

## 5. Document Processing Integration

- [ ] 5.1 Update document parser to handle Docling VLM output format
      - Modify `src/Medical_KG_rev/services/parsing/docling.py` to support VLM output
      - Add DoclingVLMOutputParser class for processing VLM results
      - Handle structured content extraction (text, tables, figures, metadata)
      - Convert Docling VLM output to internal Document IR format
      - Add validation for VLM-extracted content quality

- [ ] 5.2 Modify chunking pipeline to work with VLM-extracted content
      - Update `src/Medical_KG_rev/chunking/service.py` chunking strategies
      - Add VLM-aware chunking that preserves document structure from VLM
      - Modify chunking to work with enhanced table and figure recognition
      - Update chunking configuration for VLM-processed content
      - Add chunking quality metrics for VLM vs OCR content

- [ ] 5.3 Update entity extraction to leverage VLM document understanding
      - Enhance `src/Medical_KG_rev/services/extraction/service.py`
      - Update entity extraction templates to work with VLM-structured content
      - Improve entity recognition using VLM's document understanding
      - Add VLM-specific entity confidence scoring
      - Maintain existing entity extraction API compatibility

- [ ] 5.4 Ensure provenance tracking works with Docling processing
      - Update `src/Medical_KG_rev/models/provenance.py` for VLM processing
      - Add DoclingVLMProcessingActivity with model_version, processing_config
      - Include GPU memory usage and processing time in provenance
      - Update provenance queries to handle both MinerU and Docling activities
      - Add provenance validation for VLM processing results

- [ ] 5.5 Maintain backward compatibility with existing document formats
      - Ensure VLM-processed documents use same Document IR structure
      - Maintain existing document storage and retrieval APIs
      - Update document validation to accept VLM-generated content
      - Ensure existing chunking and embedding work with VLM documents
      - Add compatibility tests for mixed MinerU/Docling document processing

## 6. Gateway and API Updates

- [ ] 6.1 Update PDF processing endpoints to use Docling service
      - Modify `src/Medical_KG_rev/gateway/rest/router.py` PDF endpoints
      - Update ingestion endpoints to use DoclingVLMService when feature flag enabled
      - Add VLM-specific request parameters (model_version, processing_options)
      - Update response schemas to include VLM processing metadata
      - Add endpoint for direct Docling VLM processing requests

- [ ] 6.2 Modify gRPC services to support Docling-based processing
      - Update `src/Medical_KG_rev/proto/mineru.proto` with Docling support
      - Add DoclingVLMProcessingRequest message type
      - Update MineruService gRPC service to handle Docling requests
      - Modify `src/Medical_KG_rev/gateway/grpc/server.py` implementation
      - Add backward compatibility for existing MinerU gRPC calls

- [ ] 6.3 Update OpenAPI documentation for new PDF processing capabilities
      - Update `docs/openapi.yaml` with Docling VLM endpoints
      - Add request/response schemas for VLM processing
      - Document new feature flags and configuration options
      - Update API examples to show Docling usage
      - Add migration guide for API consumers

- [ ] 6.4 Add monitoring endpoints for Docling service health
      - Create `/health/docling` endpoint in `src/Medical_KG_rev/gateway/rest/router.py`
      - Add DoclingVLMService health status checks
      - Include model availability, GPU memory usage, processing queue status
      - Update `/health` endpoint to include Docling service status
      - Add Prometheus metrics endpoint for VLM processing

- [ ] 6.5 Update error responses for VLM-specific failures
      - Add VLM-specific error codes in `src/Medical_KG_rev/gateway/models.py`
      - Update error handling in `src/Medical_KG_rev/gateway/middleware.py`
      - Add DoclingModelUnavailableError, DoclingProcessingTimeoutError
      - Modify `src/Medical_KG_rev/gateway/presentation/problem_details.py`
      - Update error documentation with VLM-specific troubleshooting

## 7. Testing and Validation

- [ ] 7.1 Create comprehensive unit tests for DoclingVLMService
      - Create `tests/services/parsing/test_docling_vlm_service.py`
      - Test DoclingVLMService initialization with valid/invalid configs
      - Mock transformers pipeline for model loading tests
      - Test PDF processing with sample PDF files
      - Test error handling for model failures and GPU issues
      - Test batch processing with multiple PDFs and partial failures

- [ ] 7.2 Add integration tests for Docling-based PDF processing pipeline
      - Create `tests/integration/test_docling_vlm_pipeline.py`
      - Test end-to-end PDF processing from download to chunking
      - Test integration with orchestration stages
      - Test feature flag routing between MinerU and Docling
      - Test error propagation through the pipeline
      - Test performance with realistic PDF corpus

- [ ] 7.3 Update performance benchmarks for VLM vs OCR processing
      - Create `tests/performance/test_vlm_vs_ocr_benchmark.py`
      - Compare processing times for same PDF corpus
      - Measure accuracy improvements (table extraction, entity recognition)
      - Test GPU memory usage and throughput
      - Create performance regression tests
      - Update existing performance test suite for VLM

- [ ] 7.4 Add contract tests for API compatibility
      - Update `tests/contract/test_pdf_processing_api.py`
      - Test API endpoints work with both MinerU and Docling backends
      - Verify response schemas remain consistent
      - Test feature flag behavior in API responses
      - Add contract tests for new Docling-specific endpoints
      - Ensure backward compatibility for existing API consumers

- [ ] 7.5 Create regression tests comparing MinerU vs Docling outputs
      - Create `tests/regression/test_mineru_vs_docling_comparison.py`
      - Use same input PDF corpus for both processing methods
      - Compare extracted text, tables, and metadata
      - Measure accuracy differences quantitatively
      - Track regression metrics over time
      - Alert on significant accuracy degradation

## 8. Monitoring and Observability

- [ ] 8.1 Add Prometheus metrics for VLM processing performance
      - Update `src/Medical_KG_rev/observability/metrics.py` with VLM metrics
      - Add docling_vlm_processing_time_seconds histogram
      - Add docling_vlm_gpu_memory_usage_mb gauge
      - Add docling_vlm_batch_size gauge and docling_vlm_success_rate counter
      - Add docling_vlm_model_load_time_seconds histogram
      - Update Prometheus configuration to scrape new VLM metrics

- [ ] 8.2 Update Grafana dashboards for Docling-specific monitoring
      - Create new Grafana dashboard `docling-vlm-performance.json`
      - Add panels for VLM processing time, success rate, GPU usage
      - Include comparison panels showing MinerU vs Docling performance
      - Add alerting panels for VLM model failures and performance degradation
      - Update existing PDF processing dashboard with VLM metrics

- [ ] 8.3 Add structured logging for VLM processing operations
      - Update `src/Medical_KG_rev/services/parsing/docling_vlm_service.py` logging
      - Add structured logs for model loading, processing start/end, errors
      - Include correlation IDs and request tracing in VLM logs
      - Add performance metrics to log entries (processing_time, gpu_usage)
      - Update log aggregation to handle VLM-specific log fields

- [ ] 8.4 Implement alerting for VLM model failures or performance degradation
      - Create alerting rules in `config/monitoring/alerts.yml`
      - Alert on docling_vlm_processing_time_seconds > 300 (5 minutes)
      - Alert on docling_vlm_success_rate < 0.95 (95% success rate)
      - Alert on docling_vlm_gpu_memory_usage_mb > 22000 (22GB threshold)
      - Add PagerDuty/Slack integration for VLM-specific alerts

- [ ] 8.5 Update tracing to include VLM processing spans
      - Modify `src/Medical_KG_rev/observability/tracing.py` for VLM spans
      - Add "docling_vlm.process_pdf" span with model_version, batch_size tags
      - Add "docling_vlm.model_load" span for model initialization
      - Include GPU memory usage in span attributes
      - Update Jaeger configuration to display VLM processing traces

## 9. Documentation and Migration

- [ ] 9.1 Update architecture documentation for VLM-based processing
      - Update `docs/architecture/overview.md` with VLM processing details
      - Modify `docs/guides/developer_guide.md` to include Docling integration
      - Update system diagrams to show Docling VLM flow
      - Add section on VLM model management and GPU requirements
      - Update performance characteristics documentation

- [ ] 9.2 Create migration guide for transitioning from MinerU to Docling
      - Create `docs/guides/docling_migration_guide.md`
      - Document step-by-step migration process with rollback procedures
      - Include before/after configuration examples
      - Add troubleshooting section for common migration issues
      - Provide timeline and risk assessment for migration

- [ ] 9.3 Update operational runbooks for Docling maintenance
      - Update `docs/operational-runbook.md` with Docling-specific procedures
      - Add Docling model update procedures
      - Include GPU memory management guidelines
      - Add VLM processing troubleshooting workflows
      - Update monitoring and alerting procedures

- [ ] 9.4 Add troubleshooting guide for VLM-specific issues
      - Create `docs/troubleshooting/docling_vlm_issues.md`
      - Document common VLM processing failures and solutions
      - Include GPU memory troubleshooting steps
      - Add model loading and configuration debugging guides
      - Provide performance optimization recommendations

- [ ] 9.5 Update developer documentation for Docling integration
      - Update `docs/guides/developer_guide.md` with Docling sections
      - Add DoclingVLMService API documentation
      - Include configuration examples and best practices
      - Add development setup instructions for VLM development
      - Update testing guidelines for VLM-based features

## 10. Deployment and Rollout

- [ ] 10.1 Update Kubernetes manifests for Docling service deployment
      - Modify `ops/k8s/docling-vlm-deployment.yaml` with Gemma3 requirements
      - Add GPU resource requests/limits for 24GB VRAM requirement
      - Configure model volume mounts for Gemma3 12B caching
      - Update health checks to verify Docling service readiness
      - Add resource monitoring for GPU memory usage

- [ ] 10.2 Create database migration scripts for configuration changes
      - Create `scripts/migrations/add_docling_config.sql`
      - Add docling_vlm_config table for storing model settings
      - Create indexes on model_version and enabled flags
      - Add migration script for feature flag table if needed
      - Update existing configuration tables for Docling support

- [ ] 10.3 Implement blue-green deployment strategy for VLM rollout
      - Create blue-green deployment configuration in `ops/k8s/`
      - Set up separate blue/green environments for VLM testing
      - Implement automated traffic shifting based on success metrics
      - Add rollback triggers for failed deployments
      - Configure monitoring for deployment success/failure

- [ ] 10.4 Add rollback procedures for reverting to MinerU if needed
      - Create `scripts/rollback_to_mineru.sh` automated rollback script
      - Document manual rollback steps in `docs/guides/rollback_procedures.md`
      - Set up automated rollback triggers based on error rates
      - Preserve MinerU service alongside Docling during transition
      - Update monitoring to detect when rollback is needed

- [ ] 10.5 Update CI/CD pipelines for Docling dependency management
      - Modify `.github/workflows/ci-cd.yml` to include Docling dependencies
      - Add Gemma3 model download and validation in CI pipeline
      - Update Docker build process for Docling requirements
      - Add integration tests for Docling service in CI
      - Configure artifact storage for model caching

## 11. Security and Compliance

- [ ] 11.1 Review Docling for security implications in medical data processing
      - Conduct security assessment of Docling[vlm] library dependencies
      - Review Gemma3 model for potential security vulnerabilities
      - Assess GPU memory handling for data leakage risks
      - Document security findings in `docs/security/docling_security_assessment.md`
      - Create mitigation strategies for identified risks

- [ ] 11.2 Ensure VLM processing maintains HIPAA compliance for medical documents
      - Review Docling processing for PHI data handling compliance
      - Update `docs/guides/compliance_documentation.md` with VLM requirements
      - Verify data encryption at rest and in transit for VLM processing
      - Add HIPAA compliance checklist for VLM model deployment
      - Document data retention policies for VLM processing artifacts

- [ ] 11.3 Update audit logging for VLM processing operations
      - Modify `src/Medical_KG_rev/auth/audit.py` for VLM processing events
      - Add audit events for model loading, processing start/end, errors
      - Include user context and document identifiers in VLM audit logs
      - Update audit log retention policies for VLM processing data
      - Add compliance reporting for VLM processing activities

- [ ] 11.4 Verify data encryption works with VLM processing pipeline
      - Test encryption/decryption of PDF data during VLM processing
      - Verify GPU memory is properly cleared after processing
      - Update `src/Medical_KG_rev/validation/fhir.py` for VLM compatibility
      - Add encryption validation tests for VLM processing pipeline
      - Document encryption requirements for VLM model storage

- [ ] 11.5 Update access controls for VLM model management
      - Add RBAC permissions for DoclingVLMService management
      - Implement model access controls in `src/Medical_KG_rev/auth/scopes.py`
      - Add audit logging for model configuration changes
      - Update `src/Medical_KG_rev/auth/dependencies.py` for VLM endpoints
      - Document access control requirements for VLM operations

## 12. Performance Optimization

- [ ] 12.1 Optimize batch sizes for Gemma3 12B model processing
      - Update `src/Medical_KG_rev/config/docling_config.py` with dynamic batch sizing
      - Implement adaptive batch sizing based on GPU memory availability
      - Add batch size testing and optimization in `scripts/benchmark_batch_sizes.py`
      - Monitor batch processing performance and adjust automatically
      - Document optimal batch sizes for different document types

- [ ] 12.2 Implement caching for repeated VLM operations
      - Add Redis-based caching for VLM processing results in `src/Medical_KG_rev/services/parsing/`
      - Implement content-based cache keys for PDF similarity detection
      - Add cache invalidation strategies for model updates
      - Monitor cache hit rates and adjust cache sizes accordingly
      - Add cache warming for frequently processed document types

- [ ] 12.3 Add model warm-up procedures for consistent performance
      - Create `src/Medical_KG_rev/services/parsing/model_warmup.py`
      - Implement GPU memory pre-allocation for Gemma3 12B
      - Add model loading and inference warm-up routines
      - Measure and document warm-up time requirements
      - Include warm-up status in health checks

- [ ] 12.4 Monitor and optimize GPU memory usage for VLM models
      - Update `src/Medical_KG_rev/services/gpu/manager.py` for VLM monitoring
      - Add real-time GPU memory usage tracking for Gemma3 processing
      - Implement memory defragmentation for long-running VLM services
      - Add memory usage alerts and automatic cleanup procedures
      - Create performance profiling tools for VLM memory optimization

- [ ] 12.5 Implement request queuing for VLM processing under load
      - Add Redis-based request queue in `src/Medical_KG_rev/services/parsing/queue.py`
      - Implement priority queuing for different document types
      - Add queue length monitoring and alerting
      - Implement graceful degradation under high load
      - Add request timeout and retry mechanisms for queued requests
