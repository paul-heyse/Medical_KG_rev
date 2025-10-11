## 1. Infrastructure and Dependencies

- [x] 1.1 Add core Docling and retrieval dependencies to requirements.in
      - Add `docling-core>=2.0.0` for core Docling functionality (repo)
      - Add `transformers>=4.36.0` for model loading and inference
      - Add `faiss-cpu>=1.12.0` for vector similarity search
      - Add `pyserini>=1.2.0` for information retrieval toolkit
      - Add `duckdb>=1.4.1` for chunk store database
      - Run `pip-compile requirements.in` to update requirements.txt

- [x] 1.2 Update Docker configuration for Docling VLM service
      - Modify `Dockerfile` to install docling[vlm]>=2.0.0 and vllm>=0.11.0
      - Add CUDA 12.1+ support for Gemma3 12B requirements
      - Configure model cache directory `/models/docling-vlm`
      - Update `docker-compose.yml` with Docling VLM service
      - Add GPU resource allocation for 24GB VRAM requirement
      - Include torch>=2.1.0 with CUDA support in Docker
      - Create `ops/docker/docling_vlm_dockerfile.py` for Docker setup

- [x] 1.3 Configure GPU memory allocation for Docling VLM (requires ~24GB VRAM)
      - Update `config/gpu.yaml` with Docling VLM memory requirements
      - Set `gpu_memory_fraction: 0.95` for Docling VLM processing
      - Configure `max_model_len: 4096` for document processing
      - Add GPU health check that verifies 24GB+ available memory
      - Update GPU manager to handle Docling VLM resource requirements

- [x] 1.4 Set up model download and caching for Docling VLM
      - Create `/models/docling-vlm/` directory structure
      - Add model download script in `ops/docker/docling_vlm_setup.py`
      - Configure huggingface_hub for authenticated model downloads
      - Set up model validation after download
      - Add model warm-up procedures for consistent performance

- [x] 1.5 Update health checks to verify Docling VLM availability
      - Modify `src/Medical_KG_rev/services/gpu/manager.py` health check
      - Add Docling VLM model loading verification in health endpoint
      - Update `/health` endpoint to check VLM model availability
      - Add GPU memory availability check for 24GB requirement
      - Add Docling VLM service readiness check

## 2. Configuration Management

- [x] 2.1 Create Docling VLM configuration class in `src/Medical_KG_rev/config/`
      - Create `src/Medical_KG_rev/config/docling_vlm_config.py` with DoclingVLMConfig class
      - Include model_name, model_cache_dir, batch_size, timeout, retry_attempts
      - Add GPU memory allocation and model warm-up settings
      - Implement from_dict/from_yaml class methods for configuration loading
      - Add validation for model availability and GPU requirements

- [x] 2.2 Create retrieval configuration classes
      - Create `src/Medical_KG_rev/config/retrieval_config.py` with retrieval settings
      - Add BM25Config, SPLADEConfig, Qwen3Config dataclasses
      - Include field boosts, tokenizers, thresholds, and storage backends
      - Add validation for tokenizer alignment and model compatibility
      - Implement configuration loading from YAML files

- [x] 2.3 Add Docling and retrieval settings to main application configuration
      - Update `src/Medical_KG_rev/config/settings.py` to include DoclingVLMSettings
      - Add docling_vlm: DoclingVLMConfig section to main Settings class
      - Add retrieval: RetrievalConfig section with BM25, SPLADE, Qwen3 configs
      - Set sensible defaults: batch_size=8, timeout=300, retry_attempts=3
      - Add environment variable mapping (DOCLING_VLM_*, RETRIEVAL_* prefixes)

- [ ] 2.4 Migrate existing vLLM configuration to support Docling model switching
      - Update `src/Medical_KG_rev/config/vllm_config.py` to support multiple model types
      - Add model_type field ("vllm" | "docling_vlm") to configuration
      - Create unified GPUConfig that works for both vLLM and Docling
      - Update config loading to handle both model types
      - Add feature flag for Docling vs vLLM processing modes

- [x] 2.5 Add feature flag for Docling vs MinerU processing modes
      - Create feature flag in `src/Medical_KG_rev/config/settings.py`
      - Add `pdf_processing_backend: str = "minerv"  # "minerv" | "docling_vlm"`
      - Add `retrieval_backend: str = "hybrid"  # "bm25" | "splade" | "qwen3" | "hybrid"`
      - Update environment variable: PDF_PROCESSING_BACKEND, RETRIEVAL_BACKEND
      - Add validation to ensure only valid backend values are accepted

## 3. Core Docling VLM Integration

- [x] 3.1 Create gRPC service definitions for Docling VLM
      - Create `src/Medical_KG_rev/proto/docling_vlm_service.proto`
      - Define DoclingVLMService with ProcessPDF, ProcessPDFBatch, GetHealth RPCs
      - Define DocTagsResult message with DocumentStructure, Table, Figure, TextBlock
      - Include ProcessingMetadata and ProcessingStatus messages
      - Add comprehensive error handling and status codes

- [x] 3.2 Implement gRPC client for Docling VLM service
      - Create `src/Medical_KG_rev/services/parsing/docling_vlm_client.py`
      - Implement DoclingVLMClient class with gRPC communication
      - Add async process_pdf(pdf_path: str) -> DocTagsResult method
      - Include gRPC error handling and circuit breaker patterns
      - Add comprehensive logging and monitoring for gRPC calls

- [x] 3.3 Add Docling VLM error handling and fallbacks
      - Create custom exception classes in `src/Medical_KG_rev/services/parsing/exceptions.py`
      - Add DoclingVLMError, DoclingModelLoadError, DoclingProcessingError
      - Implement retry logic with exponential backoff for transient failures
      - Add circuit breaker pattern for persistent model failures
      - Include detailed error logging with model state and GPU memory info

- [x] 3.4 Implement batch processing for multiple PDFs
      - Add process_pdf_batch(pdf_paths: List[str]) -> List[DocTagsResult] method
      - Implement intelligent batching based on available GPU memory
      - Add progress tracking with estimated completion times
      - Handle partial batch failures gracefully (return successful results)
      - Optimize batch sizes based on document complexity and available memory

- [x] 3.5 Add performance monitoring for VLM processing
      - Create VLM-specific metrics class in `src/Medical_KG_rev/services/parsing/metrics.py`
      - Track processing_time_seconds, gpu_memory_usage_mb, model_load_time
      - Add counters for successful/failed processing, retry attempts
      - Implement Prometheus metrics integration
      - Add detailed timing breakdowns (model_loading, pdf_rendering, inference)

## 4. Chunking with Tokenizer Alignment

- [ ] 4.1 Implement hybrid chunker with tokenizer alignment
      - Create `src/Medical_KG_rev/chunking/hybrid_chunker.py`
      - Implement hierarchy-first segmentation (titles, sections, paragraphs, tables)
      - Add tokenizer-aware split/merge using SPLADE tokenizer
      - Ensure "measured tokens" in chunking match SPLADE's 512 cap
      - Configure chunk sizes: 350-500 tokens, up to 700 when structure warrants

- [ ] 4.2 Create contextualized text serialization
      - Add contextualized_text and content_only_text fields to chunks
      - Prepend section_path to contextualized text for dense embeddings
      - Keep content_only_text for BM25/SPLADE without synthetic prefixes
      - Include caption text in contextualized text when present
      - Preserve table structure in both serializations

- [ ] 4.3 Implement deterministic chunk identifiers
      - Create deterministic chunk_id from (doc_id, page_no, element_path, char_span)
      - Ensure chunk_id is stable across reprocessing of same document
      - Add chunk validation to catch empty or pathological chunks
      - Implement automatic quarantine for malformed inputs
      - Add chunk quality metrics for monitoring

## 5. SPLADE-v3 with Rep-Max Aggregation

- [x] 5.1 Create SPLADE-v3 service integration
      - Create `src/Medical_KG_rev/services/retrieval/splade_service.py`
      - Implement SPLADE-v3 encoder with transformers pipeline
      - Add model loading and tokenizer initialization
      - Configure SPLADE-v3 model for document processing
      - Add model warm-up and memory management

- [x] 5.2 Implement chunk segmentation for SPLADE
      - Add segment_chunk_for_splade(chunk_text: str, tokenizer) -> List[Segment]
      - Split chunk text into ≤512-token segments using SPLADE tokenizer
      - Keep segment order and boundaries for aggregation
      - Handle edge cases (very short chunks, very long chunks)
      - Add segment validation and error handling

- [x] 5.3 Implement Rep-Max aggregation
      - Add aggregate_splade_segments(segments: List[Segment]) -> SparseVector
      - Merge segment maps by taking maximum weight per term
      - Create one learned-sparse vector per chunk
      - Handle term conflicts and weight normalization
      - Add aggregation validation and quality checks

- [x] 5.4 Implement sparsity control and quantization
      - Add apply_sparsity_threshold(vector: SparseVector, threshold: float) -> SparseVector
      - Cap maximum terms per chunk (top few thousand by weight)
      - Quantize weights to fixed-point integers with consistent scale factor
      - Store as Lucene "impacts" for compact representation
      - Add quantization validation and round-trip testing

- [x] 5.5 Create SPLADE impact index storage
      - Create `src/Medical_KG_rev/services/vector_store/stores/splade_index.py`
      - Implement Lucene impact index for SPLADE vectors
      - Add {chunk_id → [(term, impact), …]} mapping
      - Include tokenizer name, model name, quantization scale in manifest
      - Add index validation and consistency checks

## 6. BM25 Index with Medical Structure

- [x] 6.1 Create structured BM25 service
      - Create `src/Medical_KG_rev/services/retrieval/bm25_service.py`
      - Implement multi-field Lucene index for lexical retrieval
      - Add field configuration: title, section_headers, paragraph, caption, table_text
      - Include footnote and refs_text fields with appropriate boosts
      - Add field-specific analyzers and tokenizers

- [x] 6.2 Implement BM25 field mapping
      - Create field mapping from chunk structure to BM25 fields
      - Map title and section_headers with high boosts
      - Map paragraph as backbone with standard settings
      - Map caption and table_text with moderate boosts
      - Map footnote with small boost, refs_text with low/zero boost

- [x] 6.3 Configure BM25 analyzers and tokenizers
      - Add standard tokenizer with lowercase, stopword removal, light stemming
      - Preserve medical terms exactly in title field (no stemming)
      - Consider MeSH/UMLS synonym filter for BM25-only index
      - Tune BM25 parameters per field family
      - Document and freeze BM25 defaults as experiment config

- [x] 6.4 Create BM25 index storage and management
      - Create `src/Medical_KG_rev/services/vector_store/stores/bm25_index.py`
      - Implement Lucene index with multi-field configuration
      - Add index building from chunk store data
      - Include field boosts and analyzer configurations
      - Add index validation and consistency checks

## 7. Qwen3 Embedding Integration

- [x] 7.1 Create Qwen3 embedding service
      - Create `src/Medical_KG_rev/services/retrieval/qwen3_service.py`
      - Implement Qwen3 4096-dimension embedding generation
      - Add model loading and tokenizer initialization
      - Configure model for 4096-dimension vectors
      - Add GPU memory management for embedding processing

- [x] 7.2 Implement contextualized text embedding
      - Use chunk's contextualized serialization for embedding input
      - Include section_path and caption text in embedding context
      - Generate 4096-dimension vectors for each chunk
      - Add embedding quality validation and error handling
      - Include provenance tracking (model_version, preprocessing_version)

- [x] 7.3 Create Qwen3 vector storage backend
      - Create `src/Medical_KG_rev/services/vector_store/stores/qwen3_index.py`
      - Implement FAISS index with IVF configuration for scale
      - Add vector_id == chunk_id mapping for consistency
      - Include embedding model version and preprocessing version
      - Add vector validation and quality checks

## 8. Storage Model Implementation

- [ ] 8.1 Create chunk store database
      - Create `src/Medical_KG_rev/storage/chunk_store.py`
      - Implement DuckDB database for chunk storage
      - Add chunk table with all metadata and text fields
      - Include convenience views for analytics (chunks by label, token distribution)
      - Add chunk validation and consistency checks

- [ ] 8.2 Implement storage layout and manifests
      - Create `/data/raw/` for original PDFs
      - Create `/data/doctags/` for DocTags blobs (gzipped)
      - Create `/indexes/bm25_index/` for BM25 Lucene index
      - Create `/indexes/splade_v3_repmax/` for SPLADE impact index
      - Create `/vectors/qwen3_8b_4096.*` for FAISS files
      - Add YAML/JSON manifests beside each asset

- [ ] 8.3 Create manifest management system
      - Create `src/Medical_KG_rev/storage/manifests.py`
      - Record model versions, preprocessing parameters, build timestamps
      - Add manifest validation and consistency checks
      - Include input checksums and versions in manifests
      - Add manifest reading/writing utilities

## 9. Orchestration Pipeline Stages

- [ ] 9.1 Create discrete, restartable pipeline stages
      - Create `src/Medical_KG_rev/orchestration/stages/docling_convert_stage.py`
      - Create `src/Medical_KG_rev/orchestration/stages/chunk_stage.py`
      - Create `src/Medical_KG_rev/orchestration/stages/features_stage.py`
      - Create `src/Medical_KG_rev/orchestration/stages/index_stage.py`
      - Add stage checkpointing and manifest writing

- [ ] 9.2 Implement Convert stage
      - Enqueue documents to Docling VLM service
      - Save DocTags blobs and conversion reports
      - Handle Docling VLM service errors and retries
      - Write stage manifest with input checksums and versions
      - Add conversion validation and error handling

- [ ] 9.3 Implement Chunk stage
      - Load DocTags into DoclingDocument objects
      - Run Hybrid chunking with designated tokenizer
      - Write chunk rows to chunk store database
      - Include validation pass for empty/pathological chunks
      - Write stage manifest with chunking parameters

- [ ] 9.4 Implement Features stage
      - Generate BM25 field texts from chunk store
      - Generate SPLADE vectors with Rep-Max aggregation
      - Generate Qwen3 embeddings from contextualized text
      - Write feature data to chunk store feature tables
      - Write stage manifest with model versions and parameters

- [ ] 9.5 Implement Index stage
      - Build BM25 Lucene index from chunk store BM25 fields
      - Build SPLADE impact index from chunk store SPLADE vectors
      - Add Qwen3 vectors to FAISS/Qdrant backend
      - Write index manifests with model and preprocessing versions
      - Add index validation and consistency checks

## 10. Query-Time Retrieval System

- [ ] 10.1 Create hybrid retrieval service
      - Create `src/Medical_KG_rev/services/retrieval/hybrid_service.py`
      - Implement parallel BM25, SPLADE, and Qwen3 retrieval
      - Add chunk_id joining and result fusion
      - Include reciprocal rank fusion for initial implementation
      - Add performance monitoring for each retriever

- [ ] 10.2 Implement BM25 query processing
      - Add BM25 query generation from user text
      - Implement multi-field query with appropriate boosts
      - Add query expansion and term weighting
      - Return top-K results with relevance scores
      - Include field-specific scoring and ranking

- [ ] 10.3 Implement SPLADE query processing
      - Add SPLADE query vector generation from user text
      - Use same SPLADE tokenizer/model as indexing
      - Score against SPLADE impact index with dot product
      - Return top-K results with sparse similarity scores
      - Add query preprocessing and normalization

- [ ] 10.4 Implement Qwen3 dense retrieval
      - Add Qwen3 query vector generation from user text
      - Query FAISS/Qdrant index with ANN search
      - Return top-K results with cosine similarity scores
      - Add query preprocessing for embedding consistency
      - Include vector normalization and distance calculation

- [ ] 10.5 Implement result fusion and ranking
      - Add reciprocal rank fusion for combining BM25, SPLADE, Qwen3 results
      - Implement chunk_id joining with chunk store lookup
      - Return unified results with chunk text, section_path, page_no, bbox
      - Add provenance tracking for retrieval method identification
      - Include performance metrics for each retrieval method

## 11. Medical Corpus Specifics

- [ ] 11.1 Implement medical text normalization
      - Create `src/Medical_KG_rev/services/parsing/medical_normalization.py`
      - Fix hyphenation at line breaks in medical terms
      - Harmonize Unicode (Greek letters, micro sign vs "u")
      - Preserve units and dosages exactly
      - Add machine-only field with unit-standardized forms
      - Add normalization validation and quality checks

- [ ] 11.2 Implement table fidelity preservation
      - Add table structure preservation in chunk serialization
      - Keep header mapping intact when flattening tables
      - Use "Header: Value" phrasing for table content
      - Include caption text in table chunks
      - Add table schema preservation for rendering

- [ ] 11.3 Add medical terminology support
      - Add MeSH/UMLS synonym filter for BM25 index only
      - Keep learned-sparse and dense encoders on original text
      - Add controlled vocabulary expansions for medical terms
      - Add terminology validation and quality checks
      - Document terminology handling in processing pipeline

## 12. Testing and Validation Framework

- [ ] 12.1 Create comprehensive unit tests for DoclingVLMService
      - Create `tests/services/parsing/test_docling_vlm_service.py`
      - Test DoclingVLMService initialization with valid/invalid configs
      - Mock Docling VLM pipeline for model loading tests
      - Test PDF processing with sample medical PDF files
      - Test error handling for model failures and GPU issues
      - Test batch processing with multiple PDFs and partial failures

- [ ] 12.2 Add integration tests for Docling-based PDF processing pipeline
      - Create `tests/integration/test_docling_vlm_pipeline.py`
      - Test end-to-end PDF processing from Docling to chunking
      - Test integration with orchestration stages
      - Test feature flag routing between MinerU and Docling
      - Test error propagation through the pipeline
      - Test performance with realistic medical document corpus

- [ ] 12.3 Update performance benchmarks for VLM vs OCR processing
      - Create `tests/performance/test_docling_vs_mineru_benchmark.py`
      - Compare processing times for same medical PDF corpus
      - Measure accuracy improvements (table extraction, entity recognition)
      - Test GPU memory usage and throughput
      - Create performance regression tests
      - Update existing performance test suite for VLM

- [ ] 12.4 Add contract tests for retrieval API compatibility
      - Update `tests/contract/test_hybrid_retrieval_api.py`
      - Test retrieval endpoints work with hybrid search
      - Verify response schemas remain consistent across backends
      - Test feature flag behavior in retrieval responses
      - Add contract tests for new hybrid search endpoints
      - Ensure backward compatibility for existing retrieval consumers

- [ ] 12.5 Create regression tests for retrieval accuracy
      - Create `tests/regression/test_hybrid_retrieval_accuracy.py`
      - Use medical document corpus for both processing methods
      - Compare retrieval accuracy across BM25, SPLADE, Qwen3, and hybrid
      - Measure nDCG@10 and Hit@1 improvements
      - Track regression metrics over time
      - Alert on significant accuracy degradation

## 13. Monitoring and Observability

- [ ] 13.1 Add comprehensive metrics for hybrid retrieval system
      - Update `src/Medical_KG_rev/observability/metrics.py` with retrieval metrics
      - Add bm25_query_time_seconds, splade_query_time_seconds, qwen3_query_time_seconds
      - Add retrieval_success_rate, chunk_lookup_time, fusion_time
      - Add index_size_bytes, memory_usage_mb for each index type
      - Update Prometheus configuration to scrape retrieval metrics

- [ ] 13.2 Update Grafana dashboards for hybrid retrieval monitoring
      - Create new Grafana dashboard `hybrid-retrieval-performance.json`
      - Add panels for each retrieval method (BM25, SPLADE, Qwen3) performance
      - Include comparison panels showing hybrid vs individual method performance
      - Add alerting panels for retrieval failures and performance degradation
      - Update existing retrieval dashboard with hybrid metrics

- [ ] 13.3 Add structured logging for hybrid retrieval operations
      - Update `src/Medical_KG_rev/services/retrieval/hybrid_service.py` logging
      - Add structured logs for query processing, result fusion, errors
      - Include correlation IDs and request tracing in retrieval logs
      - Add performance metrics to log entries (query_time, result_count)
      - Update log aggregation to handle retrieval-specific log fields

- [ ] 13.4 Implement alerting for retrieval failures or performance degradation
      - Create alerting rules in `config/monitoring/alerts.yml`
      - Alert on retrieval query_time > 500ms (P95 threshold)
      - Alert on retrieval success_rate < 0.95 (95% success rate)
      - Alert on index corruption or missing chunks
      - Add PagerDuty/Slack integration for retrieval-specific alerts

- [ ] 13.5 Update tracing to include hybrid retrieval spans
      - Modify `src/Medical_KG_rev/observability/tracing.py` for retrieval spans
      - Add "retrieval.hybrid.query" span with method breakdown
      - Add "retrieval.bm25.search", "retrieval.splade.search", "retrieval.qwen3.search" spans
      - Include result counts and performance metrics in span attributes
      - Update Jaeger configuration to display hybrid retrieval traces

## 14. Documentation and Migration

- [ ] 14.1 Update architecture documentation for hybrid retrieval system
      - Update `docs/architecture/overview.md` with hybrid retrieval details
      - Modify `docs/guides/developer_guide.md` to include hybrid retrieval
      - Update system diagrams to show BM25 + SPLADE + Qwen3 flow
      - Add section on storage model with chunk store + indexes + manifests
      - Update performance characteristics documentation

- [ ] 14.2 Create migration guide for transitioning from MinerU to Docling
      - Create `docs/guides/docling_hybrid_retrieval_migration.md`
      - Document step-by-step migration process with rollback procedures
      - Include before/after configuration examples
      - Add troubleshooting section for common migration issues
      - Provide timeline and risk assessment for migration

- [ ] 14.3 Update operational runbooks for Docling hybrid retrieval
      - Update `docs/operational-runbook.md` with hybrid retrieval procedures
      - Add Docling VLM model update procedures
      - Include retrieval index rebuild and maintenance procedures
      - Add hybrid retrieval performance monitoring workflows
      - Update monitoring and alerting procedures

- [ ] 14.4 Add troubleshooting guide for hybrid retrieval issues
      - Create `docs/troubleshooting/hybrid_retrieval_issues.md`
      - Document common retrieval failures and solutions
      - Include index corruption and chunk store troubleshooting
      - Add performance optimization recommendations
      - Provide debugging guides for retrieval pipeline issues

- [ ] 14.5 Update developer documentation for hybrid retrieval integration
      - Update `docs/guides/developer_guide.md` with hybrid retrieval sections
      - Add DoclingVLMService and hybrid retrieval API documentation
      - Include configuration examples and best practices
      - Add development setup instructions for hybrid retrieval development
      - Update testing guidelines for hybrid retrieval features

## 15. Deployment and Rollout

- [ ] 15.1 Update Kubernetes manifests for Docling hybrid retrieval deployment
      - Modify `ops/k8s/docling-vlm-deployment.yaml` with hybrid retrieval requirements
      - Add GPU resource requests/limits for 24GB VRAM requirement
      - Configure model volume mounts for Docling VLM caching
      - Update health checks to verify hybrid retrieval service readiness
      - Add resource monitoring for retrieval system components

- [ ] 15.2 Create database migration scripts for hybrid retrieval
      - Create `scripts/migrations/add_hybrid_retrieval_config.sql`
      - Add chunk_store, bm25_index, splade_index, qwen3_vectors tables
      - Create indexes on chunk_id and doc_id for performance
      - Add feature flag table for migration control
      - Update existing configuration tables for hybrid retrieval support

- [ ] 15.3 Implement blue-green deployment strategy for hybrid retrieval rollout
      - Create blue-green deployment configuration in `ops/k8s/`
      - Set up separate blue/green environments for hybrid retrieval testing
      - Implement automated traffic shifting based on success metrics
      - Add rollback triggers for failed deployments
      - Configure monitoring for deployment success/failure

- [ ] 15.4 Add rollback procedures for reverting to MinerU if needed
      - Create `scripts/rollback_to_mineru_hybrid.sh` automated rollback script
      - Document manual rollback steps in `docs/guides/rollback_procedures.md`
      - Set up automated rollback triggers based on error rates and performance
      - Preserve MinerU service alongside hybrid retrieval during transition
      - Update monitoring to detect when rollback is needed

- [ ] 15.5 Update CI/CD pipelines for hybrid retrieval dependency management
      - Modify `.github/workflows/ci-cd.yml` to include hybrid retrieval dependencies
      - Add Docling VLM model download and validation in CI pipeline (Docker container)
      - Update Docker build process for hybrid retrieval requirements
      - Add integration tests for hybrid retrieval service in CI
      - Configure artifact storage for model and index caching
      - Test Docker container functionality in CI pipeline

## 16. Security and Compliance

- [ ] 16.1 Review Docling and hybrid retrieval for security implications
      - Conduct security assessment of Docling[vlm] library dependencies
      - Review hybrid retrieval components for potential security vulnerabilities
      - Assess GPU memory handling for data leakage risks in retrieval
      - Document security findings in `docs/security/hybrid_retrieval_security_assessment.md`
      - Create mitigation strategies for identified risks

- [ ] 16.2 Ensure hybrid retrieval maintains HIPAA compliance
      - Review hybrid retrieval processing for PHI data handling compliance
      - Update `docs/guides/compliance_documentation.md` with hybrid retrieval requirements
      - Verify data encryption at rest and in transit for retrieval operations
      - Add HIPAA compliance checklist for hybrid retrieval deployment
      - Document data retention policies for retrieval artifacts

- [ ] 16.3 Update audit logging for hybrid retrieval operations
      - Modify `src/Medical_KG_rev/auth/audit.py` for hybrid retrieval events
      - Add audit events for retrieval queries, result fusion, errors
      - Include user context and query parameters in retrieval audit logs
      - Update audit log retention policies for retrieval data
      - Add compliance reporting for retrieval activities

- [ ] 16.4 Verify data encryption works with hybrid retrieval pipeline
      - Test encryption/decryption of retrieval data and results
      - Verify GPU memory is properly cleared after retrieval processing
      - Update `src/Medical_KG_rev/validation/` for hybrid retrieval compatibility
      - Add encryption validation tests for hybrid retrieval pipeline
      - Document encryption requirements for retrieval model storage

- [ ] 16.5 Update access controls for hybrid retrieval management
      - Add RBAC permissions for hybrid retrieval service management
      - Implement retrieval access controls in `src/Medical_KG_rev/auth/scopes.py`
      - Add audit logging for retrieval configuration changes
      - Update `src/Medical_KG_rev/auth/dependencies.py` for retrieval endpoints
      - Document access control requirements for hybrid retrieval operations

## 17. Performance Optimization

- [ ] 17.1 Optimize batch sizes for hybrid retrieval processing
      - Update `src/Medical_KG_rev/config/` with dynamic batch sizing for retrieval
      - Implement adaptive batch sizing based on GPU memory availability
      - Add batch size testing and optimization in `scripts/benchmark_retrieval_batches.py`
      - Monitor batch processing performance and adjust automatically
      - Document optimal batch sizes for different retrieval operations

- [ ] 17.2 Implement caching for repeated retrieval operations
      - Add Redis-based caching for retrieval results in `src/Medical_KG_rev/services/retrieval/`
      - Implement query-based cache keys for retrieval similarity detection
      - Add cache invalidation strategies for index updates
      - Monitor cache hit rates and adjust cache sizes accordingly
      - Add cache warming for frequently queried document types

- [ ] 17.3 Add model warm-up procedures for hybrid retrieval
      - Create `src/Medical_KG_rev/services/retrieval/model_warmup.py`
      - Implement GPU memory pre-allocation for all retrieval models
      - Add model loading and inference warm-up routines
      - Measure and document warm-up time requirements
      - Include warm-up status in retrieval health checks

- [ ] 17.4 Monitor and optimize GPU memory usage for hybrid retrieval
      - Update `src/Medical_KG_rev/services/gpu/manager.py` for hybrid retrieval monitoring
      - Add real-time GPU memory usage tracking for all retrieval models
      - Implement memory defragmentation for long-running retrieval services
      - Add memory usage alerts and automatic cleanup procedures
      - Create performance profiling tools for hybrid retrieval memory optimization

- [ ] 17.5 Implement request queuing for hybrid retrieval under load
      - Add Redis-based request queue in `src/Medical_KG_rev/services/retrieval/queue.py`
      - Implement priority queuing for different retrieval types
      - Add queue length monitoring and alerting
      - Implement graceful degradation under high retrieval load
      - Add request timeout and retry mechanisms for queued retrieval requests
