## Context

This change replaces the current MinerU + vLLM PDF processing pipeline with Docling's vision-language model (VLM) approach using the Gemma3 12B model. The current system processes PDFs in two phases:

1. **Phase 1 (Fast)**: Metadata extraction and basic document structure
2. **Phase 2 (GPU-intensive)**: MinerU OCR processing with vLLM for layout analysis

The new approach will use Docling's VLM capabilities to perform both layout analysis and text extraction in a single, more accurate pipeline.

## Goals / Non-Goals

### Goals

- Improve PDF processing accuracy through VLM understanding
- Simplify infrastructure by replacing dual OCR+LLM pipeline with single VLM model
- Enhance table and figure recognition capabilities
- Maintain backward compatibility with existing document formats
- Achieve better multi-language support

### Non-Goals

- Completely redesign the document processing pipeline architecture
- Change the external API interfaces (maintain compatibility)
- Remove support for existing MinerU-processed documents
- Implement real-time PDF processing (batch processing is acceptable)

## Decisions

### 1. Docling Integration Architecture

**Decision**: Create a new `DoclingVLMService` that implements the same interface as the current `MineruProcessor` to maintain pipeline compatibility.

**Implementation Details**:

- Create `src/Medical_KG_rev/services/parsing/docling_vlm_service.py` with DoclingVLMService class
- Implement `process_pdf(pdf_path: str) -> DoclingVLMResult` method signature matching MineruResult
- Use `transformers.pipeline("document-question-answering", model="google/gemma-3-12b-it")` for VLM processing
- Return structured data: `{"text": str, "tables": List[Table], "figures": List[Figure], "metadata": Dict}`
- Include provenance tracking: `{"model_version": "gemma-3-12b", "processing_time": float, "gpu_memory_used": int}`

**Rationale**: This allows for gradual migration and maintains the existing orchestration pipeline structure while replacing the underlying processing engine.

**Alternatives Considered**:

- Complete pipeline rewrite: Too risky and time-consuming for production system
- Dual pipeline approach: Would increase complexity and maintenance burden
- API-level replacement: Would break existing integrations and external API contracts

### 2. Model Selection and Configuration

**Decision**: Use Gemma3 12B model with Docling[vlm] for optimal balance of accuracy and resource requirements.

**Implementation Details**:

```python
# Configuration in src/Medical_KG_rev/config/docling_config.py
@dataclass
class DoclingVLMConfig:
    model_name: str = "google/gemma-3-12b-it"
    model_cache_dir: str = "/models/gemma-3-12b"
    max_batch_size: int = 8
    timeout_seconds: int = 300
    retry_attempts: int = 3
    gpu_memory_fraction: float = 0.95
    device_map: str = "auto"
    torch_dtype: str = "float16"
```

**Rationale**: Gemma3 12B provides excellent vision-language capabilities while fitting within reasonable GPU memory constraints (~24GB VRAM). It's well-suited for document understanding tasks.

**Alternatives Considered**:

- Larger models (Gemma3 27B): Excessive memory requirements for current infrastructure (>32GB VRAM)
- Smaller models (Gemma3 4B): Insufficient accuracy for complex documents (<80% accuracy on tables)
- Other VLMs (LLaVA, Qwen-VL): Less proven for document processing tasks and may have licensing issues

### 3. GPU Resource Management

**Decision**: Implement dynamic GPU memory allocation with model warm-up procedures to ensure consistent performance.

**Implementation Details**:

```python
# In src/Medical_KG_rev/services/gpu/manager.py
class GPUServiceManager:
    def acquire_compute_resources(self, model_name: str, memory_requirement_gb: int) -> GPUContext:
        # Check available GPU memory >= 24GB for Gemma3 12B
        available_memory = self._get_available_gpu_memory()
        if available_memory < memory_requirement_gb:
            raise GPUResourceUnavailableError(f"Insufficient GPU memory: {available_memory}GB < {memory_requirement_gb}GB")

        # Allocate memory with fragmentation handling
        context = GPUContext(model_name=model_name, memory_gb=memory_requirement_gb)
        self._allocated_resources[model_name] = context
        return context

    def warmup_model(self, model_name: str) -> None:
        # Pre-load model and run inference warm-up
        if model_name not in self._warm_models:
            # Load Gemma3 12B model
            model = AutoModelForVision2Seq.from_pretrained(model_name)
            # Run dummy inference to warm up GPU kernels
            self._warm_models[model_name] = model
```

**Rationale**: VLM models require significant GPU resources and benefit from warm-up to achieve consistent processing times. Dynamic allocation prevents resource conflicts in multi-tenant environments.

**Alternatives Considered**:

- Static GPU allocation: Less flexible for varying workloads and doesn't handle fragmentation
- CPU fallback: Violates fail-fast principle and degrades performance significantly
- Multiple GPU instances: Increases operational complexity and infrastructure costs

### 4. Error Handling and Fallbacks

**Decision**: Implement graceful degradation with clear error reporting when VLM processing fails, but maintain fail-fast behavior for GPU unavailability.

**Rationale**: VLM processing may fail on certain document types or due to model limitations. Clear error reporting helps with debugging while maintaining system reliability.

**Alternatives Considered**:

- Silent fallbacks: Hides issues and reduces observability
- Always retry: May waste resources on fundamentally incompatible documents
- No error handling: Unacceptable for production systems

### 5. Migration Strategy

**Decision**: Implement feature flag-based migration allowing gradual rollout and rollback capability.

**Rationale**: This change affects core PDF processing functionality. Gradual rollout minimizes risk while allowing performance comparison between old and new approaches.

**Alternatives Considered**:

- Big bang migration: Too risky for critical PDF processing
- Parallel pipelines: Increases complexity and resource usage
- Staged rollout by document type: Too complex to implement and maintain

## Risks / Trade-offs

### Risk: VLM Model Accuracy vs OCR

**Risk**: Docling VLM may not achieve the same accuracy as the proven MinerU + vLLM combination on certain document types.
**Mitigation**:

- Comprehensive testing with diverse document corpus
- Performance benchmarking against current pipeline
- Gradual rollout with monitoring and comparison
- Ability to rollback if accuracy degrades significantly

### Risk: GPU Resource Requirements

**Risk**: Gemma3 12B requires ~24GB VRAM, potentially straining current GPU infrastructure.
**Mitigation**:

- Implement proper GPU memory management and monitoring
- Add circuit breaker patterns for memory exhaustion
- Scale GPU resources as needed for production load
- Monitor and alert on GPU utilization

### Risk: Processing Time Variability

**Risk**: VLM processing times may be less predictable than OCR-based approaches.
**Mitigation**:

- Implement request queuing and batching strategies
- Add performance monitoring and SLO alerts
- Optimize model configuration for consistent performance
- Document expected processing time ranges

### Trade-off: Single Model vs Dual Pipeline

**Trade-off**: Simpler infrastructure vs potentially lower accuracy on edge cases.
**Decision**: Accept this trade-off as VLM models are generally more accurate and the infrastructure simplification outweighs the risks.

### Trade-off: Model Size vs Performance

**Trade-off**: Larger model size provides better accuracy but requires more resources.
**Decision**: Gemma3 12B strikes the right balance for current infrastructure while providing significant accuracy improvements.

## Migration Plan

### Phase 1: Infrastructure Preparation (Week 1)

1. Update dependencies and Docker configuration
2. Set up Gemma3 model download and caching
3. Configure GPU resource allocation
4. Implement basic DoclingVLMService with health checks

### Phase 2: Core Integration (Week 2-3)

1. Replace MinerU service with DoclingVLMService
2. Update pipeline stages to use VLM processing
3. Implement feature flag for migration control
4. Add comprehensive monitoring and metrics

### Phase 3: Testing and Validation (Week 4)

1. Run comprehensive test suite comparing old vs new pipeline
2. Performance benchmarking with production-like load
3. Integration testing with existing document corpus
4. Security and compliance validation

### Phase 4: Gradual Rollout (Week 5-6)

1. Enable feature flag for percentage of traffic
2. Monitor performance and accuracy metrics
3. Compare results with existing MinerU pipeline
4. Gradually increase traffic to new pipeline

### Phase 5: Full Migration (Week 7)

1. Switch default processing to Docling VLM
2. Monitor for any issues or performance degradation
3. Complete migration of all PDF processing
4. Archive MinerU-related code and configurations

### Rollback Plan

If issues arise during rollout:

1. Disable Docling feature flag immediately
2. Fall back to MinerU processing
3. Investigate root cause with comprehensive logging
4. Fix issues and retry rollout
5. Maintain ability to process existing documents with either method

## Open Questions

1. **Model Fine-tuning**: Should we fine-tune Gemma3 12B on medical document corpus for better domain-specific accuracy?

2. **Multi-language Support**: How should we handle documents in languages not well-supported by Gemma3?

3. **Document Type Handling**: Are there specific document types (forms, invoices, research papers) that may need special handling?

4. **Performance Scaling**: How should we scale VLM processing for high-volume scenarios? Separate GPU instances or larger batching?

5. **Cost Optimization**: What's the cost-benefit analysis of VLM processing vs OCR+LLM approach in terms of accuracy vs infrastructure costs?
