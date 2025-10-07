# Design Document: Universal Embedding System

## Context

The Medical_KG_rev retrieval pipeline requires support for multiple embedding paradigms: dense bi-encoders (BGE, E5), learned-sparse (SPLADE), multi-vector late-interaction (ColBERT), and neural-sparse (OpenSearch ML). Each paradigm has different storage requirements, query patterns, and performance characteristics. A unified embedding system enables hybrid retrieval strategies while maintaining modularity for research and optimization.

## Goals / Non-Goals

### Goals

1. **Universal interface**: Support all embedding paradigms through BaseEmbedder protocol
2. **Namespace management**: Prevent dimension/version conflicts via strict governance
3. **Production + research**: Battle-tested models with experimental opt-ins
4. **Framework compatibility**: Wrap LangChain, LlamaIndex, Haystack embedders
5. **GPU-efficient**: Batch processing with fail-fast on unavailability
6. **Storage routing**: Automatic routing by embedding kind to appropriate backend

### Non-Goals

- Multimodal embeddings (text-only for phase 1)
- Custom embedding model training (use pre-trained only)
- Automatic embedder selection (manual configuration)

## Decisions

### Decision 1: Multi-Paradigm Universal Interface

**Rationale**: Different retrieval strategies require different embedding types. Support all through unified `BaseEmbedder` interface that returns `EmbeddingRecord` with discriminated unions.

**Interface**:

```python
class EmbeddingRecord(BaseModel):
    namespace: str
    kind: Literal["single_vector", "multi_vector", "sparse", "neural_sparse"]
    dim: int | None
    vectors: list[list[float]] | None  # Dense: 1xD, Multi-vector: NxD
    terms: dict[str, float] | None  # Sparse: term→weight
    normalized: bool = True
```

**Alternatives considered**: Separate interfaces per paradigm (rejected - reduces composability)

### Decision 2: Namespace-Based Version Control

**Rationale**: Prevent embedding conflicts via namespace isolation with format: `{kind}.{model}.{dim}.{version}` (e.g., `dense.bge.1024.v1`).

**Alternatives considered**: Implicit versioning (rejected - leads to silent errors)

### Decision 3: Fail-Fast GPU Enforcement

**Rationale**: GPU embeddings orders of magnitude faster and higher quality. Fail immediately if unavailable rather than silently degrading to CPU.

**Alternatives considered**: CPU fallback (rejected - quality degradation unacceptable)

## Risks / Trade-offs

### Risk 1: Framework Dependencies Increase Surface

**Mitigation**: Optional imports, version pinning, security audits

### Risk 2: Multi-Paradigm Complexity

**Mitigation**: Clear documentation, storage router handles complexity

## Migration Plan

Phase 1: Core + dense embedders (Weeks 1-2)
Phase 2: Sparse + multi-vector (Week 3)
Phase 3: Framework + experimental (Week 4)

## Open Questions

1. **Support multi-lingual embeddings?** - No, English-first for phase 1
2. **Auto-tune batch sizes?** - No, use fixed configs initially
