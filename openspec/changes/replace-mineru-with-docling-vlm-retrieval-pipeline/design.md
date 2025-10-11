## Context

This change replaces the current MinerU + vLLM PDF processing pipeline with a comprehensive Docling VLM-based retrieval system. The new system implements:

1. **Docling VLM in Docker** for PDF processing (replacing MinerU)
2. **Hybrid Retrieval System** with BM25 + SPLADE-v3 + Qwen3 embeddings
3. **Chunk Store + Separate Indexes** storage model for reproducibility
4. **Tokenizer-Aligned Chunking** for consistent segmentation
5. **SPLADE Rep-Max Aggregation** for learned sparse retrieval
6. **Medical Corpus Optimizations** for domain-specific handling

The system maintains backward compatibility while providing superior document understanding and retrieval accuracy.

## Goals / Non-Goals

### Goals

- Implement Docling VLM with gRPC communication for superior PDF understanding
- Create hybrid retrieval system (BM25 + SPLADE + Qwen3) for improved accuracy
- Establish chunk store + separate indexes storage model for reproducibility
- Implement deterministic chunking with tokenizer alignment
- Add comprehensive provenance tracking for clinical trust
- Design new Doctags-based interfaces (no MinerU compatibility needed)
- Achieve 20-30% accuracy improvement over current MinerU pipeline
- Implement gRPC service architecture for all GPU services

### Non-Goals

- Maintain MinerU compatibility or interfaces
- Use HTTP REST APIs for service communication
- Support existing MinerU-processed documents (clean migration)
- Implement real-time PDF processing (batch processing is acceptable)
- Install torch dependencies in main codebase (torch remains Docker-only)

## Decisions

### 1. Docling VLM gRPC Integration Architecture

**Decision**: Replace MinerU + vLLM with Docling VLM using gRPC communication for PDF processing.

**Implementation Details**:

```protobuf
// docling_vlm_service.proto
service DoclingVLMService {
    rpc ProcessPDF(ProcessPDFRequest) returns (ProcessPDFResponse);
    rpc ProcessPDFBatch(ProcessPDFBatchRequest) returns (ProcessPDFBatchResponse);
    rpc GetHealth(HealthRequest) returns (HealthResponse);
}

message ProcessPDFRequest {
    bytes pdf_content = 1;
    string pdf_path = 2;
    DoclingConfig config = 3;
}

message ProcessPDFResponse {
    DocTagsResult doctags = 1;
    ProcessingMetadata metadata = 2;
    ProcessingStatus status = 3;
    string error_message = 4;
}

message DocTagsResult {
    DocumentStructure document = 1;
    repeated Table tables = 2;
    repeated Figure figures = 3;
    repeated TextBlock text_blocks = 4;
    DocumentMetadata doc_metadata = 5;
}
```

```python
# gRPC client for Docling VLM service
class DoclingVLMClient:
    def __init__(self, grpc_endpoint: str):
        self.channel = grpc.aio.insecure_channel(grpc_endpoint)
        self.stub = DoclingVLMServiceStub(self.channel)

    async def process_pdf(self, pdf_path: str) -> DocTagsResult:
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()

        request = ProcessPDFRequest(
            pdf_content=pdf_content,
            pdf_path=pdf_path,
            config=DoclingConfig()
        )

        response = await self.stub.ProcessPDF(request)
        return response.doctags
```

**Rationale**: gRPC provides better performance, type safety, and streaming capabilities compared to HTTP. DocTags format offers superior document structure representation than MinerU output.

**Alternatives Considered**:

- HTTP REST API: Less efficient for binary data and lacks type safety
- Direct integration: Increases coupling and deployment complexity
- Multiple VLM models: Increases complexity and resource requirements

### 2. Hybrid Retrieval System Architecture

**Decision**: Implement three parallel retrieval systems (BM25 + SPLADE + Qwen3) with fusion ranking.

**Implementation Details**:

```python
class HybridRetrievalService:
    def __init__(self, bm25_index, splade_index, qwen3_index):
        self.bm25 = BM25Retriever(bm25_index)
        self.splade = SPLADERetriever(splade_index)
        self.qwen3 = Qwen3Retriever(qwen3_index)

    async def search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        # Parallel retrieval
        bm25_results = await self.bm25.search(query, k)
        splade_results = await self.splade.search(query, k)
        qwen3_results = await self.qwen3.search(query, k)

        # Fusion ranking with RRF
        fused_results = reciprocal_rank_fusion([bm25_results, splade_results, qwen3_results])
        return fused_results[:k]
```

**Rationale**: Multiple retrieval strategies provide complementary signals for improved accuracy.

**Alternatives Considered**:

- Single retrieval method: Insufficient accuracy for complex medical queries
- Learned fusion: Requires training data and increases complexity
- Sequential retrieval: Slower performance and may miss relevant results

### 3. Storage Model with Chunk Store + Separate Indexes

**Decision**: Implement chunk store database + separate index storage for reproducibility and rebuild capability.

**Implementation Details**:

```python
# Chunk store database schema
CREATE TABLE chunks (
    chunk_id UUID PRIMARY KEY,
    doc_id UUID NOT NULL,
    doctags_sha VARCHAR(64) NOT NULL,
    page_no INTEGER NOT NULL,
    bbox JSONB,  -- normalized coordinates
    element_label VARCHAR(50),  -- TITLE, PARAGRAPH, TABLE, etc.
    section_path TEXT,  -- "Introduction > Methods > Eligibility"
    char_start INTEGER,
    char_end INTEGER,
    contextualized_text TEXT,  -- for dense embeddings
    content_only_text TEXT,    -- for BM25/SPLADE
    table_payload JSONB,       -- for table rendering
    created_at TIMESTAMP DEFAULT NOW()
);

# Separate index storage with manifests
# /indexes/bm25_index/ (Lucene)
/indexes/splade_v3_repmax/ (Lucene)
/vectors/qwen3_8b_4096.faiss (FAISS)
# /manifests/ for model versions and parameters
```

**Rationale**: Separate storage allows rebuilding indexes without touching chunks, enables provenance tracking, and supports multiple retrieval strategies.

**Alternatives Considered**:

- Single monolithic index: Difficult to rebuild individual components
- Embedded storage: Less flexibility for index optimization
- Cloud-only storage: Increases operational complexity and costs

### 4. Tokenizer-Aligned Chunking

**Decision**: Implement hybrid chunker with SPLADE tokenizer alignment for consistent segmentation.

**Implementation Details**:

```python
class HybridChunker:
    def __init__(self, splade_tokenizer):
        self.tokenizer = splade_tokenizer
        self.max_tokens = 512  # SPLADE limit

    def chunk_document(self, docling_document) -> List[Chunk]:
        # Hierarchy-first: titles, sections, paragraphs, tables
        chunks = []
        for element in docling_document.body.body:
            if element.label in ['TITLE', 'SECTION_HEADER']:
                # Keep headers as separate chunks
                chunks.append(self._create_chunk(element))
            elif element.label == 'PARAGRAPH':
                # Split/merge paragraphs to fit token limits
                chunks.extend(self._split_paragraph(element))
            elif element.label == 'TABLE':
                # Keep tables as single chunks
                chunks.append(self._create_table_chunk(element))
        return chunks
```

**Rationale**: Consistent tokenization ensures SPLADE segmentation matches chunking boundaries.

**Alternatives Considered**:

- Simple text splitting: Ignores document structure and semantic coherence
- Fixed-size chunks: May split important content across boundaries
- No tokenizer alignment: Inconsistent segmentation between chunking and retrieval

### 5. SPLADE-v3 with Rep-Max Aggregation

**Decision**: Implement SPLADE-v3 with Rep-Max aggregation for learned sparse retrieval.

**Implementation Details**:

```python
class SPLADERetriever:
    def __init__(self, impact_index_path: str):
        self.index = ImpactIndexReader.open(impact_index_path)
        self.tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3")

    def encode_query(self, query: str) -> SparseVector:
        # Tokenize and encode query with SPLADE
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._extract_sparse_vector(outputs.logits)

    def search(self, query: str, k: int) -> List[RetrievalResult]:
        query_vector = self.encode_query(query)
        # Score against impact index
        scores = {}
        for term, weight in query_vector.items():
            postings = self.index.get_postings(term)
            for doc_id, impact in postings:
                scores[doc_id] = scores.get(doc_id, 0) + weight * impact
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
```

**Rationale**: Rep-Max aggregation creates one vector per chunk while preserving learned sparsity patterns.

**Alternatives Considered**:

- Score-Max aggregation: May lose important term relationships
- ExactSDM: More complex and computationally expensive
- No aggregation: Requires multiple vectors per chunk

### 6. Medical Corpus Optimizations

**Decision**: Implement medical domain-specific text normalization and terminology support.

**Implementation Details**:

```python
class MedicalTextNormalizer:
    def normalize(self, text: str) -> str:
        # Fix hyphenation at line breaks
        text = re.sub(r'-\s*\n\s*', '', text)
        # Harmonize Unicode (μ → u, etc.)
        text = unicodedata.normalize('NFKD', text)
        # Preserve units and dosages exactly
        # Add medical terminology normalization
        return text

class MedicalBM25Analyzer:
    def __init__(self):
        # Standard analyzer with medical term preservation
        self.analyzer = StandardAnalyzer()
        # Add MeSH/UMLS synonym filter (BM25-only)
        self.synonym_filter = SynonymFilter.load_mesh_synonyms()
```

**Rationale**: Medical documents require specialized handling for terminology, units, and formatting.

**Alternatives Considered**:

- Generic text processing: Insufficient for medical domain accuracy
- No normalization: Preserves original formatting but may hurt retrieval
- Aggressive normalization: May lose important medical distinctions

## Risks / Trade-offs

### Risk: VLM Model Performance vs OCR

**Risk**: Docling VLM may not achieve expected accuracy improvements on all document types.
**Mitigation**:

- Comprehensive testing with diverse medical document corpus
- Performance benchmarking against current MinerU pipeline
- Gradual rollout with monitoring and comparison
- Ability to rollback if accuracy degrades significantly

### Risk: GPU Resource Requirements

**Risk**: Docling VLM requires ~24GB VRAM, potentially straining current GPU infrastructure.
**Mitigation**:

- Implement proper GPU memory management and monitoring
- Add circuit breaker patterns for memory exhaustion
- Scale GPU resources as needed for production load
- Monitor and alert on GPU utilization

### Risk: Index Storage Complexity

**Risk**: Separate indexes increase storage complexity and operational overhead.
**Mitigation**:

- Comprehensive monitoring and health checks for all indexes
- Automated index rebuild procedures
- Manifest-based version tracking for reproducibility
- Separate storage allows rebuilding individual components

### Trade-off: Multiple Retrieval Methods vs Simplicity

**Trade-off**: Increased complexity from hybrid retrieval vs single method simplicity.
**Decision**: Accept complexity for 20-30% accuracy improvement in medical domain.

### Trade-off: Chunk Store Storage vs Query Performance

**Trade-off**: Additional storage overhead for chunk store vs direct index queries.
**Decision**: Accept storage overhead for provenance tracking and rebuild capability.

## Migration Plan

### Phase 1: gRPC Infrastructure Preparation (Week 1)

1. Create gRPC service definitions for Docling VLM
2. Set up Docling VLM gRPC service in Docker
3. Configure GPU resource allocation for 24GB VRAM
4. Implement gRPC client for Docling VLM communication

### Phase 2: Doctags Interface Design (Week 2)

1. Design new Doctags-based processing interfaces
2. Create Doctags result models and data structures
3. Implement Doctags processing pipeline
4. Add comprehensive Doctags validation and quality checks

### Phase 3: Retrieval System Implementation (Week 3-4)

1. Implement hybrid chunker with tokenizer alignment
2. Add SPLADE-v3 with Rep-Max aggregation
3. Integrate Qwen3 embeddings with FAISS storage
4. Implement structured BM25 indexing

### Phase 4: Storage and Provenance (Week 5)

1. Implement chunk store database with DuckDB
2. Create separate index storage with manifests
3. Add comprehensive provenance tracking
4. Implement medical text normalization

### Phase 5: Testing and Validation (Week 6)

1. Run comprehensive test suite for Doctags pipeline
2. Performance benchmarking with medical document corpus
3. Integration testing with new Doctags format
4. Security and compliance validation

### Phase 6: gRPC Service Integration (Week 7)

1. Integrate all GPU services with gRPC communication
2. Implement service discovery and health checks
3. Add circuit breaker patterns for service resilience
4. Monitor gRPC service performance and metrics

### Phase 7: Full Deployment (Week 8)

1. Deploy complete gRPC-based Docling VLM system
2. Monitor for any issues or performance degradation
3. Complete migration to Doctags-based processing
4. Archive all MinerU-related code and configurations

### Rollback Plan

If issues arise during rollout:

1. Disable Docling gRPC service immediately
2. Switch to backup processing pipeline
3. Investigate root cause with comprehensive logging
4. Fix issues and retry rollout
5. Maintain ability to reprocess documents with corrected pipeline

## Open Questions

1. **Model Fine-tuning**: Should we fine-tune Gemma3 12B on medical document corpus for better domain-specific accuracy?

2. **Multi-language Support**: How should we handle documents in languages not well-supported by Gemma3?

3. **Document Type Handling**: Are there specific document types (forms, invoices, research papers) that may need special handling?

4. **Performance Scaling**: How should we scale hybrid retrieval processing for high-volume scenarios? Separate GPU instances or larger batching?

5. **Cost Optimization**: What's the cost-benefit analysis of hybrid retrieval vs single method in terms of accuracy vs infrastructure costs?
