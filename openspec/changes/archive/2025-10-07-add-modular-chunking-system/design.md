# Design Document: Modular Chunking System

## Context

The Medical_KG_rev system ingests heterogeneous biomedical documents (clinical trials, research papers, drug labels, guidelines) that require domain-aware segmentation for effective retrieval. Current ingestion lacks flexible chunking, limiting retrieval accuracy and preventing multi-granularity search strategies. This design implements a modular, adapter-based chunking system supporting 20+ strategies from classical algorithms to modern semantic splitters.

### Stakeholders

- **Research teams** evaluating chunking strategies for biomedical retrieval
- **Data engineers** integrating diverse document sources
- **ML engineers** optimizing retrieval pipelines
- **Operations** managing production deployments with performance SLOs

### Constraints

- English-first optimization (no multilingual complexity)
- Must integrate with existing IR (Intermediate Representation) from MinerU
- Fail-fast on GPU unavailability for GPU-gated chunkers
- Preserve complete provenance (document offsets, section paths)
- Tables must remain atomic (never split)
- Multi-tenancy support through existing tenant_id model

## Goals / Non-Goals

### Goals

1. **Modular architecture**: Swap 20+ chunking strategies via configuration without code changes
2. **Domain-awareness**: Respect biomedical document structure (IMRaD, clinical trials, drug labels)
3. **Multi-granularity**: Support concurrent paragraph/section/window chunking for hierarchical retrieval
4. **Production + Research**: Provide both stable adapters and experimental/research tracks
5. **Framework integration**: Wrap LangChain, LlamaIndex, Unstructured, Haystack chunkers
6. **Evaluation-driven**: Built-in harness for boundary F1 and retrieval nDCG@K metrics
7. **English-optimized**: Use English sentence splitters and embedding models
8. **Provenance**: Maintain character offsets, title paths, section labels

### Non-Goals

- Multilingual chunking (English-only for phase 1)
- Multimodal content (images, formulas) beyond atomic preservation
- Real-time streaming chunking (batch processing only)
- Automatic chunker selection (manual configuration required)
- Cross-document boundary detection

## Decisions

### Decision 1: Ports & Adapters Architecture

**Rationale**: Decouple chunking algorithms from core system via `BaseChunker` interface. This enables:

- Adding new chunkers without modifying ingestion pipeline
- Swapping strategies via YAML configuration
- Unit testing chunkers in isolation
- Supporting experimental algorithms alongside production ones

**Interface**:

```python
class BaseChunker(Protocol):
    name: str
    def chunk(
        self,
        doc: Document,
        blocks: Iterable[Block],
        tables: Iterable[Table],
        *,
        granularity: Granularity | None = None
    ) -> List[Chunk]: ...

    def explain(self) -> dict: ...  # Debug info for evaluation
```

**Alternatives considered**:

- **Monolithic chunker with strategy pattern**: Rejected due to poor extensibility and tight coupling
- **Plugin system with dynamic loading**: Rejected as overkill; static registry sufficient

**Trade-offs**:

- **Pro**: Maximum flexibility, testability, framework integration ease
- **Con**: Slight overhead from adapter layer (negligible vs chunking computation)

### Decision 2: Multi-Granularity via Concurrent Execution

**Rationale**: Different retrieval scenarios benefit from different chunk sizes:

- **Paragraph-level** (~250-600 tokens): Precise answers, QA
- **Section-level** (~800-1200 tokens): Context, orientation
- **Window-level** (~512 tokens, overlapping): Micro-granularity for dense corpora
- **Document-level**: Entire document for metadata retrieval

Run multiple chunkers in parallel, tag outputs with `granularity`, and fuse at retrieval time.

**Implementation**:

```python
class MultiGranularityPipeline:
    def __init__(self, primary: BaseChunker, auxiliaries: List[BaseChunker]):
        self.primary = primary
        self.auxiliaries = auxiliaries

    async def chunk(self, doc, blocks, tables) -> List[Chunk]:
        tasks = [
            asyncio.create_task(self.primary.chunk(doc, blocks, tables, granularity="paragraph")),
            *[asyncio.create_task(aux.chunk(doc, blocks, tables, granularity=gran))
              for aux, gran in zip(self.auxiliaries, ["section", "window"])]
        ]
        results = await asyncio.gather(*tasks)
        return self._assign_chunk_ids(results)  # Namespace by chunker + granularity
```

**Alternatives considered**:

- **Sequential multi-pass**: Rejected due to latency (parallel execution 3-5x faster)
- **Single configurable chunker**: Rejected as insufficiently flexible

**Trade-offs**:

- **Pro**: Simultaneous indexing at multiple granularities, superior retrieval recall
- **Con**: 2-3x storage overhead (mitigated by shared base documents)

### Decision 3: Framework Adapters for Maximum Coverage

**Rationale**: Leverage existing chunking implementations (LangChain, LlamaIndex, Haystack, Unstructured) to provide 15+ strategies without reimplementation. Wrap each framework's chunkers behind `BaseChunker`.

**LangChain Example**:

```python
class LangChainSplitterChunker(BaseChunker):
    def __init__(self, splitter_cls, **kwargs):
        self.splitter = splitter_cls(**kwargs)
        self.name = f"langchain.{splitter_cls.__name__}"

    def chunk(self, doc, blocks, tables, *, granularity=None):
        text = "\n\n".join(b.text for b in blocks)
        parts = self.splitter.split_text(text)
        return self._assemble_chunks(doc.doc_id, parts, blocks, tables, granularity)
```

**Supported frameworks**:

- **LangChain**: RecursiveCharacter, Token, Markdown, HTML, Spacy, NLTK splitters
- **LlamaIndex**: Semantic, Hierarchical, Sentence node parsers
- **Haystack**: PreProcessor with word/sentence/passage splits
- **Unstructured**: chunk_by_title, chunk_by_element layout-aware chunking

**Alternatives considered**:

- **Reimplement all algorithms**: Rejected as redundant and maintenance-heavy
- **Use only one framework**: Rejected as each has unique strengths

**Trade-offs**:

- **Pro**: 15+ chunkers with minimal code, battle-tested implementations
- **Con**: External dependencies (mitigated by optional imports)

### Decision 4: Classical + Modern + Experimental Tracks

**Rationale**: Support three tiers for different use cases:

1. **Stable/Production** (default, high reliability):
   - `SectionAwareChunker`: IMRaD/CT.gov/SPL section boundaries
   - `SlidingWindowChunker`: Fixed windows with overlap
   - `SemanticSplitterChunker`: Embedding-drift boundaries
   - `TableChunker`: Row/rowgroup/summary modes
   - `ClinicalRoleChunker`: PICO/eligibility/endpoint segmentation

2. **Framework-Integrated** (battle-tested, configurable):
   - LangChain adapters (8 splitters)
   - LlamaIndex adapters (3 parsers)
   - Haystack, Unstructured adapters

3. **Experimental/Research** (opt-in, evaluation):
   - `TextTilingChunker`, `C99Chunker`, `BayesSegChunker` (classical topic segmentation)
   - `SemanticClusterChunker`, `GraphPartitionChunker` (clustering/graphs)
   - `LLMChapteringChunker` (few-shot prompted boundaries)
   - `DiscourseSegmenterChunker` (rhetorical units)
   - `GrobidSectionChunker`, `LayoutAwareChunker`, `GraphRAGChunker` (layout/structure)

**Configuration**:

```yaml
chunker:
  primary: semantic_splitter  # Stable default
  auxiliaries: [section_aware, sliding_window]
  experimental:
    enabled: false
    algorithms: [texttiling, c99, llm_chaptering]
```

**Alternatives considered**:

- **Production-only**: Rejected as limiting research capabilities
- **All experimental by default**: Rejected as unstable for production

**Trade-offs**:

- **Pro**: Flexibility for research, stability for production, clear boundaries
- **Con**: Larger codebase (mitigated by optional imports and lazy loading)

### Decision 5: Domain-Specific Clinical Chunker

**Rationale**: Biomedical documents have unique structural requirements:

- **Clinical trials**: Eligibility criteria, interventions, outcomes should not split
- **Drug labels**: Indications, contraindications, dosing sections are semantic units
- **Research papers**: PICO elements (Population, Intervention, Comparison, Outcome) should stay together
- **Adverse events**: Event descriptions should be atomic

Implement `ClinicalRoleChunker` with lightweight rules + classifier to detect roles and cut at role switches.

**Role taxonomy**:

```python
ClinicalRole = Literal[
    "pico_population",
    "pico_intervention",
    "pico_comparison",
    "pico_outcome",
    "eligibility",
    "endpoint",
    "adverse_event",
    "dose_regimen",
    "general"
]
```

**Algorithm**:

1. Tag sentences/blocks with roles using regex rules + small classifier (e.g., fine-tuned BERT)
2. Cut boundaries when role changes
3. Enforce "endpoint + effect" and "eligibility criteria" atomic rules
4. Set `facet_type` metadata for downstream KG mapping

**Alternatives considered**:

- **Generic chunker with domain rules**: Rejected as insufficient for complex clinical boundaries
- **LLM-based role detection**: Rejected as too slow and GPU-intensive

**Trade-offs**:

- **Pro**: High precision for clinical sections, enables faceted search
- **Con**: Requires domain-specific training data (mitigated by rule-based fallbacks)

### Decision 6: Semantic Splitter with Embedding-Drift Detection

**Rationale**: Modern retrieval benefits from semantic boundaries that avoid mid-thought cuts. Use sentence embeddings to detect coherence breaks.

**Algorithm**:

1. Encode sentences with small English model (BGE-small-en, 384D)
2. Compute sliding window cosine similarity
3. Cut boundary if:
   - Cosine similarity < `tau_coh` (e.g., 0.82) AND
   - Cumulative token count >= `min_tokens` (e.g., 400)
4. Merge small tail chunks
5. Enforce hard stops at headings/tables

**Parameters**:

```yaml
semantic:
  encoder: BAAI/bge-small-en-v1.5
  tau_coh: 0.82
  delta_drift: 0.35
  target_tokens: 600
  min_tokens: 400
  gpu_semantic_checks: true
```

**GPU fail-fast**:

```python
if config.gpu_semantic_checks and not torch.cuda.is_available():
    raise RuntimeError("GPU required for semantic chunking but CUDA unavailable")
```

**Alternatives considered**:

- **CPU-only embeddings**: Allowed as fallback if `gpu_semantic_checks: false`
- **Larger embedding models**: Rejected due to latency (small models sufficient for boundaries)

**Trade-offs**:

- **Pro**: Fewer mid-thought cuts, better retrieval quality (5-10% nDCG improvement)
- **Con**: GPU requirement, 2-5x slower than rule-based (acceptable for quality gains)

### Decision 7: Configuration-Driven with Per-Source Profiles

**Rationale**: Different document sources have optimal chunking strategies:

**PMC (PubMed Central)**:

- Target: 650 tokens (longer for full-text papers)
- Strategy: `semantic_splitter` (respects narrative flow)
- Auxiliary: `section_aware` (coarse navigation)

**DailyMed (drug labels)**:

- Target: 450 tokens (denser, structured content)
- Strategy: `section_aware` (SPL sections are well-defined)
- Auxiliary: `table` (AE tables, dosing tables)

**ClinicalTrials.gov**:

- Target: 350 tokens (shorter, structured fields)
- Strategy: `clinical_role` (eligibility, endpoints)
- Auxiliary: `sliding_window` (fallback for long fields)

**Configuration**:

```yaml
profiles:
  pmc:
    primary: semantic_splitter
    auxiliaries: [section_aware]
    target_tokens: 650
  dailymed:
    primary: section_aware
    auxiliaries: [table]
    target_tokens: 450
  ctgov:
    primary: clinical_role
    auxiliaries: [sliding_window]
    target_tokens: 350
```

**Profile detection**:

```python
def detect_profile(doc: Document) -> str:
    if "pmc" in doc.source or doc.source == "openalex":
        return "pmc"
    elif doc.source == "openfda_druglabels" or doc.source == "dailymed":
        return "dailymed"
    elif doc.source == "clinicaltrials":
        return "ctgov"
    else:
        return "default"
```

**Alternatives considered**:

- **Single universal chunker**: Rejected as insufficient for diverse sources
- **Per-document configuration**: Rejected as too granular

**Trade-offs**:

- **Pro**: Optimal strategies per source, easy to tune
- **Con**: Requires profile maintenance (mitigated by clear documentation)

### Decision 8: Provenance-First Design

**Rationale**: Every chunk must be traceable to original document with exact offsets for:

- Citation generation in RAG
- Span-grounded extraction validation
- Audit trails for compliance (HIPAA, GDPR)

**Chunk model**:

```python
class Chunk(BaseModel):
    chunk_id: str  # Format: {doc_id}:{chunker}:{granularity}:{index}
    doc_id: str
    body: str  # Chunk text
    title_path: list[str]  # Breadcrumb from root to section
    section: str | None  # Current section title
    start_char: int  # Character offset in original document
    end_char: int  # Character offset (end)
    granularity: Granularity  # "window" | "paragraph" | "section" | "document" | "table"
    page_no: int | None  # PDF page number if available
    meta: dict[str, Any]  # Additional metadata (is_table, facet_type, etc.)
    created_at: datetime
    tenant_id: str
```

**Offset preservation**:

```python
def _assemble_chunks(doc_id, text_parts, blocks, tables, granularity):
    chunks = []
    current_offset = 0
    for idx, part in enumerate(text_parts):
        # Map part back to blocks to get accurate offsets
        block_spans = _find_block_spans(part, blocks)
        start_char = block_spans[0].start_char
        end_char = block_spans[-1].end_char

        chunk = Chunk(
            chunk_id=f"{doc_id}:{chunker_name}:{granularity}:{idx}",
            doc_id=doc_id,
            body=part,
            start_char=start_char,
            end_char=end_char,
            granularity=granularity,
            ...
        )
        chunks.append(chunk)
    return chunks
```

**Alternatives considered**:

- **Approximate offsets**: Rejected as insufficient for compliance
- **Store only chunk text**: Rejected as losing provenance

**Trade-offs**:

- **Pro**: Complete traceability, enables span-grounded extraction
- **Con**: Requires careful offset mapping (complexity managed by utilities)

### Decision 9: Table Atomic Preservation with Three Modes

**Rationale**: Tables in biomedical documents contain critical structured data (AE rates, baseline characteristics, outcomes) that must not split.

**Three modes**:

1. **Row mode**: Each table row → one chunk
   - Use case: Fine-grained AE retrieval by specific events
   - Prepend column headers to each row for context

2. **Rowgroup mode**: Group related rows (e.g., by arm, grade)
   - Use case: Outcome tables grouped by intervention arm
   - Preserves multi-row coherence

3. **Summary mode**: Generate table digest (structured summary)
   - Use case: Index high-level table semantics
   - Store full table in `facet_json` for downstream processing

**Configuration**:

```yaml
tables:
  mode: rowgroup  # row | rowgroup | summary
  include_header: true
  max_rows_per_chunk: 20
```

**Algorithm (rowgroup example)**:

```python
def chunk_table(table: Table, mode: str) -> List[Chunk]:
    if mode == "rowgroup":
        groups = _detect_row_groups(table)  # By arm, grade, etc.
        chunks = []
        for group in groups:
            body = _format_rowgroup(table.header, group.rows)
            chunk = Chunk(
                body=body,
                granularity="table",
                meta={
                    "is_table": True,
                    "table_id": table.id,
                    "row_indices": group.indices,
                }
            )
            chunks.append(chunk)
        return chunks
```

**Alternatives considered**:

- **Always split tables**: Rejected as losing structure
- **Always keep tables whole**: Rejected as creating oversized chunks

**Trade-offs**:

- **Pro**: Flexible table granularity, preserves structure
- **Con**: Rowgroup detection requires heuristics (mitigated by configuration)

### Decision 10: Evaluation Harness for Data-Driven Selection

**Rationale**: With 20+ chunkers, need empirical validation to select optimal strategies.

**Metrics**:

1. **Segmentation quality**: Boundary F1 vs hand-labeled gold standard
   - Precision: % of predicted boundaries that are correct
   - Recall: % of true boundaries that are detected

2. **Retrieval impact**: Recall@K, nDCG@K on biomedical QA benchmarks
   - Fix embedding model and retrieval method
   - Vary only chunking strategy
   - Measure downstream retrieval quality

3. **Efficiency**: Latency distribution, throughput

**Gold standard creation**:

- Annotate 10-20 documents per source type (PMC, DailyMed, CT.gov)
- Mark true section/paragraph boundaries
- Store in `eval/gold_standards/chunking/`

**Evaluation runner**:

```python
class ChunkingEvaluator:
    def __init__(self, gold_standards):
        self.gold = gold_standards

    def evaluate_chunker(self, chunker: BaseChunker, docs: List[Document]) -> Metrics:
        predictions = []
        for doc in docs:
            blocks = self._load_blocks(doc)
            chunks = chunker.chunk(doc, blocks, [])
            boundaries = self._extract_boundaries(chunks)
            predictions.append(boundaries)

        f1 = self._compute_boundary_f1(predictions, self.gold)
        recall_20, ndcg_10 = self._retrieval_metrics(chunks)
        latency = self._measure_latency(chunker, docs)

        return Metrics(f1=f1, recall_20=recall_20, ndcg_10=ndcg_10, latency_p50=latency.p50)
```

**Leaderboard**:

```
| Chunker            | Boundary F1 | nDCG@10 | Recall@20 | P50 Latency |
|--------------------|-------------|---------|-----------|-------------|
| semantic_splitter  | 0.87        | 0.72    | 0.91      | 245ms       |
| section_aware      | 0.82        | 0.68    | 0.88      | 12ms        |
| texttiling         | 0.79        | 0.65    | 0.85      | 95ms        |
| sliding_window     | 0.71        | 0.63    | 0.89      | 8ms         |
```

**Alternatives considered**:

- **Manual selection**: Rejected as subjective and unscalable
- **End-to-end eval only**: Rejected as insufficient for debugging

**Trade-offs**:

- **Pro**: Data-driven decisions, regression detection
- **Con**: Requires gold standard creation (one-time cost)

## Risks / Trade-offs

### Risk 1: Experimental Chunkers May Be Unstable

**Mitigation**:

- Mark clearly as experimental in registry
- Disable by default (`experimental.enabled: false`)
- Isolate in separate module (`chunking/experimental/`)
- Add try-catch wrappers with fallback to `sliding_window`

### Risk 2: Multi-Granularity Storage Overhead

**Mitigation**:

- Share base document storage (chunks reference doc_id)
- Use compression for OpenSearch/Qdrant indices
- Make multi-granularity opt-in
- Document storage multiplier (typically 2-3x vs single granularity)

### Risk 3: Framework Dependencies Increase Attack Surface

**Mitigation**:

- Pin exact versions in `pyproject.toml`
- Make framework adapters optional imports
- Audit dependencies with `pip-audit`
- Document security best practices

### Risk 4: Semantic Chunker GPU Requirement Limits Deployment

**Mitigation**:

- Support CPU fallback if `gpu_semantic_checks: false`
- Use lightweight models (BGE-small-en, 384D) to minimize GPU load
- Batch multiple documents for GPU efficiency
- Provide non-semantic alternatives (section_aware, sliding_window)

### Risk 5: Profile Detection May Misclassify Documents

**Mitigation**:

- Implement explicit profile override in API
- Log profile detection for audit
- Provide `default` profile fallback
- Allow per-document profile hints in metadata

### Risk 6: Evaluation Harness Requires Manual Gold Standards

**Mitigation**:

- Start with small gold set (10 docs per source)
- Use inter-annotator agreement to validate
- Iterate on gold standard as needed
- Consider semi-automated boundary detection for scale

### Risk 7: LLM Chunker May Be Slow and Expensive

**Mitigation**:

- Cache boundaries by (doc_id, prompt_ver)
- Make LLM chunker experimental only
- Limit to offline/batch processing
- Validate with semantic splitter before accepting

## Migration Plan

### Phase 1: Foundation (Week 1-2)

1. Implement `BaseChunker` interface and `Chunk` model
2. Create registry and factory
3. Implement 3 stable chunkers (`sliding_window`, `section_aware`, `semantic_splitter`)
4. Add basic configuration and profile support

### Phase 2: Framework Integration (Week 3)

1. Implement LangChain adapter wrapper
2. Add LlamaIndex adapter wrapper
3. Integrate Haystack and Unstructured adapters
4. Test all framework adapters with sample documents

### Phase 3: Advanced & Experimental (Week 4-5)

1. Implement classical chunkers (TextTiling, C99, BayesSeg)
2. Add clustering-based chunkers
3. Implement LLM chaptering chunker
4. Add clinical role chunker

### Phase 4: Multi-Granularity & Evaluation (Week 6)

1. Implement `MultiGranularityPipeline`
2. Create gold standard annotations
3. Build evaluation harness
4. Run benchmarks and tune parameters

### Phase 5: Integration & Testing (Week 7-8)

1. Integrate with `IngestionService`
2. Extend `RetrievalService` for multi-granularity
3. Comprehensive testing (unit, integration, performance)
4. Documentation and examples

### Rollback Plan

- Chunking is opt-in via `chunker.enabled: false` (default)
- Existing ingestion continues unchanged if chunking disabled
- Can disable specific chunkers via registry configuration
- Multi-granularity can be disabled independently

## Open Questions

1. **Should we support query-aware chunking (retrieval-time chunk refinement)?**
   - Defer to Phase 2; requires integration with retrieval service

2. **What is the optimal number of gold standard documents per source?**
   - Start with 10-20; evaluate inter-annotator agreement; expand if needed

3. **Should LLM chunker use local LLM or cloud API?**
   - Use existing vLLM infrastructure for consistency; no cloud APIs

4. **How to handle documents with mixed languages (English + figures with non-English)?**
   - Out of scope for English-first phase; document as known limitation

5. **Should we auto-tune chunker parameters per tenant?**
   - No auto-tuning in Phase 1; provide recommended defaults
   - Consider auto-tuning in future based on retrieval metrics

## References

### Academic Papers

- Hearst, M. A. (1997). TextTiling: Segmenting text into multi-paragraph subtopic passages. *Computational Linguistics*, 23(1), 33-64.
- Choi, F. Y. Y. (2000). Advances in domain independent linear text segmentation. *NAACL 2000*.
- Eisenstein, J., & Barzilay, R. (2008). Bayesian unsupervised topic segmentation. *EMNLP 2008*.
- Riedl, M., & Biemann, C. (2012). TopicTiling: A text segmentation algorithm based on LDA. *ACL 2012 Student Research Workshop*.

### Technical Documentation

- LangChain Text Splitters: <https://python.langchain.com/docs/modules/data_connection/document_transformers/>
- LlamaIndex Node Parsers: <https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/>
- Haystack PreProcessor: <https://docs.haystack.deepset.ai/docs/preprocessor>
- Unstructured Chunking: <https://unstructured-io.github.io/unstructured/chunking.html>

### Internal Documents

- `1) docs/Chunking_Approaches.md` - Comprehensive chunking methodology catalogue
- `1) docs/Modular Document Retrieval Pipeline – Design & Scaffold.pdf` - System architecture
- `openspec/project.md` - Project conventions and standards
