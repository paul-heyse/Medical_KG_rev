# Implementation Tasks: Hybrid Retrieval, Fusion Ranking & Evaluation

**Total**: 270+ tasks across 13 work streams
**Timeline**: 6 weeks (2 build, 2 integrate, 2 validate)
**Strategy**: Hard cutover for evaluation (new), additive for retrieval (enhancements)

---

## CRITICAL: Hard Cutover Strategy

**Retrieval Enhancement Approach**:

- Existing retrieval code is **ENHANCED**, not replaced
- No legacy code removal (BM25/vector search retained)
- New components **ADDED** for fusion, reranking, evaluation

**Evaluation Framework Approach**:

- Evaluation is **NEW** (no legacy evaluation exists)
- No decommissioning required for evaluation

**Codebase Growth Validation**:

- Expected: +670 lines (fusion, reranking, evaluation)
- Modified: 450 lines (hybrid coordination)
- Net impact: +220 lines (+33% retrieval code for +26% Recall)
- Acceptable growth for significant quality improvement

---

## Work Stream #0: Legacy Decommission & Deployment Readiness (Added)

**Goal**: Pivot to the production retrieval stack and ensure new evaluation assets ship with the application.

- [x] 0.1 Retire orchestration retrieval pipeline implementation
  - **Action**: Remove `Medical_KG_rev.orchestration.retrieval_pipeline` modules and associated tests.
  - **Action**: Adopt the `RetrievalService` integration as the supported retrieval path.
- [x] 0.2 Package evaluation datasets for deployment
  - **Action**: Relocate seeded YAML datasets under the Python package, update `TestSetManager` to load from package resources, and ensure package metadata includes the files.

---

## Work Stream #1: Legacy Analysis & Validation (Pre-Implementation)

**Goal**: Confirm no conflicting retrieval implementations to remove

### 1.1 Inventory Existing Retrieval Code (3 tasks)

- [x] 1.1.1 List all retrieval services
  - **Files**: `src/Medical_KG_rev/services/retrieval/*.py`
  - **Expected**: `search.py`, `opensearch.py`, `faiss_query.py`, `fusion.py` (basic)
  - **Action**: Document for enhancement (no removal)

- [x] 1.1.2 Identify retrieval API endpoints
  - **Files**: `src/Medical_KG_rev/gateway/rest/routes/retrieval.py`
  - **Expected**: `/v1/search`, `/v1/retrieve`
  - **Action**: Enhance with hybrid parameters (no removal)

- [x] 1.1.3 Document existing fusion logic (if any)
  - **Files**: `src/Medical_KG_rev/services/retrieval/fusion.py`
  - **Expected**: Basic RRF implementation or none
  - **Action**: Enhance with weighted normalization, per-component scores

### 1.2 Validate No Conflicting Implementations (2 tasks)

- [x] 1.2.1 Check for duplicate BM25 implementations
  - **Action**: Confirm only one OpenSearch BM25 path exists
  - **Expected**: Single `opensearch.py` implementation

- [x] 1.2.2 Check for ad-hoc fusion code
  - **Action**: Identify any ad-hoc result merging in gateway or orchestration
  - **Expected**: Either centralized fusion or none (ad-hoc to be replaced)

---

## Work Stream #2: Hybrid Retrieval Coordination (40 tasks)

**Goal**: Implement coordinated BM25 + SPLADE + Dense retrieval

### 2.1 Hybrid Search Coordinator (12 tasks)

- [ ] 2.1.1 Create `HybridSearchCoordinator` class
  - **File**: `src/Medical_KG_rev/services/retrieval/hybrid.py`
  - **Signature**: `async def search(query: str, k: int, components: list[str]) -> HybridSearchResult`
  - **Components**: `["bm25", "splade", "dense"]` (configurable)

- [ ] 2.1.2 Implement parallel component execution
  - **Method**: `asyncio.gather()` for concurrent BM25, SPLADE, dense searches
  - **Timeout**: 300ms per component (fail gracefully if one slow)

- [ ] 2.1.3 Add component selection logic
  - **Config**: `config/retrieval/components.yaml` (enable/disable components)
  - **Feature flags**: `enable_splade`, `enable_dense` (BM25 always enabled)

- [ ] 2.1.4 Implement result aggregation
  - **Input**: 3 ranked lists (BM25, SPLADE, dense)
  - **Output**: `HybridSearchResult` with `component_results: dict[str, list[SearchResult]]`

- [ ] 2.1.5 Add per-component score tracking
  - **Model**: `SearchResult` includes `component_scores: dict[str, float]`
  - **Example**: `{"bm25": 12.5, "splade": 8.3, "dense": 0.87}`

- [ ] 2.1.6 Handle component failures gracefully
  - **Strategy**: If SPLADE fails, continue with BM25+Dense
  - **Logging**: Warn on component failure, emit CloudEvent

- [ ] 2.1.7 Implement query preprocessing
  - **Steps**: Unicode normalization, stopword handling (optional), tokenization
  - **Consistency**: Same preprocessing for all components

- [ ] 2.1.8 Add query expansion (optional)
  - **Method**: Synonym expansion using concept catalog
  - **Config**: `enable_query_expansion=false` (default off)

- [ ] 2.1.9 Implement caching layer
  - **Key**: `hash(query + k + components + filters)`
  - **TTL**: 5 minutes (short cache for real-time updates)
  - **Backend**: Redis

- [ ] 2.1.10 Add correlation ID propagation
  - **Flow**: Gateway → Coordinator → Components
  - **Tracing**: OpenTelemetry span per component

- [ ] 2.1.11 Write unit tests for coordinator
  - **Cases**: All components enabled, partial failures, empty results
  - **Assertions**: Result structure, score tracking, graceful degradation

- [ ] 2.1.12 Write integration tests with mocked components
  - **Setup**: Mock BM25, SPLADE, Dense services
  - **Cases**: Verify parallel execution, result aggregation

### 2.2 BM25/BM25F Enhancement (8 tasks)

- [ ] 2.2.1 Update OpenSearch query template for field boosting
  - **Fields**: `title^3, section_label^2, facet_json^2, text^1`
  - **Rationale**: Prioritize titles and structured metadata

- [ ] 2.2.2 Add domain-specific analyzer
  - **Config**: Biomedical synonym filter, lowercase, stop words
  - **File**: `config/retrieval/opensearch_analyzer.json`

- [ ] 2.2.3 Implement BM25F (field-weighted BM25)
  - **Method**: Use OpenSearch multi_match with per-field boosts
  - **Validation**: Compare BM25F vs standard BM25 on test set

- [ ] 2.2.4 Add OData filter integration
  - **Filters**: `source`, `date_range`, `tenant_id`
  - **Combination**: Filter first, then BM25 ranking

- [ ] 2.2.5 Optimize BM25 parameters (k1, b)
  - **Default**: k1=1.2, b=0.75 (standard)
  - **Tuning**: Grid search on test set if needed

- [ ] 2.2.6 Add phrase query support
  - **Detection**: Queries in quotes: "diabetes treatment"
  - **Method**: OpenSearch `match_phrase` with slop=2

- [ ] 2.2.7 Implement result highlighting
  - **Output**: `highlights` field with matched terms in context
  - **Config**: Max 3 snippets per result, 150 chars per snippet

- [ ] 2.2.8 Write BM25 component tests
  - **Cases**: Simple queries, phrase queries, filters, highlighting

### 2.3 SPLADE Integration (10 tasks)

- [ ] 2.3.1 Integrate SPLADE query expansion from Proposal 2
  - **Service**: Call Pyserini wrapper for query-side expansion
  - **Config**: `enable_splade_query_expansion=false` (default off)

- [ ] 2.3.2 Implement OpenSearch rank_features query
  - **Field**: `splade_terms` (from Proposal 2)
  - **Boost**: 2.0x relative to BM25

- [ ] 2.3.3 Add SPLADE-only search method
  - **Use Case**: Testing, debugging, A/B comparison
  - **Endpoint**: Internal method `search_splade_only()`

- [ ] 2.3.4 Implement SPLADE score normalization
  - **Method**: Min-max normalization to [0, 1] for fusion compatibility
  - **Cache**: Normalize per query, not per document

- [ ] 2.3.5 Add SPLADE failure fallback
  - **Strategy**: If SPLADE query fails, fall back to BM25 only
  - **Logging**: Log warning, emit CloudEvent

- [ ] 2.3.6 Optimize SPLADE top_k parameter
  - **Default**: top_k=100 for query expansion
  - **Tuning**: Test top_k=50, 100, 200 on test set

- [ ] 2.3.7 Implement SPLADE caching
  - **Key**: `hash(query + "splade")`
  - **TTL**: 30 minutes (query expansions stable)

- [ ] 2.3.8 Add SPLADE telemetry
  - **Metrics**: Query expansion time, term count, top term weights
  - **Tracing**: OpenTelemetry span for expansion

- [ ] 2.3.9 Write SPLADE component tests
  - **Cases**: Query expansion, rank_features query, fallback

- [ ] 2.3.10 Integration test: BM25 + SPLADE fusion
  - **Setup**: Real OpenSearch with rank_features indexed
  - **Validation**: Verify score boosting, ranking changes

### 2.4 Dense Vector KNN Integration (10 tasks)

- [ ] 2.4.1 Integrate FAISS KNN from Proposal 2
  - **Service**: Call FAISS index search method
  - **Parameters**: k=100 (retrieve more candidates for fusion)

- [ ] 2.4.2 Implement query embedding
  - **Service**: Call vLLM embedding service (Proposal 2)
  - **Namespace**: Use same namespace as indexed chunks

- [ ] 2.4.3 Add dense-only search method
  - **Use Case**: Testing, debugging, semantic-only queries
  - **Endpoint**: Internal method `search_dense_only()`

- [ ] 2.4.4 Implement distance-to-score conversion
  - **Method**: Convert L2 distance to similarity score (1 - distance/max_distance)
  - **Normalization**: Scale to [0, 1] for fusion

- [ ] 2.4.5 Add GPU health check before KNN
  - **Check**: Verify FAISS GPU index available
  - **Fallback**: If GPU unavailable, use CPU FAISS (degraded mode)

- [ ] 2.4.6 Implement dense vector caching
  - **Key**: `hash(query + "dense" + namespace)`
  - **TTL**: 15 minutes

- [ ] 2.4.7 Add dense search timeout
  - **Timeout**: 200ms (FAISS should be <50ms, allow buffer)
  - **Fallback**: If timeout, exclude dense from fusion

- [ ] 2.4.8 Implement batch query embedding
  - **Use Case**: Multiple queries in single API call
  - **Optimization**: Batch embed queries for efficiency

- [ ] 2.4.9 Write dense component tests
  - **Cases**: Single query, batch queries, GPU unavailable fallback

- [ ] 2.4.10 Integration test: BM25 + Dense fusion
  - **Setup**: Real FAISS index + OpenSearch
  - **Validation**: Verify semantic ranking, fusion behavior

---

## Work Stream #3: Fusion Ranking (35 tasks)

**Goal**: Implement RRF and weighted normalization fusion methods

### 3.1 Reciprocal Rank Fusion (RRF) (10 tasks)

- [ ] 3.1.1 Implement RRF algorithm
  - **File**: `src/Medical_KG_rev/services/retrieval/fusion/rrf.py`
  - **Formula**: `score = Σ(1 / (rank_i + k))` where k=60
  - **Input**: Multiple ranked lists (BM25, SPLADE, dense)

- [ ] 3.1.2 Add configurable k parameter
  - **Config**: `config/retrieval/fusion.yaml` → `rrf_k: 60`
  - **Tuning**: Test k=30, 60, 100 on test set

- [ ] 3.1.3 Implement result deduplication
  - **Method**: Merge results by document ID, sum RRF scores
  - **Preserve**: Per-component scores for explainability

- [ ] 3.1.4 Add tie-breaking logic
  - **Rule**: If RRF scores equal, use original rank from primary component (BM25)

- [ ] 3.1.5 Implement top-K selection
  - **Output**: Return top-K results after RRF fusion
  - **Efficiency**: Use heap for O(n log k) complexity

- [ ] 3.1.6 Add RRF score normalization (optional)
  - **Method**: Min-max normalize RRF scores to [0, 1]
  - **Use Case**: When downstream needs absolute scores

- [ ] 3.1.7 Write RRF unit tests
  - **Cases**: 2 components, 3 components, empty lists, ties

- [ ] 3.1.8 Write RRF property tests
  - **Properties**: Symmetric (order-independent), stable (repeatable)

- [ ] 3.1.9 Benchmark RRF performance
  - **Metric**: Fusion time for 100 results per component
  - **Target**: <5ms

- [ ] 3.1.10 Integration test: RRF on real data
  - **Setup**: BM25 + SPLADE + Dense results
  - **Validation**: Verify ranking stability across paraphrases

### 3.2 Weighted Normalization Fusion (10 tasks)

- [ ] 3.2.1 Implement score normalization
  - **Method**: Min-max per component, then weighted sum
  - **Formula**: `score = w_bm25·norm(s_bm25) + w_splade·norm(s_splade) + w_dense·norm(s_dense)`

- [ ] 3.2.2 Add configurable weights
  - **Config**: `config/retrieval/fusion.yaml` → `weights: {bm25: 0.3, splade: 0.4, dense: 0.3}`
  - **Default**: Equal weights (1/3 each)

- [ ] 3.2.3 Implement weight learning (optional)
  - **Method**: Grid search on test set to find optimal weights
  - **Metric**: Maximize nDCG@10

- [ ] 3.2.4 Add per-query-type weights (advanced)
  - **Config**: Different weights for `exact_term`, `paraphrase`, `complex` queries
  - **Detection**: Classify query type before fusion

- [ ] 3.2.5 Implement min-max normalization
  - **Per Component**: Normalize BM25, SPLADE, dense independently
  - **Handle Edge Cases**: All scores equal (avoid division by zero)

- [ ] 3.2.6 Add z-score normalization (alternative)
  - **Method**: Standardize scores: (s - mean) / std
  - **Use Case**: When score distributions known

- [ ] 3.2.7 Implement weighted deduplication
  - **Method**: Sum weighted scores for duplicate document IDs

- [ ] 3.2.8 Write weighted fusion unit tests
  - **Cases**: Equal weights, biased weights, edge cases

- [ ] 3.2.9 A/B test: RRF vs Weighted
  - **Setup**: 50-query test set, compare nDCG@10
  - **Decision**: Choose method with higher nDCG

- [ ] 3.2.10 Integration test: Weighted fusion on real data
  - **Validation**: Verify weight impact on ranking

### 3.3 Fusion Configuration & Selection (8 tasks)

- [ ] 3.3.1 Create fusion method registry
  - **Methods**: `rrf`, `weighted`, `borda_count` (future)
  - **Default**: `rrf`

- [ ] 3.3.2 Add API parameter for fusion method selection
  - **Endpoint**: `/v1/search?fusion_method=rrf`
  - **Validation**: Check method exists in registry

- [ ] 3.3.3 Implement fusion method factory
  - **Pattern**: Strategy pattern for pluggable fusion methods
  - **Interface**: `FusionMethod.fuse(results: list[list[SearchResult]]) -> list[SearchResult]`

- [ ] 3.3.4 Add fusion logging
  - **Metrics**: Fusion method used, component count, fusion time
  - **Tracing**: OpenTelemetry span for fusion

- [ ] 3.3.5 Implement fusion cache
  - **Key**: `hash(component_results + fusion_method + weights)`
  - **TTL**: 10 minutes

- [ ] 3.3.6 Add fusion explainability
  - **Output**: `fusion_metadata` field with method, weights, per-component contributions

- [ ] 3.3.7 Write fusion selection tests
  - **Cases**: RRF selected, weighted selected, invalid method

- [ ] 3.3.8 Integration test: Switch fusion methods dynamically
  - **Validation**: Same query with RRF vs Weighted returns different rankings

### 3.4 Per-Component Score Tracking (7 tasks)

- [ ] 3.4.1 Extend `SearchResult` model
  - **New Field**: `component_scores: dict[str, float]`
  - **Example**: `{"bm25": 12.5, "splade": 8.3, "dense": 0.87}`

- [ ] 3.4.2 Propagate component scores through fusion
  - **Preserve**: Original scores from BM25, SPLADE, dense
  - **Add**: Fused score as `component_scores["fused"]`

- [ ] 3.4.3 Implement score aggregation for duplicates
  - **Method**: When same doc appears in multiple components, retain all scores

- [ ] 3.4.4 Add component score serialization
  - **REST API**: Include in JSON response
  - **GraphQL**: Add `componentScores` field to SearchResult type

- [ ] 3.4.5 Implement component score filtering
  - **Use Case**: Show only results with high dense score (semantic filtering)
  - **API**: `/v1/search?min_dense_score=0.8`

- [ ] 3.4.6 Write component score tests
  - **Cases**: Single component, multi-component, deduplication

- [ ] 3.4.7 Integration test: Component scores match raw queries
  - **Validation**: Component scores equal to direct BM25/SPLADE/Dense queries

---

## Work Stream #4: Optional Reranking (30 tasks)

**Goal**: Implement cross-encoder reranking for high-precision scenarios

### 4.1 Cross-Encoder Service (12 tasks)

- [ ] 4.1.1 Create `CrossEncoderReranker` service
  - **File**: `src/Medical_KG_rev/services/retrieval/rerank/cross_encoder.py`
  - **Model**: BGE-reranker-base or ms-marco-MiniLM-L-12-v2

- [ ] 4.1.2 Implement model loading
  - **Library**: `sentence-transformers` CrossEncoder
  - **GPU Check**: Fail if GPU unavailable (GPU-only policy)

- [ ] 4.1.3 Add batch reranking method
  - **Signature**: `async def rerank(query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]`
  - **Batch Size**: 32 pairs (query, doc) per batch

- [ ] 4.1.4 Implement pairwise scoring
  - **Input**: (query, document) pairs
  - **Output**: Relevance scores (0-1 scale)

- [ ] 4.1.5 Add score normalization
  - **Method**: Sigmoid or softmax for cross-encoder logits
  - **Output**: Comparable scores across batches

- [ ] 4.1.6 Implement top-K selection after reranking
  - **Strategy**: Rerank top-100, return top-10
  - **Efficiency**: Only rerank what's needed

- [ ] 4.1.7 Add reranking timeout
  - **Timeout**: 200ms for 100 pairs
  - **Fallback**: If timeout, return original ranking

- [ ] 4.1.8 Implement result cache
  - **Key**: `hash(query + result_ids + model)`
  - **TTL**: 30 minutes

- [ ] 4.1.9 Add GPU health check
  - **Check**: Verify GPU available before reranking
  - **Fallback**: Skip reranking if GPU unavailable

- [ ] 4.1.10 Implement telemetry
  - **Metrics**: Reranking time, GPU utilization, score distribution
  - **Tracing**: OpenTelemetry span

- [ ] 4.1.11 Write reranker unit tests
  - **Cases**: Single batch, multiple batches, timeout

- [ ] 4.1.12 Write reranker integration tests
  - **Setup**: Real model, mock results
  - **Validation**: Verify ranking changes, score distribution

### 4.2 Reranking Integration (10 tasks)

- [ ] 4.2.1 Add reranking feature flag
  - **Config**: `config/retrieval/reranking.yaml` → `enabled: false`
  - **API**: `/v1/search?rerank=true` (opt-in)

- [ ] 4.2.2 Integrate reranker into hybrid search
  - **Flow**: Hybrid → Fusion → (Optional) Rerank → Return
  - **Conditional**: Only rerank if flag enabled

- [ ] 4.2.3 Add reranking metadata to response
  - **Field**: `reranked: true/false`
  - **Model**: Include reranker model name

- [ ] 4.2.4 Implement reranking threshold
  - **Config**: Only rerank if hybrid score > threshold (avoid reranking low-quality results)
  - **Default**: Rerank all top-100

- [x] 4.2.5 Add per-tenant reranking settings
  - **Config**: Some tenants enable reranking by default
  - **Override**: API parameter overrides tenant default

- [x] 4.2.6 Implement reranking fallback
  - **Strategy**: If reranker fails, return fusion ranking
  - **Logging**: Emit warning, CloudEvent

- [x] 4.2.7 Add reranking A/B testing support
  - **Traffic Split**: 10% reranked, 90% fusion-only
  - **Metrics**: Compare nDCG@10 across groups

- [x] 4.2.8 Write reranking integration tests
  - **Cases**: Enabled, disabled, fallback

- [x] 4.2.9 Performance benchmark: Reranking latency
  - **Metric**: P95 latency for reranking top-100
  - **Target**: <150ms

- [x] 4.2.10 A/B test: Reranking impact on nDCG@10
  - **Setup**: 50-query test set, compare with/without reranking
  - **Decision**: Enable if +5% nDCG improvement

### 4.3 Reranking Models & Configuration (8 tasks)

- [x] 4.3.1 Download BGE-reranker-base model
  - **Source**: HuggingFace `BAAI/bge-reranker-base`
  - **Cache**: Store in model cache directory

- [x] 4.3.2 Test alternative reranker models
  - **Models**: ms-marco-MiniLM-L-12-v2, colbert-reranker
  - **Metric**: Compare nDCG@10 on test set

- [x] 4.3.3 Create reranker model registry
  - **Config**: `config/retrieval/reranking_models.yaml`
  - **Default**: BGE-reranker-base

- [x] 4.3.4 Add model selection API parameter
  - **Endpoint**: `/v1/search?rerank=true&rerank_model=bge-reranker-base`

- [x] 4.3.5 Implement model caching
  - **Strategy**: Load model once on startup, cache in memory
  - **GPU**: Keep model on GPU for fast inference

- [x] 4.3.6 Add model versioning
  - **Track**: Model version in response metadata
  - **Migration**: Support multiple model versions simultaneously

- [x] 4.3.7 Write model loading tests
  - **Cases**: Valid model, invalid model, GPU unavailable

- [x] 4.3.8 Document reranking model selection guide
  - **Guide**: When to use BGE vs ms-marco vs colbert
  - **Trade-offs**: Latency vs quality

---

## Work Stream #5: Table-Aware Routing (20 tasks)

**Goal**: Prioritize table chunks for tabular queries

### 5.1 Tabular Query Detection (8 tasks)

- [x] 5.1.1 Create query intent classifier
  - **File**: `src/Medical_KG_rev/services/retrieval/routing/intent_classifier.py`
  - **Method**: Rule-based or simple ML classifier

- [x] 5.1.2 Define tabular query patterns
  - **Keywords**: "adverse events", "effect sizes", "outcome measures", "results table"
  - **Regex**: Match clinical trial registry terminology

- [x] 5.1.3 Implement keyword matching
  - **Method**: Check query for tabular keywords
  - **Output**: `is_tabular_query: bool`

- [x] 5.1.4 Add query intent enumeration
  - **Enum**: `QueryIntent.TABULAR`, `QueryIntent.NARRATIVE`, `QueryIntent.MIXED`
  - **Default**: `NARRATIVE`

- [x] 5.1.5 Implement confidence scoring
  - **Output**: `tabular_confidence: float` (0-1 scale)
  - **Use**: Higher confidence → stronger boosting

- [x] 5.1.6 Add manual intent override
  - **API**: `/v1/search?query_intent=tabular`
  - **Use Case**: User explicitly wants tabular results

- [x] 5.1.7 Write intent classifier tests
  - **Cases**: Clear tabular, clear narrative, ambiguous

- [x] 5.1.8 Benchmark classifier accuracy
  - **Test Set**: 100 queries labeled by domain experts
  - **Target**: >85% accuracy

- [x] 5.2.1 Identify table chunks in index
  - **Field**: `is_unparsed_table=true` or `intent_hint="ae"`
  - **Source**: From Proposal 1 chunking

- [x] 5.2.2 Implement OpenSearch boosting query
  - **Method**: Use `function_score` with `field_value_factor`
  - **Boost**: 3x for table chunks when tabular query detected

- [x] 5.2.3 Add dynamic boosting based on confidence
  - **Formula**: `boost = 1 + (2 * tabular_confidence)`
  - **Range**: 1x (no boost) to 3x (high confidence)

- [x] 5.2.4 Preserve table HTML in results
  - **Field**: `table_html` (from Proposal 1)
  - **Use**: Frontend rendering of structured tables

- [x] 5.2.5 Add table metadata to results
  - **Fields**: `is_table: bool`, `table_type: str` (ae, outcomes, demographics)
  - **Display**: Show table icon in UI

- [x] 5.2.6 Implement table-only search mode
  - **API**: `/v1/search?table_only=true`
  - **Use Case**: "Show me all adverse event tables"

- [x] 5.2.7 Add table ranking heuristics
  - **Rules**: Prioritize tables with more rows, complete data

- [x] 5.2.8 Implement fallback for no tables
  - **Strategy**: If no tables found, return narrative results
  - **Logging**: Log "no tables found for tabular query"

- [x] 5.2.9 Write table boosting tests
  - **Cases**: Tabular query with tables, without tables

- [ ] 5.2.10 Integration test: Table routing end-to-end
  - **Query**: "adverse events for pembrolizumab"
  - **Expected**: Table chunks ranked first

- [ ] 5.2.11 A/B test: Table routing impact
  - **Metric**: User clicks on first result (CTR)
  - **Comparison**: With vs without table routing

- [ ] 5.2.12 Document table routing behavior
  - **Guide**: When table routing activates, how to disable

---

## Work Stream #6: Clinical Intent Boosting (25 tasks)

**Goal**: Boost sections and intent hints based on query analysis

### 6.1 Clinical Intent Detection (10 tasks)

- [ ] 6.1.1 Create clinical intent analyzer
  - **File**: `src/Medical_KG_rev/services/retrieval/boosting/clinical_intent.py`
  - **Input**: Query text
  - **Output**: `ClinicalIntent` enum

- [ ] 6.1.2 Define clinical intent taxonomy
  - **Intents**: `ELIGIBILITY`, `ADVERSE_EVENTS`, `RESULTS`, `METHODS`, `DOSAGE`, `INDICATIONS`
  - **Source**: IMRaD structure, SPL LOINC sections

- [ ] 6.1.3 Implement keyword-based detection
  - **Mapping**: Keywords → Intent
  - **Example**: "eligibility" → `ELIGIBILITY`, "side effects" → `ADVERSE_EVENTS`

- [ ] 6.1.4 Add medical entity recognition (optional)
  - **Method**: Use scispaCy NER for drugs, diseases
  - **Boost**: Drug queries → prioritize `DOSAGE`, `INDICATIONS`

- [ ] 6.1.5 Implement multi-intent support
  - **Output**: List of intents with confidence scores
  - **Example**: Query may have both `RESULTS` and `ADVERSE_EVENTS` intent

- [ ] 6.1.6 Add confidence thresholding
  - **Rule**: Only apply boosting if confidence > 0.6
  - **Avoid**: Over-boosting ambiguous queries

- [ ] 6.1.7 Write intent detection tests
  - **Cases**: Clear intent, ambiguous, no intent

- [ ] 6.1.8 Benchmark detection accuracy
  - **Test Set**: 100 queries labeled by clinicians
  - **Target**: >80% accuracy

- [ ] 6.1.9 Add manual intent override
  - **API**: `/v1/search?clinical_intent=eligibility`

- [ ] 6.1.10 Document intent detection logic
  - **Guide**: Keywords, entities, edge cases

### 6.2 Section-Aware Boosting (15 tasks)

- [ ] 6.2.1 Map intents to section labels
  - **Mapping**: `ELIGIBILITY` → boost `section_label="Eligibility Criteria"`
  - **Source**: IMRaD sections, LOINC codes

- [ ] 6.2.2 Implement OpenSearch field boosting
  - **Method**: Use `multi_match` with per-field boosts
  - **Dynamic**: Boost values depend on detected intent

- [ ] 6.2.3 Define boosting rules
  - **Rule 1**: `ELIGIBILITY` intent → boost eligibility sections by 3x
  - **Rule 2**: `ADVERSE_EVENTS` intent → boost AE sections by 2x
  - **Rule 3**: `RESULTS` intent → boost Results sections by 2x

- [ ] 6.2.4 Add intent-hint boosting
  - **Field**: `intent_hint` from Proposal 1
  - **Boost**: Same as section boosting

- [ ] 6.2.5 Implement combined section + intent boosting
  - **Strategy**: Multiplicative boost for matching section AND intent hint
  - **Example**: `section_label="Results"` + `intent_hint="outcome"` → 4x boost

- [ ] 6.2.6 Add configurable boost multipliers
  - **Config**: `config/retrieval/boosting.yaml`
  - **Tuning**: Adjust multipliers based on user feedback

- [ ] 6.2.7 Implement fallback for no matching sections
  - **Strategy**: If no sections match intent, apply uniform ranking
  - **Logging**: Log "no matching sections for intent X"

- [ ] 6.2.8 Add boost decay for low-confidence intents
  - **Formula**: `boost = base_boost * intent_confidence`
  - **Example**: 50% confidence → 1.5x instead of 3x

- [ ] 6.2.9 Write boosting rule tests
  - **Cases**: Each intent, combined intents, no intent

- [ ] 6.2.10 Integration test: Section boosting end-to-end
  - **Query**: "eligibility criteria for cancer trials"
  - **Expected**: Eligibility sections ranked first

- [ ] 6.2.11 A/B test: Clinical boosting impact
  - **Metric**: nDCG@10 on clinical queries
  - **Comparison**: With vs without boosting

- [ ] 6.2.12 Domain expert validation
  - **Process**: Clinicians review top-10 results for 20 queries
  - **Feedback**: Adjust boosting rules based on feedback

- [ ] 6.2.13 Document boosting rules
  - **Guide**: Intent-to-section mapping, multipliers, when to disable

- [ ] 6.2.14 Add boosting explainability
  - **Output**: `boost_metadata` field with applied boosts

- [ ] 6.2.15 Implement boosting override
  - **API**: `/v1/search?disable_clinical_boosting=true`
  - **Use Case**: Testing, comparing with/without boosting

---

## Work Stream #7: Evaluation Framework (45 tasks)

**Goal**: Systematic retrieval quality measurement with Recall@K, nDCG, MRR

### 7.1 Metrics Implementation (15 tasks)

- [x] 7.1.1 Create metrics module
  - **File**: `src/Medical_KG_rev/services/evaluation/metrics.py`
  - **Functions**: `recall_at_k`, `ndcg_at_k`, `mrr`

- [x] 7.1.2 Implement Recall@K
  - **Formula**: `Recall@K = |relevant ∩ retrieved_top_K| / |relevant|`
  - **K Values**: 5, 10, 20

- [x] 7.1.3 Implement nDCG@K
  - **Formula**: Normalized Discounted Cumulative Gain
  - **Library**: Use scikit-learn `ndcg_score`

- [x] 7.1.4 Implement MRR
  - **Formula**: Mean Reciprocal Rank = (1/N) Σ(1/rank_i)
  - **Use Case**: Position of first relevant result

- [x] 7.1.5 Add graded relevance support for nDCG
  - **Levels**: 0 (irrelevant), 1 (somewhat), 2 (relevant), 3 (highly relevant)
  - **Source**: Manual labels from domain experts

- [x] 7.1.6 Implement Precision@K (bonus)
  - **Formula**: `Precision@K = |relevant ∩ retrieved_top_K| / K`

- [x] 7.1.7 Implement MAP (Mean Average Precision)
  - **Use Case**: Overall ranking quality metric

- [x] 7.1.8 Add per-query metric calculation
  - **Output**: Metrics for each query in test set

- [x] 7.1.9 Implement aggregate metrics
  - **Output**: Mean, median, std dev across all queries

- [x] 7.1.10 Add confidence intervals
  - **Method**: Bootstrap confidence intervals for metrics
  - **Output**: 95% CI for Recall@10, nDCG@10

- [x] 7.1.11 Write metrics unit tests
  - **Cases**: Known inputs, edge cases (empty results)

- [x] 7.1.12 Validate metrics implementation
  - **Compare**: Against reference implementations (TREC eval)

- [ ] 7.1.13 Benchmark metrics computation time
  - **Target**: <10ms for 50-query test set

- [ ] 7.1.14 Document metrics formulas
  - **Guide**: When to use each metric, interpretation

- [ ] 7.1.15 Add metrics visualization
  - **Output**: Matplotlib charts for Recall@K curves

### 7.2 Test Set Management (15 tasks)

- [x] 7.2.1 Create test set storage
  - **File**: `src/Medical_KG_rev/services/evaluation/test_sets.py`
  - **Format**: JSON with queries, relevant docs, graded labels

- [x] 7.2.2 Define test set schema
  - **Fields**: `query_id`, `query_text`, `query_type`, `relevant_docs: list[{doc_id, grade}]`

- [ ] 7.2.3 Create initial test set (50 queries)
  - **Stratification**: 20 exact term, 15 paraphrase, 15 complex clinical
  - **Labeling**: Manual labels by 2 domain experts

- [x] 7.2.4 Implement test set loader
  - **Method**: `load_test_set(name: str) -> TestSet`
  - **Validation**: Check schema, required fields

- [x] 7.2.5 Add test set versioning
  - **Format**: `test_set_v1.json`, `test_set_v2.json`
  - **Tracking**: Track which version used in evaluation

- [x] 7.2.6 Implement query type stratification
  - **Types**: `exact_term`, `paraphrase`, `complex_clinical`
  - **Analysis**: Compare metrics per query type

- [x] 7.2.7 Add relevance judgment validation
  - **Check**: All queries have ≥1 relevant doc
  - **Check**: Graded labels in valid range (0-3)

- [x] 7.2.8 Implement inter-annotator agreement
  - **Metric**: Cohen's kappa for 2 annotators
  - **Target**: κ > 0.6 (substantial agreement)

- [x] 7.2.9 Create test set refresh process
  - **Frequency**: Quarterly refresh with new queries
  - **Validation**: Ensure no query drift (overfitting)

- [x] 7.2.10 Add test set export/import
  - **Format**: JSON for portability
  - **Use Case**: Share with collaborators

- [x] 7.2.11 Implement test set splitting
  - **Splits**: 80% evaluation, 20% held-out validation
  - **Use**: Prevent overfitting during tuning

- [x] 7.2.12 Write test set loading tests
  - **Cases**: Valid test set, invalid schema, missing file

- [ ] 7.2.13 Document test set creation process
  - **Guide**: How to label queries, quality criteria

- [ ] 7.2.14 Create labeling UI (optional)
  - **Tool**: Simple web UI for labeling relevance
  - **Output**: Export to test set JSON

- [ ] 7.2.15 Validate test set quality
  - **Metrics**: Query diversity, label distribution

### 7.3 Evaluation Harness (15 tasks)

- [x] 7.3.1 Create evaluation runner
  - **File**: `src/Medical_KG_rev/services/evaluation/runner.py`
  - **Method**: `evaluate(retrieval_fn, test_set) -> EvaluationResult`

- [x] 7.3.2 Implement batch evaluation
  - **Process**: Run all test set queries, collect results
  - **Metrics**: Calculate Recall@K, nDCG@K, MRR

- [ ] 7.3.3 Add per-component evaluation
  - **Analysis**: Evaluate BM25-only, SPLADE-only, Dense-only
  - **Comparison**: vs hybrid fusion

- [x] 7.3.4 Implement A/B testing framework
  - **Setup**: Compare two retrieval configurations
  - **Output**: Statistical significance test (t-test)

- [x] 7.3.5 Add evaluation caching
  - **Key**: `hash(retrieval_config + test_set_version)`
  - **Use Case**: Avoid re-running expensive evaluations

- [ ] 7.3.6 Implement evaluation reports
  - **Format**: Markdown report with tables, charts
  - **Sections**: Overall metrics, per-query-type, per-component

- [ ] 7.3.7 Add evaluation logging
  - **Output**: Log all queries, results, metrics
  - **Use Case**: Debug low-performing queries

- [x] 7.3.8 Implement CI integration
  - **Trigger**: Run evaluation on every PR
  - **Check**: Fail if Recall@10 drops >5%

- [ ] 7.3.9 Add evaluation dashboard
  - **Tool**: Grafana dashboard with metrics trends
  - **Data Source**: Prometheus metrics

- [ ] 7.3.10 Implement regression detection
  - **Alert**: If metrics drop below baseline
  - **Action**: Notify team, block deployment

- [x] 7.3.11 Write evaluation harness tests
  - **Cases**: Small test set, A/B comparison

- [ ] 7.3.12 Benchmark evaluation time
  - **Target**: <30 seconds for 50-query test set

- [ ] 7.3.13 Document evaluation workflow
  - **Guide**: How to run, interpret results, add queries

- [x] 7.3.14 Add evaluation REST endpoint
  - **Endpoint**: `POST /v1/evaluate` with test set upload
  - **Output**: Evaluation report JSON

- [ ] 7.3.15 Implement evaluation versioning
  - **Track**: Retrieval config version, test set version
  - **Use Case**: Compare across system versions

---

## Work Stream #8: API Integration (25 tasks)

**Goal**: Expose hybrid retrieval, fusion, reranking through REST/GraphQL/gRPC

### 8.1 REST API Enhancements (10 tasks)

- [ ] 8.1.1 Update `/v1/search` endpoint
  - **Parameters**: `fusion_method`, `rerank`, `query_intent`, `table_only`
  - **Response**: Add `component_scores`, `fusion_metadata`, `reranked`

- [ ] 8.1.2 Add `/v1/search/hybrid` endpoint (explicit)
  - **Purpose**: Explicit hybrid search (vs implicit in `/v1/search`)

- [ ] 8.1.3 Create `/v1/evaluate` endpoint
  - **Method**: POST with test set JSON
  - **Response**: Evaluation metrics

- [ ] 8.1.4 Add OpenAPI schema updates
  - **Models**: `HybridSearchRequest`, `HybridSearchResponse`, `EvaluationRequest`

- [ ] 8.1.5 Implement request validation
  - **Check**: Valid fusion method, reranking model, query intent

- [ ] 8.1.6 Add response serialization
  - **Format**: JSON:API v1.1 compliance
  - **Include**: Component scores, metadata

- [ ] 8.1.7 Implement rate limiting per endpoint
  - **Limit**: 100 req/min for `/v1/search`, 10 req/min for `/v1/evaluate`

- [ ] 8.1.8 Add JWT scope requirements
  - **Scopes**: `retrieve:read` for search, `evaluate:write` for evaluation

- [ ] 8.1.9 Write REST API tests
  - **Cases**: Valid requests, invalid parameters, rate limiting

- [ ] 8.1.10 Update API documentation
  - **Docs**: Hybrid search guide, fusion methods, reranking

### 8.2 GraphQL API Enhancements (8 tasks)

- [ ] 8.2.1 Update `search` query
  - **Arguments**: `fusionMethod`, `rerank`, `queryIntent`
  - **Response**: Add `componentScores`, `fusionMetadata`

- [ ] 8.2.2 Add `hybridSearch` query (explicit)
  - **Purpose**: Explicit hybrid search with full configuration

- [ ] 8.2.3 Create `evaluate` mutation
  - **Input**: Test set data
  - **Output**: Evaluation metrics

- [ ] 8.2.4 Update GraphQL schema
  - **Types**: `HybridSearchOptions`, `ComponentScores`, `EvaluationResult`

- [ ] 8.2.5 Implement DataLoader for component scores
  - **Optimization**: Batch load component scores if needed

- [ ] 8.2.6 Write GraphQL tests
  - **Cases**: Query with options, mutation, nested fields

- [ ] 8.2.7 Update GraphQL documentation
  - **Docs**: Schema changes, new fields, examples

- [ ] 8.2.8 Test GraphQL Inspector for breaking changes
  - **Tool**: Run GraphQL Inspector on schema changes
  - **Check**: No breaking changes to existing queries

### 8.3 gRPC API Enhancements (7 tasks)

- [ ] 8.3.1 Update `Search` RPC
  - **Request**: Add `fusion_method`, `rerank`, `query_intent` fields
  - **Response**: Add `component_scores`, `reranked` fields

- [ ] 8.3.2 Add `HybridSearch` RPC (explicit)
  - **Purpose**: Full hybrid search with all options

- [ ] 8.3.3 Create `Evaluate` RPC
  - **Request**: Test set proto message
  - **Response**: Evaluation metrics proto

- [ ] 8.3.4 Update proto definitions
  - **Messages**: `HybridSearchRequest`, `ComponentScores`, `EvaluationRequest`

- [ ] 8.3.5 Compile proto files
  - **Tool**: `buf generate`
  - **Check**: No breaking changes (buf breaking)

- [ ] 8.3.6 Write gRPC tests
  - **Cases**: Valid requests, error handling

- [ ] 8.3.7 Update gRPC documentation
  - **Docs**: Proto changes, usage examples

---

## Work Stream #9: Storage Configuration (15 tasks)

**Goal**: Configure OpenSearch field boosting, FAISS index optimization

### 9.1 OpenSearch Configuration (8 tasks)

- [ ] 9.1.1 Update index mapping for field boosting
  - **Fields**: `title^3, section_label^2, facet_json^2, intent_hint^2, text^1`

- [ ] 9.1.2 Configure BM25 parameters
  - **Settings**: k1=1.2, b=0.75 (or tuned values)

- [ ] 9.1.3 Add domain-specific analyzers
  - **Analyzer**: Biomedical synonyms, stopwords, lowercase

- [ ] 9.1.4 Configure `rank_features` field for SPLADE
  - **Field**: `splade_terms` (from Proposal 2)
  - **Validation**: Ensure indexed correctly

- [ ] 9.1.5 Optimize shard configuration
  - **Shards**: 3 primary, 1 replica (adjust for data size)

- [ ] 9.1.6 Configure refresh interval
  - **Setting**: `refresh_interval: 30s` (balance real-time vs performance)

- [ ] 9.1.7 Write OpenSearch configuration tests
  - **Cases**: Mapping correct, analyzers work, boosting applies

- [ ] 9.1.8 Document OpenSearch configuration
  - **Guide**: Field boosting rationale, tuning parameters

### 9.2 FAISS Index Optimization (7 tasks)

- [ ] 9.2.1 Optimize HNSW parameters
  - **M**: 32 (connections per node)
  - **efConstruction**: 200 (build quality)
  - **efSearch**: 64 (search quality)

- [ ] 9.2.2 Test IVF index for large scale
  - **Use Case**: If >10M vectors, consider IVF+HNSW
  - **Validation**: Compare latency vs HNSW-only

- [ ] 9.2.3 Implement GPU index optimization
  - **Config**: GPU memory utilization, batch sizes

- [ ] 9.2.4 Add index compression (optional)
  - **Method**: Product quantization (PQ) for storage reduction
  - **Trade-off**: 10% recall loss for 4x storage savings

- [ ] 9.2.5 Optimize index loading time
  - **Method**: Memory-mapped files for fast startup

- [ ] 9.2.6 Write FAISS configuration tests
  - **Cases**: HNSW parameters, GPU index, compression

- [ ] 9.2.7 Document FAISS tuning guide
  - **Guide**: Parameter selection, trade-offs, when to use IVF

---

## Work Stream #10: Observability & Monitoring (30 tasks)

**Goal**: Comprehensive metrics, logging, tracing for retrieval quality

### 10.1 Prometheus Metrics (10 tasks)

- [ ] 10.1.1 Add retrieval latency per component
  - **Metric**: `medicalkg_retrieval_duration_seconds{component="bm25|splade|dense"}`

- [ ] 10.1.2 Add fusion latency
  - **Metric**: `medicalkg_fusion_duration_seconds{method="rrf|weighted"}`

- [ ] 10.1.3 Add reranking latency
  - **Metric**: `medicalkg_reranking_duration_seconds{model="bge-reranker-base"}`

- [ ] 10.1.4 Track component usage
  - **Metric**: `medicalkg_retrieval_component_used_total{component}`

- [ ] 10.1.5 Track fusion method usage
  - **Metric**: `medicalkg_fusion_method_used_total{method}`

- [ ] 10.1.6 Add Recall@K metric
  - **Metric**: `medicalkg_retrieval_recall_at_k{k=5|10|20}`

- [ ] 10.1.7 Add nDCG@K metric
  - **Metric**: `medicalkg_retrieval_ndcg_at_k{k=5|10|20}`

- [ ] 10.1.8 Track query intent distribution
  - **Metric**: `medicalkg_query_intent_detected_total{intent}`

- [ ] 10.1.9 Add reranking opt-in rate
  - **Metric**: `medicalkg_reranking_enabled_total{enabled=true|false}`

- [ ] 10.1.10 Track evaluation runs
  - **Metric**: `medicalkg_evaluation_runs_total{test_set_version}`

### 10.2 CloudEvents (10 tasks)

- [ ] 10.2.1 Define retrieval lifecycle events
  - **Events**: `retrieval.started`, `retrieval.completed`, `retrieval.failed`

- [ ] 10.2.2 Add component execution events
  - **Event**: `retrieval.component.completed{component, duration, result_count}`

- [ ] 10.2.3 Add fusion events
  - **Event**: `retrieval.fusion.completed{method, duration, result_count}`

- [ ] 10.2.4 Add reranking events
  - **Event**: `retrieval.reranking.completed{model, duration, ranking_changes}`

- [ ] 10.2.5 Add evaluation events
  - **Event**: `evaluation.completed{test_set, recall@10, ndcg@10}`

- [ ] 10.2.6 Implement CloudEvents publisher
  - **Service**: Publish to Kafka topic `retrieval.events.v1`

- [ ] 10.2.7 Add correlation ID to events
  - **Flow**: Gateway → Retrieval → Components → Fusion

- [ ] 10.2.8 Implement event sampling
  - **Strategy**: Sample 10% of events for high-volume queries

- [ ] 10.2.9 Write CloudEvents tests
  - **Cases**: Event format, publishing, correlation ID

- [ ] 10.2.10 Document CloudEvents schema
  - **Docs**: Event types, fields, usage

### 10.3 Grafana Dashboards (10 tasks)

- [ ] 10.3.1 Create retrieval overview dashboard
  - **Panels**: Latency percentiles, component usage, Recall@10 trend

- [ ] 10.3.2 Add fusion method comparison panel
  - **Visualization**: RRF vs Weighted nDCG@10 over time

- [ ] 10.3.3 Add reranking impact panel
  - **Visualization**: With/without reranking comparison

- [ ] 10.3.4 Create per-component latency panel
  - **Visualization**: BM25, SPLADE, Dense latency distributions

- [ ] 10.3.5 Add query intent heatmap
  - **Visualization**: Query intent distribution over time

- [ ] 10.3.6 Create evaluation metrics trend panel
  - **Visualization**: Recall@10, nDCG@10, MRR over time

- [ ] 10.3.7 Add alerting rules
  - **Alerts**: Recall@10 <75%, Latency P95 >600ms, Component failures >5%

- [ ] 10.3.8 Implement dashboard variables
  - **Variables**: Tenant, time range, component selection

- [ ] 10.3.9 Add dashboard export/import
  - **Format**: JSON for version control

- [ ] 10.3.10 Document dashboard usage
  - **Guide**: Panel descriptions, alerting thresholds

---

## Work Stream #11: Testing & Validation (40 tasks)

**Goal**: Comprehensive unit, integration, performance tests

### 11.1 Unit Tests (15 tasks)

- [ ] 11.1.1 Test hybrid coordinator
  - **Cases**: All components, partial failures, empty results

- [ ] 11.1.2 Test RRF fusion
  - **Cases**: 2 components, 3 components, ties, empty lists

- [ ] 11.1.3 Test weighted fusion
  - **Cases**: Equal weights, biased weights, normalization

- [ ] 11.1.4 Test cross-encoder reranker
  - **Cases**: Single batch, multiple batches, timeout

- [ ] 11.1.5 Test table-aware routing
  - **Cases**: Tabular query with tables, without tables

- [ ] 11.1.6 Test clinical intent detection
  - **Cases**: Clear intent, ambiguous, no intent

- [ ] 11.1.7 Test section boosting
  - **Cases**: Each intent, combined intents, no matching sections

- [ ] 11.1.8 Test Recall@K metric
  - **Cases**: Known inputs, edge cases (empty results)

- [ ] 11.1.9 Test nDCG@K metric
  - **Cases**: Graded relevance, binary relevance

- [ ] 11.1.10 Test MRR metric
  - **Cases**: First result relevant, no relevant results

- [ ] 11.1.11 Test test set loader
  - **Cases**: Valid test set, invalid schema, missing file

- [ ] 11.1.12 Test evaluation runner
  - **Cases**: Small test set, per-component evaluation

- [ ] 11.1.13 Test per-component score tracking
  - **Cases**: Single component, multi-component, deduplication

- [ ] 11.1.14 Test query intent classifier
  - **Cases**: Tabular, narrative, ambiguous

- [ ] 11.1.15 Test clinical intent analyzer
  - **Cases**: Each intent, multi-intent, no intent

### 11.2 Integration Tests (15 tasks)

- [ ] 11.2.1 Test hybrid retrieval end-to-end
  - **Setup**: Real OpenSearch, FAISS, test query
  - **Validation**: Results from all 3 components

- [ ] 11.2.2 Test BM25 + SPLADE fusion
  - **Setup**: Real OpenSearch with rank_features indexed
  - **Validation**: Verify boosting, ranking changes

- [ ] 11.2.3 Test BM25 + Dense fusion
  - **Setup**: Real OpenSearch + FAISS
  - **Validation**: Semantic ranking improves lexical baseline

- [ ] 11.2.4 Test hybrid + reranking pipeline
  - **Setup**: Real reranker model
  - **Validation**: Ranking changes, nDCG improves

- [ ] 11.2.5 Test table routing end-to-end
  - **Query**: "adverse events for pembrolizumab"
  - **Validation**: Table chunks ranked first

- [ ] 11.2.6 Test clinical boosting end-to-end
  - **Query**: "eligibility criteria for cancer trials"
  - **Validation**: Eligibility sections ranked first

- [ ] 11.2.7 Test evaluation framework end-to-end
  - **Setup**: 10-query test set, run evaluation
  - **Validation**: Metrics calculated correctly

- [ ] 11.2.8 Test A/B testing framework
  - **Setup**: Compare RRF vs Weighted on test set
  - **Validation**: Statistical significance test runs

- [ ] 11.2.9 Test gateway API integration
  - **Endpoint**: `/v1/search?fusion_method=rrf&rerank=true`
  - **Validation**: Response includes component scores, reranked flag

- [ ] 11.2.10 Test GraphQL API integration
  - **Query**: `search(query: "diabetes", fusionMethod: RRF, rerank: true)`
  - **Validation**: Response structure matches schema

- [ ] 11.2.11 Test gRPC API integration
  - **RPC**: `HybridSearch` with options
  - **Validation**: Proto response correct

- [ ] 11.2.12 Test multi-tenant isolation
  - **Setup**: 2 tenants, same query
  - **Validation**: Results filtered by tenant_id

- [ ] 11.2.13 Test caching behavior
  - **Setup**: Same query twice
  - **Validation**: Second query uses cache (faster)

- [ ] 11.2.14 Test rate limiting
  - **Setup**: Exceed rate limit for `/v1/search`
  - **Validation**: HTTP 429 returned

- [ ] 11.2.15 Test correlation ID propagation
  - **Setup**: Send query with correlation ID
  - **Validation**: Correlation ID in all component logs

### 11.3 Performance Tests (10 tasks)

- [ ] 11.3.1 Benchmark hybrid retrieval latency
  - **Target**: P95 <500ms for hybrid + fusion
  - **Load**: 50 concurrent users, 1000 queries

- [ ] 11.3.2 Benchmark reranking latency
  - **Target**: P95 <650ms for hybrid + fusion + rerank
  - **Load**: 50 concurrent users, 1000 queries

- [ ] 11.3.3 Benchmark per-component latency
  - **Target**: BM25 P95 <100ms, SPLADE P95 <150ms, Dense P95 <50ms

- [ ] 11.3.4 Benchmark fusion latency
  - **Target**: RRF <5ms, Weighted <10ms

- [ ] 11.3.5 Benchmark evaluation time
  - **Target**: <30 seconds for 50-query test set

- [ ] 11.3.6 Load test: 100 concurrent users
  - **Duration**: 5 minutes
  - **Validation**: Latency stable, error rate <1%

- [ ] 11.3.7 Stress test: 500 concurrent users
  - **Duration**: 2 minutes
  - **Validation**: Graceful degradation, no crashes

- [ ] 11.3.8 Soak test: 24-hour continuous load
  - **Load**: 10 concurrent users
  - **Validation**: No memory leaks, stable latency

- [ ] 11.3.9 Test GPU resource usage
  - **Metric**: GPU utilization during reranking
  - **Target**: 60-80% utilization (efficient)

- [ ] 11.3.10 Test cache hit rate
  - **Metric**: Cache hit rate over 1 hour
  - **Target**: >40% hit rate for typical workload

---

## Work Stream #12: Documentation & Guides (20 tasks)

**Goal**: Comprehensive documentation for developers and users

### 12.1 User Documentation (10 tasks)

- [ ] 12.1.1 Write hybrid retrieval guide
  - **Topics**: What is hybrid search, when to use, configuration

- [ ] 12.1.2 Write fusion methods guide
  - **Topics**: RRF vs Weighted, when to use each, tuning

- [ ] 12.1.3 Write reranking guide
  - **Topics**: What is reranking, models, when to enable

- [ ] 12.1.4 Write table routing guide
  - **Topics**: Tabular queries, boosting rules, examples

- [ ] 12.1.5 Write clinical boosting guide
  - **Topics**: Intents, section boosting, examples

- [ ] 12.1.6 Write evaluation guide
  - **Topics**: Metrics, test sets, running evaluations

- [ ] 12.1.7 Write API usage guide
  - **Topics**: REST/GraphQL/gRPC examples, parameters

- [ ] 12.1.8 Create query optimization guide
  - **Topics**: Query formulation, intent specification, filters

- [ ] 12.1.9 Write troubleshooting guide
  - **Topics**: Common issues, debugging, performance tuning

- [ ] 12.1.10 Create FAQ document
  - **Topics**: When to use hybrid vs single-component, reranking cost, etc.

### 12.2 Developer Documentation (10 tasks)

- [ ] 12.2.1 Write architecture overview
  - **Topics**: Component diagram, data flow, design decisions

- [ ] 12.2.2 Document hybrid coordinator implementation
  - **Topics**: Class structure, methods, extension points

- [ ] 12.2.3 Document fusion algorithms
  - **Topics**: RRF implementation, weighted normalization, adding new methods

- [ ] 12.2.4 Document reranking integration
  - **Topics**: Cross-encoder service, model loading, optimization

- [ ] 12.2.5 Document table routing logic
  - **Topics**: Intent detection, boosting rules, customization

- [ ] 12.2.6 Document clinical boosting implementation
  - **Topics**: Intent analyzer, section mapping, configuration

- [ ] 12.2.7 Document evaluation framework
  - **Topics**: Metrics implementation, test set format, runner

- [ ] 12.2.8 Document API changes
  - **Topics**: New endpoints, parameters, response format

- [ ] 12.2.9 Create developer setup guide
  - **Topics**: Local development, testing, debugging

- [ ] 12.2.10 Document configuration files
  - **Topics**: YAML schemas, parameter descriptions

---

## Work Stream #13: Production Deployment (25 tasks)

**Goal**: Gradual rollout with validation and monitoring

### 13.1 Staging Deployment (8 tasks)

- [ ] 13.1.1 Deploy hybrid retrieval to staging
  - **Components**: Coordinator, BM25, SPLADE, Dense

- [ ] 13.1.2 Deploy fusion ranking to staging
  - **Methods**: RRF, Weighted

- [ ] 13.1.3 Deploy reranking service to staging
  - **Model**: BGE-reranker-base on GPU node

- [ ] 13.1.4 Deploy table routing to staging
  - **Config**: Tabular query detection, boosting rules

- [ ] 13.1.5 Deploy clinical boosting to staging
  - **Config**: Intent detection, section boosting

- [ ] 13.1.6 Deploy evaluation framework to staging
  - **Setup**: Test set, metrics, runner

- [ ] 13.1.7 Run smoke tests on staging
  - **Tests**: Basic queries, fusion, reranking, evaluation

- [ ] 13.1.8 Validate staging performance
  - **Metrics**: Latency, Recall@10, nDCG@10

### 13.2 Production Rollout (12 tasks)

- [ ] 13.2.1 Phase 1: Shadow traffic testing (Week 1)
  - **Setup**: Log hybrid results, don't serve
  - **Validation**: Compare with current system

- [ ] 13.2.2 Phase 2: Canary deployment (10% traffic, Week 2)
  - **Setup**: Route 10% traffic to hybrid retrieval
  - **Monitoring**: Latency, Recall@10, user feedback

- [ ] 13.2.3 Phase 3: Gradual rollout (50% traffic, Week 3)
  - **Setup**: Route 50% traffic to hybrid retrieval
  - **Validation**: No regression in key metrics

- [ ] 13.2.4 Phase 4: Full rollout (100% traffic, Week 4)
  - **Setup**: Route all traffic to hybrid retrieval
  - **Monitoring**: 48-hour intensive monitoring

- [ ] 13.2.5 Enable reranking for high-precision tenants
  - **Setup**: Opt-in reranking for specific tenants
  - **A/B Test**: Compare with/without reranking

- [ ] 13.2.6 Deploy table routing to production
  - **Gradual**: Enable for 10% traffic, then 100%

- [ ] 13.2.7 Deploy clinical boosting to production
  - **Validation**: Domain expert review of boosted results

- [ ] 13.2.8 Deploy evaluation framework to production
  - **Setup**: CI integration, regression detection

- [ ] 13.2.9 Configure Grafana dashboards for production
  - **Panels**: Latency, Recall@10, fusion usage, reranking impact

- [ ] 13.2.10 Set up alerting rules for production
  - **Alerts**: Recall@10 <75%, Latency P95 >600ms, Component failures >5%

- [ ] 13.2.11 Document production deployment
  - **Guide**: Rollout process, rollback procedures, monitoring

- [ ] 13.2.12 Conduct post-deployment review
  - **Metrics**: Before/after comparison, lessons learned

### 13.3 Post-Deployment Validation (5 tasks)

- [ ] 13.3.1 Validate Recall@10 improvement
  - **Target**: 65% → 82% (+26%)
  - **Method**: Run evaluation on production traffic

- [ ] 13.3.2 Validate nDCG@10 improvement
  - **Target**: 0.68 → 0.79 (+16%)
  - **Method**: Run evaluation on production traffic

- [ ] 13.3.3 Validate latency SLA
  - **Target**: P95 <500ms for hybrid + fusion
  - **Method**: Monitor Prometheus metrics for 7 days

- [ ] 13.3.4 Collect user feedback
  - **Method**: User surveys, qualitative feedback
  - **Metric**: User satisfaction score

- [ ] 13.3.5 Create deployment success report
  - **Sections**: Metrics before/after, issues encountered, recommendations

---

## Summary

**Total Tasks**: 270+
**Timeline**: 6 weeks
**Team Size**: 2-3 engineers
**Risk Level**: Medium (additive changes, gradual rollout)

**Key Milestones**:

- Week 1-2: Build hybrid retrieval + fusion
- Week 3-4: Add reranking + clinical boosting
- Week 5-6: Evaluation framework + production validation

**Success Criteria**:

- ✅ Recall@10: 65% → 82% (+26%)
- ✅ nDCG@10: 0.68 → 0.79 (+16%)
- ✅ Latency P95: <500ms (hybrid + fusion)
- ✅ User satisfaction: Improved (qualitative feedback)
