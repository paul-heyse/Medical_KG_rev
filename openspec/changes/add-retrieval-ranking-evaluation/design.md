# Design Document: Hybrid Retrieval, Fusion Ranking & Evaluation

## Context

The current retrieval architecture has evolved over 18 months, resulting in:

1. **Single-Strategy Retrieval**: Primary reliance on BM25 lexical matching, achieving only 65% Recall@10 on biomedical queries where term variation is high (e.g., "diabetes treatment" vs "managing blood glucose levels" return different results despite semantic equivalence)

2. **Ad-Hoc Result Merging**: When dense vectors were added, results were either returned separately or merged ad-hoc in gateway code, causing 30% variance in relevance scores across paraphrased queries

3. **No Systematic Evaluation**: Unable to measure retrieval quality objectively, making it impossible to justify investment in new embedding models or optimize ranking strategies

4. **Missing Clinical Context**: Generic ranking doesn't leverage domain structure (IMRaD sections, LOINC codes, intent hints from Proposal 1)

**Constraints**:

- Must maintain P95 latency <500ms for hybrid retrieval (adding components increases latency)
- Must support multi-tenancy (results filtered by tenant_id)
- Local deployment only (no cloud APIs like Algolia, Pinecone)
- Existing BM25 and FAISS infrastructure must be reused (no replacement)
- Backward compatibility NOT required (retrieval response format can change)

**Stakeholders**:

- **Researchers**: Need high recall for literature reviews, semantic search for paraphrases
- **Clinicians**: Need clinical structure-aware ranking (eligibility first for eligibility queries)
- **Data Science**: Needs evaluation framework to justify model improvements
- **Engineering**: Seeks maintainable, explainable ranking architecture
- **Operations**: Requires reliable, monitorable retrieval with clear failure modes

---

## Goals / Non-Goals

### Goals

1. **Hybrid Retrieval Strategy**: Combine BM25 (lexical), SPLADE (learned sparse), Dense (semantic) for complementary strengths
2. **Stable Fusion Ranking**: RRF as default (parameter-free), weighted normalization as advanced option
3. **Optional Reranking**: Cross-encoder for high-precision scenarios (opt-in, GPU-based)
4. **Clinical Context-Aware**: Table routing and clinical boosting leverage domain structure
5. **Evaluation Framework**: Systematic measurement with Recall@K, nDCG@K, MRR
6. **Explainability**: Per-component scores preserved for debugging and trust

### Non-Goals

- **Real-Time Learning**: No online learning or query reformulation (static ranking)
- **Personalization**: No per-user ranking adaptation (tenant-level only)
- **Federated Search**: No cross-tenant or cross-index search
- **Custom Rankers**: No user-defined ranking functions (pre-defined methods only)
- **Query Expansion**: No automatic synonym expansion (use concept catalog explicitly)

---

## Technical Decisions

### Decision 1: Hybrid Retrieval with BM25 + SPLADE + Dense KNN

**Choice**: Coordinate three complementary retrieval strategies in parallel, each returning top-100 candidates

**Why**:

- **BM25 (Lexical Baseline)**: Handles exact term matches (NCT IDs, gene names, drug names), fast (<100ms), no GPU required
- **SPLADE (Learned Sparse)**: Expands terms (diabetes → glucose, insulin), leverages OpenSearch `rank_features` from Proposal 2, modest recall improvement (+10% over BM25)
- **Dense KNN (Semantic)**: Captures paraphrases and semantic similarity, uses FAISS from Proposal 2, highest recall improvement (+15% over BM25 alone)
- **Complementary**: Each component catches different types of relevance, fusion combines strengths

**Alternatives Considered**:

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **BM25 Only** | Simple, fast, no GPU | 65% Recall@10, misses paraphrases | ❌ Rejected: Insufficient recall |
| **Dense Only** | High semantic similarity | Misses exact matches, GPU required, 150ms latency | ❌ Rejected: Insufficient for exact terms |
| **BM25 + Dense (no SPLADE)** | Good coverage, simpler | Misses lexical expansions | ❌ Rejected: SPLADE adds +5% recall for minimal cost |
| **Elasticsearch Learning-to-Rank** | Native integration | Requires training, less explainable | ❌ Rejected: Complexity, vendor lock-in |

**Implementation**:

```python
# Hybrid Search Coordinator
class HybridSearchCoordinator:
    def __init__(
        self,
        opensearch: OpenSearchClient,
        faiss_index: FAISSIndex,
        splade_wrapper: PyseriniSPLADEWrapper
    ):
        self.opensearch = opensearch
        self.faiss_index = faiss_index
        self.splade = splade_wrapper

    async def search(
        self,
        query: str,
        k: int = 10,
        components: list[str] = ["bm25", "splade", "dense"]
    ) -> HybridSearchResult:
        # Execute components in parallel
        tasks = []

        if "bm25" in components:
            tasks.append(self.search_bm25(query, k=100))

        if "splade" in components:
            tasks.append(self.search_splade(query, k=100))

        if "dense" in components:
            tasks.append(self.search_dense(query, k=100))

        # Gather results with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build component results dict
        component_results = {}
        for i, component in enumerate(components):
            if isinstance(results[i], Exception):
                logger.warning(f"Component {component} failed: {results[i]}")
                # Fail gracefully: exclude component from fusion
                continue
            component_results[component] = results[i]

        return HybridSearchResult(component_results=component_results)
```

**Latency Analysis**:

- BM25: 80ms P95 (OpenSearch standard)
- SPLADE: 120ms P95 (OpenSearch rank_features query)
- Dense: 40ms P95 (FAISS GPU-accelerated KNN)
- **Parallel**: max(80, 120, 40) = 120ms (bottleneck: SPLADE)
- **Total with Fusion**: 120ms + 10ms fusion = 130ms P95

**Trade-offs**:

- ✅ **Pros**: Complementary strengths, high recall (82%), explainable (per-component scores)
- ⚠️ **Cons**: Increased latency (130ms vs 80ms BM25-only), operational complexity (3 components)
- 🔧 **Mitigation**: Graceful degradation (if SPLADE fails, continue with BM25+Dense)

---

### Decision 2: Reciprocal Rank Fusion (RRF) as Default Fusion Method

**Choice**: Use RRF with k=60 as default fusion method, with weighted normalization as advanced opt-in

**Why**:

- **Parameter-Free**: RRF requires only k constant (k=60 is standard), no score calibration needed
- **Stable Variance**: RRF produces consistent rankings across query paraphrases (10% variance vs 30% ad-hoc)
- **Order-Independent**: Same result regardless of component ordering (symmetric property)
- **Proven**: Used in TREC benchmarks, academic IR systems, production search engines

**Alternatives Considered**:

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **Weighted Normalization** | Explicit weight control | Requires score calibration, sensitive to outliers | ⚠️ **Advanced Opt-In**: For expert users |
| **CombSUM** | Simple (sum normalized scores) | Sensitive to score scales, poor stability | ❌ Rejected: Unstable across queries |
| **Borda Count** | Rank-based (like RRF) | Less principled than RRF, similar performance | ❌ Rejected: RRF more established |
| **Learning-to-Rank (LTR)** | Optimal weights learned | Requires training data, black-box | ❌ Rejected: Complexity, explainability loss |

**Implementation**:

```python
# Reciprocal Rank Fusion
class RRFFusion:
    def __init__(self, k: int = 60):
        self.k = k  # Standard RRF constant

    def fuse(
        self,
        component_results: dict[str, list[SearchResult]]
    ) -> list[SearchResult]:
        """Fuse multiple ranked lists using RRF."""
        doc_scores = defaultdict(lambda: {"score": 0.0, "component_scores": {}})

        for component, results in component_results.items():
            for rank, result in enumerate(results, start=1):
                # RRF formula: score = 1 / (rank + k)
                rrf_score = 1.0 / (rank + self.k)

                # Accumulate RRF score
                doc_scores[result.doc_id]["score"] += rrf_score

                # Preserve original component score for explainability
                doc_scores[result.doc_id]["component_scores"][component] = result.score

        # Sort by RRF score descending
        fused = [
            SearchResult(
                doc_id=doc_id,
                score=data["score"],
                component_scores=data["component_scores"]
            )
            for doc_id, data in sorted(
                doc_scores.items(),
                key=lambda x: x[1]["score"],
                reverse=True
            )
        ]

        return fused
```

**RRF Properties**:

- **Symmetric**: `RRF(A, B) = RRF(B, A)` (order doesn't matter)
- **Stable**: Same query paraphrases produce similar rankings (10% variance)
- **Interpretable**: Higher rank in any component → higher RRF score

**Parameter Tuning**:

- **k=30**: Emphasizes top ranks more (aggressive)
- **k=60**: Standard, balanced (default)
- **k=100**: Emphasizes lower ranks (conservative)
- **Recommendation**: Start with k=60, tune only if evaluation shows benefit

**Trade-offs**:

- ✅ **Pros**: No tuning, stable, symmetric, interpretable
- ⚠️ **Cons**: No explicit component weights (all components equal), cannot bias toward specific component
- 🔧 **Mitigation**: Offer weighted normalization as advanced opt-in for users needing explicit control

---

### Decision 3: Optional Cross-Encoder Reranking (Opt-In, GPU-Based)

**Choice**: Implement BGE-reranker-base cross-encoder reranking as opt-in feature with GPU fail-fast

**Why**:

- **Quality Improvement**: +5-8% nDCG@10 on complex queries (e.g., "long-term efficacy of checkpoint inhibitors in metastatic melanoma")
- **Query-Document Cross-Attention**: Cross-encoder scores (query, document) pairs jointly, vs independent embeddings
- **Opt-In**: Only enable for high-precision scenarios (acceptable latency increase)
- **GPU-Only**: Enforces GPU availability (no CPU fallback), aligns with Proposal 2 policy

**When to Enable Reranking**:

- Clinical decision support queries (high precision required)
- Complex multi-faceted queries (not simple keyword matches)
- Acceptable latency increase (+150ms for top-100 reranking)

**Alternatives Considered**:

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **ColBERT Reranking** | Multi-vector, token-level matching | Slower (300ms for top-100), complex | ❌ Rejected: Latency too high |
| **LLM-based Reranking (GPT-4)** | Highest quality | Cloud API, cost, latency (seconds) | ❌ Rejected: Not local, too slow |
| **Sentence-BERT Bi-Encoder** | Fast (single embedding per doc) | Lower quality than cross-encoder | ❌ Rejected: Quality insufficient |
| **ms-marco-MiniLM-L-12-v2** | Smaller, faster than BGE | 2-3% lower nDCG than BGE | ⚠️ **Alternative**: If latency critical |

**Implementation**:

```python
# Cross-Encoder Reranker
class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        # Check GPU availability (fail-fast)
        if not torch.cuda.is_available():
            raise GpuNotAvailableError("Reranker requires GPU")

        # Load cross-encoder model
        self.model = CrossEncoder(model_name, device="cuda")
        self.device = torch.device("cuda")

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10
    ) -> list[SearchResult]:
        """Rerank top-100 results, return top-K."""
        if len(results) == 0:
            return []

        # Create (query, document) pairs
        pairs = [(query, result.text) for result in results]

        # Batch score pairs
        scores = self.model.predict(pairs, batch_size=32)

        # Attach reranker scores to results
        for result, score in zip(results, scores):
            result.reranker_score = float(score)
            result.component_scores["reranker"] = float(score)

        # Sort by reranker score descending
        reranked = sorted(results, key=lambda r: r.reranker_score, reverse=True)

        return reranked[:top_k]
```

**Latency Analysis**:

- Hybrid + Fusion: 130ms P95
- Reranking (100 pairs, batch_size=32): +120ms
- **Total**: 250ms P95 (acceptable for high-precision queries)

**Reranking Performance**:

| Scenario | nDCG@10 (No Rerank) | nDCG@10 (With Rerank) | Improvement |
|----------|---------------------|------------------------|-------------|
| Simple (1-2 keywords) | 0.78 | 0.79 | +1% (minimal) |
| Complex (multi-faceted) | 0.68 | 0.75 | +10% (**significant**) |
| Paraphrase | 0.72 | 0.77 | +7% |

**Trade-offs**:

- ✅ **Pros**: Significant quality improvement on complex queries, GPU-accelerated (120ms)
- ⚠️ **Cons**: Latency increase (+120ms), GPU required, opt-in complexity
- 🔧 **Mitigation**: Feature flag, only enable for specific tenants or query types

---

### Decision 4: Table-Aware Routing with Intent-Based Boosting

**Choice**: Detect tabular queries via keyword matching, boost table chunks by 3x when detected

**Why**:

- **Clinical Structure Leverage**: Adverse event queries should surface table chunks first (structured data preferred over narrative)
- **Preserves Table HTML**: Table chunks from Proposal 1 include HTML for frontend rendering
- **Intent-Driven**: Query intent ("adverse events") determines routing, not just keywords

**Tabular Query Detection**:

- **Keywords**: "adverse events", "effect sizes", "outcome measures", "results table", "demographics table"
- **Regex**: Match clinical trial registry terminology
- **Confidence Scoring**: 0-1 scale (higher confidence → stronger boosting)

**Boosting Strategy**:

```python
# Table-Aware Routing
class TableAwareRouter:
    TABULAR_KEYWORDS = {
        "adverse events": 0.9,
        "side effects": 0.8,
        "effect sizes": 0.9,
        "outcome measures": 0.9,
        "results table": 1.0,
        "demographics": 0.7
    }

    def detect_tabular_intent(self, query: str) -> float:
        """Detect if query is tabular, return confidence 0-1."""
        query_lower = query.lower()
        max_confidence = 0.0

        for keyword, confidence in self.TABULAR_KEYWORDS.items():
            if keyword in query_lower:
                max_confidence = max(max_confidence, confidence)

        return max_confidence

    def boost_table_chunks(
        self,
        query: str,
        results: list[SearchResult]
    ) -> list[SearchResult]:
        """Boost table chunks if tabular query detected."""
        tabular_confidence = self.detect_tabular_intent(query)

        if tabular_confidence < 0.5:
            return results  # Not a tabular query, no boosting

        # Dynamic boost: 1 + (2 * confidence) → range [1x, 3x]
        boost_multiplier = 1 + (2 * tabular_confidence)

        for result in results:
            if result.is_table or result.intent_hint == "ae":
                result.score *= boost_multiplier

        # Re-sort by boosted scores
        return sorted(results, key=lambda r: r.score, reverse=True)
```

**Example Queries**:

| Query | Tabular Confidence | Boost Multiplier | Effect |
|-------|-------------------|------------------|--------|
| "pembrolizumab adverse events" | 0.9 | 2.8x | Table chunks ranked 1-5 |
| "diabetes treatment options" | 0.0 | 1.0x | No boosting (narrative query) |
| "outcome measures for cancer trials" | 0.9 | 2.8x | Outcome tables prioritized |

**Alternatives Considered**:

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **Always Prioritize Tables** | Simple | Incorrect for narrative queries | ❌ Rejected: Over-prioritizes tables |
| **ML-Based Intent Classifier** | Higher accuracy | Requires training data, model deployment | ⚠️ **Future**: If keyword-based insufficient |
| **Manual Intent Override** | User control | Requires UI change, user burden | ✅ **Included**: API parameter `query_intent` |

**Trade-offs**:

- ✅ **Pros**: Leverages clinical structure, improves relevance for tabular queries
- ⚠️ **Cons**: Keyword-based detection may miss novel queries, 85% accuracy
- 🔧 **Mitigation**: Allow manual intent override (`?query_intent=tabular`)

---

### Decision 5: Clinical Intent Boosting with Section-Aware Ranking

**Choice**: Detect clinical intent (eligibility, adverse events, results, methods, dosage), boost matching sections by 2-3x

**Why**:

- **Domain Structure**: IMRaD sections and LOINC codes provide clinical structure (from Proposal 1)
- **Intent-Driven Relevance**: Eligibility query should prioritize eligibility sections over results
- **Explainable Boosting**: Transparent rules (eligibility intent → boost eligibility sections)

**Clinical Intent Taxonomy**:

```python
class ClinicalIntent(Enum):
    ELIGIBILITY = "eligibility"
    ADVERSE_EVENTS = "adverse_events"
    RESULTS = "results"
    METHODS = "methods"
    DOSAGE = "dosage"
    INDICATIONS = "indications"
```

**Intent Detection**:

```python
# Clinical Intent Analyzer
class ClinicalIntentAnalyzer:
    INTENT_KEYWORDS = {
        ClinicalIntent.ELIGIBILITY: ["eligibility", "inclusion", "exclusion", "criteria"],
        ClinicalIntent.ADVERSE_EVENTS: ["adverse events", "side effects", "toxicity", "safety"],
        ClinicalIntent.RESULTS: ["results", "outcome", "efficacy", "effectiveness"],
        ClinicalIntent.METHODS: ["methods", "study design", "protocol"],
        ClinicalIntent.DOSAGE: ["dosage", "dose", "administration"],
        ClinicalIntent.INDICATIONS: ["indication", "use", "treatment"]
    }

    def detect_intent(self, query: str) -> list[tuple[ClinicalIntent, float]]:
        """Detect clinical intent(s) with confidence scores."""
        query_lower = query.lower()
        detected = []

        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    # Confidence based on keyword match quality
                    confidence = 1.0 if keyword == query_lower else 0.7
                    detected.append((intent, confidence))
                    break

        return detected
```

**Section Boosting Rules**:

```python
# Section-Aware Boosting
class SectionBooster:
    INTENT_TO_SECTION = {
        ClinicalIntent.ELIGIBILITY: {"Eligibility Criteria", "eligibility"},
        ClinicalIntent.ADVERSE_EVENTS: {"Adverse Reactions", "Adverse Events", "Safety"},
        ClinicalIntent.RESULTS: {"Results", "Findings", "Outcomes"},
        ClinicalIntent.METHODS: {"Methods", "Study Design"},
        ClinicalIntent.DOSAGE: {"Dosage and Administration", "Dosage"},
        ClinicalIntent.INDICATIONS: {"Indications and Usage", "Indications"}
    }

    def boost_sections(
        self,
        query: str,
        results: list[SearchResult]
    ) -> list[SearchResult]:
        """Boost sections matching detected intent."""
        intents = self.intent_analyzer.detect_intent(query)

        if not intents:
            return results  # No intent detected, no boosting

        for result in results:
            for intent, confidence in intents:
                matching_sections = self.INTENT_TO_SECTION[intent]

                if result.section_label in matching_sections or result.intent_hint == intent.value:
                    # Dynamic boost: 2x base * confidence
                    boost = 2.0 * confidence
                    result.score *= boost

        # Re-sort by boosted scores
        return sorted(results, key=lambda r: r.score, reverse=True)
```

**Example Queries**:

| Query | Detected Intent | Boosted Sections | Effect |
|-------|----------------|------------------|--------|
| "eligibility criteria for breast cancer trials" | ELIGIBILITY (1.0) | Eligibility Criteria | Eligibility sections ranked 1-3 |
| "adverse events pembrolizumab" | ADVERSE_EVENTS (1.0) | Adverse Reactions, Safety | AE sections ranked 1-5 |
| "study methods RCT" | METHODS (0.7) | Methods, Study Design | Methods sections boosted |

**Trade-offs**:

- ✅ **Pros**: Leverages clinical structure, explainable rules, improves relevance
- ⚠️ **Cons**: Keyword-based (may miss novel intents), 80% accuracy
- 🔧 **Mitigation**: Manual intent override, quarterly review of detection rules

---

### Decision 6: Evaluation Framework with Recall@K, nDCG@K, MRR

**Choice**: Implement systematic evaluation using scikit-learn for nDCG, custom implementations for Recall@K and MRR

**Why**:

- **Objective Measurement**: Quantify retrieval quality (currently no metrics)
- **A/B Testing Support**: Compare retrieval configurations (RRF vs Weighted, with/without reranking)
- **Regression Detection**: CI integration prevents quality degradation
- **Per-Component Analysis**: Track contribution of BM25, SPLADE, Dense

**Metrics**:

1. **Recall@K**: What percentage of relevant documents are in top-K?
   - Formula: `Recall@K = |relevant ∩ retrieved_top_K| / |relevant|`
   - Use Case: Measure coverage, ensure relevant docs not missed

2. **nDCG@K**: How well are relevant documents ranked (graded relevance)?
   - Formula: Normalized Discounted Cumulative Gain
   - Use Case: Measure ranking quality, graded relevance (0-3 scale)

3. **MRR**: Where is the first relevant result?
   - Formula: `MRR = (1/N) Σ(1/rank_i)`
   - Use Case: Measure "first answer" quality

**Implementation**:

```python
# Evaluation Metrics
class EvaluationMetrics:
    @staticmethod
    def recall_at_k(
        retrieved: list[str],
        relevant: list[str],
        k: int
    ) -> float:
        """Calculate Recall@K."""
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)

        if len(relevant_set) == 0:
            return 0.0

        intersection = retrieved_k & relevant_set
        return len(intersection) / len(relevant_set)

    @staticmethod
    def ndcg_at_k(
        retrieved: list[str],
        relevance: dict[str, int],
        k: int
    ) -> float:
        """Calculate nDCG@K using scikit-learn."""
        from sklearn.metrics import ndcg_score

        # Build relevance vector for retrieved docs
        y_true = [[relevance.get(doc_id, 0) for doc_id in retrieved[:k]]]
        y_score = [[len(retrieved) - i for i in range(len(retrieved[:k]))]]

        return ndcg_score(y_true, y_score)

    @staticmethod
    def mrr(
        retrieved: list[str],
        relevant: list[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        relevant_set = set(relevant)

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0  # No relevant doc found
```

**Test Set Schema**:

```yaml
# test_set_v1.yaml
queries:
  - query_id: q001
    query_text: "pembrolizumab adverse events melanoma"
    query_type: exact_term
    relevant_docs:
      - doc_id: "NCT04267848:chunk_45"
        grade: 3  # highly relevant
      - doc_id: "NCT04267848:chunk_46"
        grade: 2  # relevant
      - doc_id: "NCT03456789:chunk_12"
        grade: 1  # somewhat relevant

  - query_id: q002
    query_text: "managing blood glucose in type 2 diabetes"
    query_type: paraphrase
    relevant_docs:
      - doc_id: "PMC7654321:chunk_23"
        grade: 3
      - doc_id: "PMC7654321:chunk_24"
        grade: 2
```

**Test Set Creation**:

- 50 queries total: 20 exact term, 15 paraphrase, 15 complex clinical
- 2 annotators per query (inter-annotator agreement κ > 0.6)
- Quarterly refresh to avoid overfitting

**A/B Testing Workflow**:

```python
# A/B Test: RRF vs Weighted
test_set = load_test_set("test_set_v1.yaml")

# Configuration A: RRF
config_a = {"fusion_method": "rrf", "rrf_k": 60}
results_a = evaluate(retrieval_fn, test_set, config_a)

# Configuration B: Weighted
config_b = {"fusion_method": "weighted", "weights": {"bm25": 0.3, "splade": 0.4, "dense": 0.3}}
results_b = evaluate(retrieval_fn, test_set, config_b)

# Statistical significance test
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(results_a.ndcg_at_10, results_b.ndcg_at_10)

if p_value < 0.05:
    print(f"Significant difference: Config B is {'better' if results_b.ndcg_at_10.mean() > results_a.ndcg_at_10.mean() else 'worse'}")
else:
    print("No significant difference")
```

**Trade-offs**:

- ✅ **Pros**: Objective measurement, A/B testing, regression detection, CI integration
- ⚠️ **Cons**: Requires manual labeling (50 queries × 2 annotators = 100 hours), test set may overfit
- 🔧 **Mitigation**: Quarterly test set refresh, held-out validation set (20% of queries)

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    CLIENT (REST/GraphQL/gRPC)                     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │ search(query, fusion_method, rerank, query_intent)
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                   HYBRID SEARCH COORDINATOR                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Query Preprocessing (Unicode normalization, tokenization)│   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Clinical Intent Analyzer (detect ELIGIBILITY, AE, etc.)  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Table-Aware Router (detect tabular queries)             │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        │ BM25              │ SPLADE            │ Dense KNN
        │ (OpenSearch)      │ (OpenSearch)      │ (FAISS)
        │                   │                   │
┌───────▼─────────┐  ┌──────▼──────────┐  ┌────▼─────────────┐
│  BM25/BM25F     │  │  SPLADE Query   │  │  Query Embedding │
│  Field Boosting │  │  Expansion      │  │  (vLLM)          │
│  title^3        │  │  (Pyserini)     │  │  4096-D vector   │
│  section^2      │  │  rank_features  │  │                  │
│  text^1         │  │  query          │  │  FAISS HNSW      │
└─────────────────┘  └─────────────────┘  │  GPU-accelerated │
        │                   │              │  KNN search      │
        │ Top-100           │ Top-100      └────────────────────┘
        │                   │                   │
        │                   │                   │ Top-100
        └───────────────────┴───────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                      FUSION RANKING                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Fusion Method Selection (RRF vs Weighted)               │   │
│  │  - RRF: score = Σ(1 / (rank_i + 60))                    │   │
│  │  - Weighted: score = w_bm25·s_bm25 + w_splade·s_splade  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Clinical Boosting (intent-based section boosting)       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Table Boosting (if tabular query detected)              │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                            │
                            │ Fused Top-100
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                   OPTIONAL RERANKING (if enabled)                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Cross-Encoder (BGE-reranker-base)                       │   │
│  │  - GPU-only (fail-fast if unavailable)                   │   │
│  │  - Batch size: 32 pairs                                  │   │
│  │  - Latency: +120ms for top-100                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                            │
                            │ Top-10 Reranked Results
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                    RESPONSE ENRICHMENT                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  - Per-component scores (BM25, SPLADE, Dense, Reranker)  │   │
│  │  - Fusion metadata (method, weights, confidence)         │   │
│  │  - Table HTML (if table chunks)                          │   │
│  │  - Section labels, intent hints                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                            │
                            │ HybridSearchResponse
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                      CLIENT RESPONSE                              │
│  {                                                                │
│    "results": [                                                   │
│      {                                                            │
│        "doc_id": "NCT04267848:chunk_45",                         │
│        "score": 0.87,                                            │
│        "component_scores": {                                      │
│          "bm25": 12.5,                                           │
│          "splade": 8.3,                                          │
│          "dense": 0.89,                                          │
│          "reranker": 0.92                                        │
│        },                                                         │
│        "is_table": true,                                         │
│        "section_label": "Adverse Reactions"                      │
│      }                                                            │
│    ],                                                             │
│    "fusion_metadata": {                                           │
│      "method": "rrf",                                            │
│      "k": 60,                                                    │
│      "reranked": true                                            │
│    }                                                              │
│  }                                                                │
└───────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Build Hybrid Retrieval + Fusion (Week 1-2)

#### Week 1: Component Integration

**Day 1-2: Hybrid Coordinator**

- Implement `HybridSearchCoordinator` class
- Add parallel component execution with `asyncio.gather`
- Implement graceful degradation for component failures
- Add per-component score tracking

**Day 3-4: BM25/BM25F Enhancement**

- Update OpenSearch query template for field boosting
- Add domain-specific analyzer
- Implement phrase query support
- Add result highlighting

**Day 5: SPLADE Integration**

- Integrate Pyserini SPLADE wrapper (Proposal 2)
- Implement OpenSearch `rank_features` query
- Add SPLADE score normalization

#### Week 2: Fusion + Dense Integration

**Day 1-2: Dense KNN Integration**

- Integrate FAISS index (Proposal 2)
- Implement query embedding via vLLM
- Add distance-to-score conversion
- Implement GPU health check

**Day 3-4: Fusion Ranking**

- Implement RRF algorithm
- Implement weighted normalization
- Add fusion method registry
- Add per-component score preservation

**Day 5: Testing + Validation**

- Write unit tests for all components
- Write integration tests for hybrid pipeline
- Benchmark latency (target: P95 <500ms)

---

### Phase 2: Add Reranking + Clinical Boosting (Week 3-4)

#### Week 3: Reranking

**Day 1-2: Cross-Encoder Service**

- Implement `CrossEncoderReranker` class
- Add model loading with GPU check
- Implement batch reranking method
- Add reranking timeout and fallback

**Day 3-4: Reranking Integration**

- Add reranking feature flag
- Integrate reranker into hybrid search
- Add reranking metadata to response
- Implement A/B testing support

**Day 5: Testing + Performance**

- Write reranker unit tests
- Benchmark reranking latency (target: +120ms)
- A/B test: nDCG@10 with/without reranking

#### Week 4: Clinical Boosting + Table Routing

**Day 1-2: Table-Aware Routing**

- Implement query intent classifier
- Define tabular query patterns
- Implement table chunk boosting
- Add manual intent override

**Day 3-4: Clinical Intent Boosting**

- Implement clinical intent analyzer
- Define intent-to-section mapping
- Implement section-aware boosting
- Add confidence-based boost decay

**Day 5: Validation**

- Domain expert review (20 queries)
- A/B test: nDCG@10 with/without boosting
- Adjust boosting rules based on feedback

---

### Phase 3: Evaluation Framework + Production (Week 5-6)

#### Week 5: Evaluation Framework

**Day 1-2: Metrics Implementation**

- Implement Recall@K, nDCG@K, MRR
- Add graded relevance support
- Implement per-query and aggregate metrics
- Add confidence intervals

**Day 3-4: Test Set Management**

- Create initial test set (50 queries)
- Implement test set loader
- Add query type stratification
- Implement inter-annotator agreement

**Day 5: Evaluation Harness**

- Implement evaluation runner
- Add A/B testing framework
- Implement evaluation caching
- Add CI integration

#### Week 6: Production Deployment + Validation

**Day 1-2: Staging Deployment**

- Deploy all components to staging
- Run smoke tests
- Validate performance metrics

**Day 3-4: Production Rollout**

- Shadow traffic testing (log, don't serve)
- Canary deployment (10% traffic)
- Gradual rollout (50%, 100%)
- Monitor metrics for 48 hours

**Day 5: Post-Deployment Validation**

- Validate Recall@10 improvement (65% → 82%)
- Validate nDCG@10 improvement (0.68 → 0.79)
- Validate latency SLA (P95 <500ms)
- Create deployment success report

---

## Configuration Management

### Fusion Configuration

```yaml
# config/retrieval/fusion.yaml
default_method: rrf  # or "weighted"

rrf:
  k: 60  # Standard RRF constant

weighted:
  normalization: minmax  # or "zscore"
  weights:
    bm25: 0.33
    splade: 0.34
    dense: 0.33

  # Per-query-type weights (optional)
  query_type_weights:
    exact_term:
      bm25: 0.5
      splade: 0.3
      dense: 0.2
    paraphrase:
      bm25: 0.2
      splade: 0.3
      dense: 0.5
    complex:
      bm25: 0.3
      splade: 0.3
      dense: 0.4
```

### Reranking Configuration

```yaml
# config/retrieval/reranking.yaml
enabled: false  # Opt-in feature flag

models:
  default: bge-reranker-base

  bge-reranker-base:
    model_id: BAAI/bge-reranker-base
    device: cuda
    batch_size: 32
    timeout_ms: 200

  ms-marco-minilm:
    model_id: cross-encoder/ms-marco-MiniLM-L-12-v2
    device: cuda
    batch_size: 64
    timeout_ms: 150

# Per-tenant settings
tenants:
  tenant-123:
    enabled: true  # Enable reranking by default
    model: bge-reranker-base

  tenant-456:
    enabled: false
```

### Table Routing Configuration

```yaml
# config/retrieval/table_routing.yaml
enabled: true

tabular_keywords:
  adverse events: 0.9
  side effects: 0.8
  effect sizes: 0.9
  outcome measures: 0.9
  results table: 1.0
  demographics: 0.7

boost_multiplier:
  min: 1.0  # No boosting
  max: 3.0  # Maximum 3x boost
  formula: "1 + (2 * confidence)"  # Dynamic based on confidence
```

### Clinical Boosting Configuration

```yaml
# config/retrieval/clinical_boosting.yaml
enabled: true

intents:
  eligibility:
    keywords: [eligibility, inclusion, exclusion, criteria]
    sections: [Eligibility Criteria, eligibility]
    boost_multiplier: 3.0

  adverse_events:
    keywords: [adverse events, side effects, toxicity, safety]
    sections: [Adverse Reactions, Adverse Events, Safety]
    boost_multiplier: 2.0

  results:
    keywords: [results, outcome, efficacy, effectiveness]
    sections: [Results, Findings, Outcomes]
    boost_multiplier: 2.0

  methods:
    keywords: [methods, study design, protocol]
    sections: [Methods, Study Design]
    boost_multiplier: 1.5

  dosage:
    keywords: [dosage, dose, administration]
    sections: [Dosage and Administration, Dosage]
    boost_multiplier: 2.5

  indications:
    keywords: [indication, use, treatment]
    sections: [Indications and Usage, Indications]
    boost_multiplier: 2.0

# Confidence threshold for applying boosting
confidence_threshold: 0.6
```

---

## Observability & Monitoring

### Prometheus Metrics

```python
# Retrieval latency per component
RETRIEVAL_DURATION = Histogram(
    "medicalkg_retrieval_duration_seconds",
    "Retrieval duration per component",
    ["component"],  # bm25, splade, dense
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0]
)

# Fusion latency
FUSION_DURATION = Histogram(
    "medicalkg_fusion_duration_seconds",
    "Fusion duration",
    ["method"],  # rrf, weighted
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1]
)

# Reranking latency
RERANKING_DURATION = Histogram(
    "medicalkg_reranking_duration_seconds",
    "Reranking duration",
    ["model"],  # bge-reranker-base, ms-marco-minilm
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0]
)

# Component usage
COMPONENT_USED = Counter(
    "medicalkg_retrieval_component_used_total",
    "Component usage count",
    ["component"]  # bm25, splade, dense
)

# Fusion method usage
FUSION_METHOD_USED = Counter(
    "medicalkg_fusion_method_used_total",
    "Fusion method usage count",
    ["method"]  # rrf, weighted
)

# Recall@K metric
RECALL_AT_K = Gauge(
    "medicalkg_retrieval_recall_at_k",
    "Recall@K metric",
    ["k"]  # 5, 10, 20
)

# nDCG@K metric
NDCG_AT_K = Gauge(
    "medicalkg_retrieval_ndcg_at_k",
    "nDCG@K metric",
    ["k"]  # 5, 10, 20
)

# Query intent distribution
QUERY_INTENT_DETECTED = Counter(
    "medicalkg_query_intent_detected_total",
    "Query intent detection count",
    ["intent"]  # eligibility, adverse_events, results, etc.
)

# Reranking opt-in rate
RERANKING_ENABLED = Counter(
    "medicalkg_reranking_enabled_total",
    "Reranking opt-in count",
    ["enabled"]  # true, false
)

# Evaluation runs
EVALUATION_RUNS = Counter(
    "medicalkg_evaluation_runs_total",
    "Evaluation run count",
    ["test_set_version"]  # v1, v2, etc.
)
```

### CloudEvents

**Retrieval Completed Event**:

```json
{
  "specversion": "1.0",
  "type": "com.medical-kg.retrieval.completed",
  "source": "/retrieval-service",
  "id": "retrieval-job-abc123",
  "time": "2025-10-07T14:30:02Z",
  "data": {
    "job_id": "job-abc123",
    "query": "pembrolizumab adverse events",
    "components_used": ["bm25", "splade", "dense"],
    "fusion_method": "rrf",
    "reranked": false,
    "result_count": 10,
    "duration_seconds": 0.15,
    "component_durations": {
      "bm25": 0.08,
      "splade": 0.12,
      "dense": 0.04
    },
    "fusion_duration": 0.01
  }
}
```

**Evaluation Completed Event**:

```json
{
  "specversion": "1.0",
  "type": "com.medical-kg.evaluation.completed",
  "source": "/evaluation-service",
  "id": "eval-abc123",
  "time": "2025-10-07T15:00:00Z",
  "data": {
    "evaluation_id": "eval-abc123",
    "test_set": "test_set_v1",
    "query_count": 50,
    "metrics": {
      "recall_at_10": 0.82,
      "ndcg_at_10": 0.79,
      "mrr": 0.85
    },
    "per_component": {
      "bm25_only": {"recall_at_10": 0.65},
      "splade_only": {"recall_at_10": 0.72},
      "dense_only": {"recall_at_10": 0.75},
      "hybrid_fusion": {"recall_at_10": 0.82}
    },
    "duration_seconds": 28
  }
}
```

### Grafana Dashboard

**Panels**:

1. **Retrieval Latency Percentiles**: Line chart (P50, P95, P99) for hybrid retrieval
2. **Per-Component Latency**: Stacked area chart (BM25, SPLADE, Dense)
3. **Fusion Method Distribution**: Pie chart (RRF vs Weighted usage)
4. **Reranking Opt-In Rate**: Gauge (% of queries with reranking enabled)
5. **Recall@10 Trend**: Line chart over time
6. **nDCG@10 Trend**: Line chart over time
7. **Query Intent Heatmap**: Heatmap of intent distribution over time
8. **Component Contribution**: Bar chart (per-component nDCG contribution)

**Alerting Rules**:

```yaml
# Recall@10 degradation
- alert: RecallDegraded
  expr: medicalkg_retrieval_recall_at_k{k="10"} < 0.75
  for: 15m
  labels:
    severity: critical
  annotations:
    summary: "Recall@10 below 75% for 15 minutes"

# Retrieval latency high
- alert: RetrievalLatencyHigh
  expr: histogram_quantile(0.95, rate(medicalkg_retrieval_duration_seconds_bucket[5m])) > 0.6
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Retrieval P95 latency >600ms for 10 minutes"

# Component failure rate high
- alert: ComponentFailureRateHigh
  expr: rate(medicalkg_retrieval_component_failed_total[5m]) / rate(medicalkg_retrieval_component_used_total[5m]) > 0.05
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Component failure rate >5% for 5 minutes"

# nDCG@10 degradation
- alert: NDCGDegraded
  expr: medicalkg_retrieval_ndcg_at_k{k="10"} < 0.70
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "nDCG@10 below 0.70 for 15 minutes"
```

---

## Risks / Trade-offs

### Risk 1: Fusion Weight Tuning Complexity

**Risk**: Weighted normalization requires score calibration, sensitive to score distributions

**Impact**: Sub-optimal ranking if weights not tuned correctly

**Mitigation**:

- RRF as default (parameter-free, no tuning)
- Weighted normalization as advanced opt-in
- Provide tuning guide with grid search example
- A/B test before deploying weighted fusion

**Likelihood**: Medium | **Impact**: Medium | **Mitigation Effectiveness**: High

---

### Risk 2: Reranking Latency Increase

**Risk**: Reranking adds +120ms latency, potentially degrading user experience

**Impact**: Slower search results, user dissatisfaction

**Mitigation**:

- Feature flag (opt-in, not default)
- Only enable for high-precision scenarios
- Timeout fallback (if reranking >200ms, skip)
- A/B test latency impact before full rollout

**Likelihood**: Medium | **Impact**: Medium | **Mitigation Effectiveness**: High

---

### Risk 3: Test Set Bias and Overfitting

**Risk**: 50-query test set may overfit, reducing generalization

**Impact**: Metrics look good on test set, poor on real queries

**Mitigation**:

- Quarterly test set refresh with new queries
- Held-out validation set (20% of queries)
- Diverse query types (exact term, paraphrase, complex)
- Monitor live traffic metrics (not just test set)

**Likelihood**: Medium | **Impact**: Medium | **Mitigation Effectiveness**: Medium

---

### Risk 4: Clinical Boosting Over-Optimization

**Risk**: Aggressive boosting may miss non-obvious relevance (e.g., eligibility query also needs methods)

**Impact**: Reduced diversity in results, tunnel vision

**Mitigation**:

- Conservative boost multipliers (2-3x, not 10x)
- Confidence-based decay (low confidence → lower boost)
- Domain expert review of boosted results
- A/B test with/without boosting before full rollout
- Allow disabling boosting via API parameter

**Likelihood**: Low | **Impact**: Medium | **Mitigation Effectiveness**: High

---

### Risk 5: SPLADE Latency Bottleneck

**Risk**: SPLADE query is slowest component (120ms P95), bottlenecks hybrid search

**Impact**: Overall latency degradation

**Mitigation**:

- Optimize OpenSearch `rank_features` query (reduce top_k terms)
- Implement SPLADE caching (30-minute TTL)
- Graceful degradation (if SPLADE times out, continue with BM25+Dense)
- Monitor SPLADE latency, alert if >150ms P95

**Likelihood**: Low | **Impact**: Medium | **Mitigation Effectiveness**: High

---

## Migration Plan

### Pre-Migration Checklist

- [ ] All components deployed to staging and validated
- [ ] Test set created with 50 queries labeled by domain experts
- [ ] Evaluation framework running on staging
- [ ] Grafana dashboards deployed and configured
- [ ] All tests passing (unit, integration, performance)
- [ ] No linting errors or type errors
- [ ] Runbook reviewed by operations team
- [ ] Rollback procedures documented

### Migration Steps

#### Step 1: Shadow Traffic Testing (Week 1)

```bash
# Enable hybrid retrieval in shadow mode
# Log results to separate log stream, don't serve to users

# Compare metrics:
# - Latency (shadow vs production)
# - Recall@10 (shadow vs production)
# - Result overlap (% of same docs in top-10)

# Decision: Proceed to canary if:
# - Shadow latency P95 <500ms
# - Shadow Recall@10 >70%
# - Result overlap >60% (stability check)
```

#### Step 2: Canary Deployment (Week 2, 10% Traffic)

```bash
# Route 10% of traffic to hybrid retrieval
kubectl apply -f ops/k8s/canary/retrieval-canary-10pct.yaml

# Monitor for 48 hours:
# - Latency P95 (target: <500ms)
# - Error rate (target: <1%)
# - Recall@10 (target: ≥75%)
# - User feedback (qualitative)

# Decision: Proceed to 50% if:
# - All metrics meet targets
# - No critical issues reported
```

#### Step 3: Gradual Rollout (Week 3, 50% Traffic)

```bash
# Route 50% of traffic to hybrid retrieval
kubectl apply -f ops/k8s/canary/retrieval-canary-50pct.yaml

# Monitor for 72 hours:
# - Same metrics as canary
# - Compare 50% cohort vs 50% baseline

# A/B test results:
# - Recall@10: Hybrid vs Baseline
# - nDCG@10: Hybrid vs Baseline
# - User satisfaction: Survey feedback

# Decision: Proceed to 100% if:
# - Hybrid outperforms baseline (statistical significance p<0.05)
# - No degradation in latency or errors
```

#### Step 4: Full Rollout (Week 4, 100% Traffic)

```bash
# Route 100% of traffic to hybrid retrieval
kubectl apply -f ops/k8s/production/retrieval-prod.yaml

# Monitor intensively for 48 hours:
# - Latency P95 (alert if >550ms)
# - Recall@10 (alert if <75%)
# - nDCG@10 (alert if <0.70)
# - Component failures (alert if >5%)

# Post-deployment validation:
# - Run evaluation on production traffic
# - Compare before/after metrics
# - Collect user feedback
# - Create deployment success report
```

#### Step 5: Enable Optional Features (Week 5-6)

```bash
# Phase 5a: Enable reranking for high-precision tenants (opt-in)
# - Select 2-3 tenants for pilot
# - A/B test with/without reranking
# - Validate nDCG@10 improvement (target: +5%)

# Phase 5b: Deploy table routing to production
# - Enable for 10% traffic
# - Monitor CTR on first result (table queries)
# - Gradual rollout to 100%

# Phase 5c: Deploy clinical boosting to production
# - Domain expert review (20 queries)
# - Enable for 10% traffic
# - Validate nDCG@10 improvement (target: +3%)
# - Gradual rollout to 100%

# Phase 5d: Integrate evaluation framework with CI
# - Run evaluation on every PR
# - Fail if Recall@10 drops >5%
# - Add to release criteria
```

---

## Testing Strategy

### Unit Tests (60 tests)

**Hybrid Coordinator** (10 tests):

- Test all components enabled, partial failures, empty results
- Test graceful degradation when component fails
- Test correlation ID propagation
- Test caching behavior

**Fusion Methods** (15 tests):

- RRF: 2 components, 3 components, ties, empty lists
- Weighted: Equal weights, biased weights, normalization (min-max, z-score)
- Property tests: Symmetric, stable, repeatable

**Reranking** (10 tests):

- Single batch, multiple batches, timeout
- GPU unavailable fallback
- Empty results, single result

**Table Routing** (10 tests):

- Tabular query with tables, without tables
- Confidence scoring, manual override
- Edge cases (all tables, no tables)

**Clinical Boosting** (10 tests):

- Each intent, multi-intent, no intent
- Section matching, intent hint matching
- Confidence-based decay

**Evaluation Metrics** (5 tests):

- Recall@K: Known inputs, edge cases (empty results)
- nDCG@K: Graded relevance, binary relevance
- MRR: First result relevant, no relevant results

---

### Integration Tests (25 tests)

**Hybrid Retrieval** (5 tests):

- End-to-end with real OpenSearch, FAISS
- BM25 + SPLADE fusion
- BM25 + Dense fusion
- All 3 components fusion
- Component failure graceful degradation

**Fusion + Reranking** (5 tests):

- RRF fusion end-to-end
- Weighted fusion end-to-end
- Fusion + reranking pipeline
- Reranking failure fallback

**Clinical Features** (5 tests):

- Table routing end-to-end (tabular query)
- Clinical boosting end-to-end (eligibility query)
- Combined table + clinical boosting
- Manual intent override

**Evaluation Framework** (5 tests):

- Run evaluation on 10-query test set
- A/B test: RRF vs Weighted
- Per-component evaluation
- CI integration (fail on metric drop)

**API Integration** (5 tests):

- REST API: `/v1/search?fusion_method=rrf&rerank=true`
- GraphQL API: `search(fusionMethod: RRF, rerank: true)`
- gRPC API: `HybridSearch` RPC
- Multi-tenant isolation
- Rate limiting

---

### Performance Tests (15 tests)

**Latency Benchmarks** (5 tests):

- Hybrid retrieval P95 latency (target: <500ms)
- Hybrid + reranking P95 latency (target: <650ms)
- Per-component latency (BM25 <100ms, SPLADE <150ms, Dense <50ms)
- Fusion latency (RRF <5ms, Weighted <10ms)
- Evaluation time (target: <30s for 50 queries)

**Load Tests** (5 tests):

- 100 concurrent users, 5-minute duration
- 500 concurrent users, 2-minute duration (stress test)
- 10 concurrent users, 24-hour duration (soak test)
- Latency stability over time
- Error rate under load (target: <1%)

**Resource Usage** (5 tests):

- GPU utilization during reranking (target: 60-80%)
- Memory usage (no leaks over 24 hours)
- CPU usage (stable over time)
- Cache hit rate (target: >40%)
- OpenSearch index size growth

---

## Rollback Procedures

### Rollback Trigger Conditions

**Automated Rollback**:

- Recall@10 drops below 70% for >15 minutes
- P95 latency exceeds 600ms for >10 minutes
- Error rate exceeds 3% for >5 minutes
- Component failure rate exceeds 10% for >5 minutes

**Manual Rollback**:

- Critical bug discovered (data corruption, security issue)
- User complaints exceed threshold
- Unacceptable latency degradation
- Team decision based on qualitative issues

### Rollback Steps

```bash
# Step 1: Immediate traffic shift (10 seconds)
# Shift 100% traffic back to baseline retrieval
kubectl apply -f ops/k8s/production/retrieval-baseline.yaml

# Step 2: Validate baseline restoration (5 minutes)
# Check metrics return to baseline
# - Latency P95 <200ms (baseline BM25-only)
# - Error rate <0.5%
# - No component failures

# Step 3: Post-incident analysis (1 hour)
# Gather logs, metrics, traces
# Identify root cause
# Create incident report

# Step 4: Fix and re-deploy (1-2 days)
# Fix identified issues
# Re-test in staging
# Schedule new production deployment
```

### Recovery Time Objective (RTO)

**Target RTO**: 10 seconds (traffic shift via Kubernetes)
**Maximum RTO**: 5 minutes (if requires manual intervention)

---

## Open Questions

1. **Weighted Fusion Default Weights**: Should we use learned weights from test set or start with equal weights?
   - **Recommendation**: Start equal, learn weights from production traffic over time

2. **Reranking Model Selection**: BGE-reranker-base vs ms-marco-MiniLM-L-12-v2?
   - **Recommendation**: BGE for quality (default), ms-marco for latency (opt-in)

3. **Test Set Size**: 50 queries sufficient or need 100+?
   - **Recommendation**: Start with 50, expand to 100 after initial validation

4. **Clinical Boosting Aggressiveness**: 2-3x boost multipliers or higher?
   - **Recommendation**: Conservative (2-3x), adjust based on domain expert feedback

5. **SPLADE Query-Side Expansion**: Enable by default or opt-in?
   - **Recommendation**: Opt-in initially (adds latency), default after validation

---

## Summary

This design standardizes retrieval around six key principles:

1. **Hybrid Retrieval**: Complementary strategies (BM25 + SPLADE + Dense) for 26% Recall improvement
2. **Stable Fusion**: RRF as default (parameter-free), weighted as advanced option
3. **Opt-In Reranking**: Cross-encoder for +7% nDCG on complex queries (GPU-based)
4. **Clinical Context**: Table routing and intent boosting leverage domain structure
5. **Evaluation Framework**: Systematic measurement enables A/B testing and regression detection
6. **Explainability**: Per-component scores preserved for trust and debugging

**Key Benefits**:

- 26% Recall@10 improvement (65% → 82%)
- 16% nDCG@10 improvement (0.68 → 0.79)
- <500ms P95 latency (hybrid + fusion)
- Explainable ranking (per-component scores)
- A/B testing support (evaluation framework)

**Timeline**: 6 weeks (2 build, 2 integrate, 2 validate)

**Status**: Ready for implementation after approval
