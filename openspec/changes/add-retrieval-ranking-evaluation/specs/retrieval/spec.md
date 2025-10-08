# Retrieval Capability: Spec Delta

## ADDED Requirements

### Requirement: Hybrid Multi-Component Retrieval

The system SHALL provide hybrid retrieval combining BM25 (lexical), SPLADE (learned sparse), and Dense KNN (semantic) strategies in parallel, each contributing top-100 candidates.

**Rationale**: Complementary strategies achieve 82% Recall@10 vs 65% single-strategy, leveraging BM25 for exact matches, SPLADE for term expansion, Dense for semantic similarity.

#### Scenario: Hybrid retrieval executes all components in parallel

- **GIVEN** a query "diabetes treatment" and components ["bm25", "splade", "dense"]
- **WHEN** the hybrid coordinator executes the search
- **THEN** BM25, SPLADE, and Dense searches run in parallel via `asyncio.gather`
- **AND** each component returns top-100 candidates
- **AND** total execution time is max(BM25_time, SPLADE_time, Dense_time) ~= 120ms P95

#### Scenario: Graceful degradation when component fails

- **GIVEN** a hybrid search with all 3 components enabled
- **WHEN** SPLADE component times out after 300ms
- **THEN** the coordinator logs a warning and excludes SPLADE from fusion
- **AND** continues with BM25 + Dense results only
- **AND** response includes `component_errors: ["splade_timeout"]`

#### Scenario: Per-component score tracking

- **GIVEN** a hybrid search returning 10 results
- **WHEN** the results are returned to the client
- **THEN** each result includes `component_scores: {"bm25": 12.5, "splade": 8.3, "dense": 0.87}`
- **AND** scores are preserved from original component rankings

---

### Requirement: Reciprocal Rank Fusion (RRF)

The system SHALL implement Reciprocal Rank Fusion with configurable k parameter as the default fusion method for combining component rankings.

**Rationale**: RRF is parameter-free (k=60 standard), stable across query paraphrases (10% variance vs 30% ad-hoc), and symmetric (order-independent).

#### Scenario: RRF fuses multiple component rankings

- **GIVEN** BM25 ranking [doc1, doc2, doc3], SPLADE ranking [doc2, doc1, doc4], Dense ranking [doc1, doc4, doc2]
- **WHEN** RRF fusion is applied with k=60
- **THEN** fused scores are: doc1 (RRF=0.049), doc2 (RRF=0.048), doc4 (RRF=0.032), doc3 (RRF=0.016)
- **AND** final ranking is [doc1, doc2, doc4, doc3]
- **AND** RRF formula: `score = Σ(1 / (rank_i + 60))` applied per document across all components

#### Scenario: RRF handles ties deterministically

- **GIVEN** two documents with identical RRF scores (0.045)
- **WHEN** fusion ranking is finalized
- **THEN** tie-breaking uses original rank from primary component (BM25)
- **AND** ranking is deterministic and repeatable

---

### Requirement: Weighted Normalization Fusion (Advanced Opt-In)

The system SHALL provide weighted normalization fusion as an advanced option, allowing explicit component weight configuration.

**Rationale**: Enables expert users to bias fusion toward specific components (e.g., 40% SPLADE, 30% Dense, 30% BM25 for semantic-heavy queries).

#### Scenario: Weighted fusion with custom weights

- **GIVEN** a search with `fusion_method="weighted"` and `weights={"bm25": 0.3, "splade": 0.4, "dense": 0.3}`
- **WHEN** fusion is applied
- **THEN** component scores are min-max normalized to [0, 1]
- **AND** fused score = 0.3·norm(bm25) + 0.4·norm(splade) + 0.3·norm(dense)
- **AND** results are ranked by weighted fused score

#### Scenario: Per-query-type weights (advanced)

- **GIVEN** a query classified as "paraphrase" type
- **WHEN** weighted fusion is configured with per-query-type weights
- **THEN** weights are `{"bm25": 0.2, "splade": 0.3, "dense": 0.5}` (favoring Dense for paraphrases)
- **AND** fusion applies query-type-specific weights

---

### Requirement: Optional Cross-Encoder Reranking

The system SHALL provide optional cross-encoder reranking (BGE-reranker-base) as an opt-in feature for high-precision scenarios, requiring GPU and failing fast if unavailable.

**Rationale**: +5-8% nDCG@10 improvement on complex queries, acceptable latency increase (+120ms), GPU-only enforces quality.

#### Scenario: Reranking top-100 results with cross-encoder

- **GIVEN** a hybrid search with `rerank=true` returning 100 fused results
- **WHEN** the reranker service processes the results
- **THEN** cross-encoder scores (query, document) pairs in batches of 32
- **AND** results are re-ranked by cross-encoder scores
- **AND** top-10 reranked results are returned
- **AND** response includes `reranked=true` and `reranker_model="bge-reranker-base"`

#### Scenario: Reranking fails fast if GPU unavailable

- **GIVEN** a reranking request when GPU is unavailable
- **WHEN** the reranker service attempts to initialize
- **THEN** the service raises `GpuNotAvailableError("Reranker requires GPU")`
- **AND** the coordinator falls back to fusion ranking (no reranking)
- **AND** response includes `reranked=false, reranker_error="gpu_unavailable"`

#### Scenario: Reranking timeout fallback

- **GIVEN** a reranking request with 200ms timeout
- **WHEN** reranking takes >200ms
- **THEN** the coordinator returns fusion ranking (no reranking)
- **AND** logs a warning "Reranking timeout, falling back to fusion"
- **AND** response includes `reranked=false, reranker_error="timeout"`

---

### Requirement: Table-Aware Routing

The system SHALL detect tabular queries via keyword matching and boost table chunks by up to 3x when tabular intent is detected.

**Rationale**: Adverse event queries should prioritize structured table chunks over narrative text, preserving table HTML for frontend rendering.

#### Scenario: Tabular query detection and boosting

- **GIVEN** a query "pembrolizumab adverse events" (contains "adverse events" keyword)
- **WHEN** the table router analyzes the query
- **THEN** tabular intent is detected with confidence 0.9
- **AND** boost multiplier is 2.8x (formula: 1 + 2·confidence)
- **AND** chunks with `is_table=true` or `intent_hint="ae"` are boosted by 2.8x
- **AND** table chunks rank 1-5 in results

#### Scenario: Non-tabular query no boosting

- **GIVEN** a query "diabetes pathophysiology" (no tabular keywords)
- **WHEN** the table router analyzes the query
- **THEN** tabular confidence is 0.0
- **AND** no boosting is applied (multiplier = 1.0x)
- **AND** results ranked by fusion scores only

#### Scenario: Manual intent override

- **GIVEN** a query with `query_intent=tabular` parameter
- **WHEN** the table router processes the request
- **THEN** tabular boosting is applied regardless of keyword detection
- **AND** boost multiplier is 3.0x (maximum)

---

### Requirement: Clinical Intent Boosting

The system SHALL detect clinical intent (eligibility, adverse_events, results, methods, dosage, indications) and boost matching sections/intent_hints by 2-3x.

**Rationale**: Eligibility queries should prioritize eligibility sections, leveraging IMRaD structure and LOINC codes from Proposal 1.

#### Scenario: Eligibility intent detection and boosting

- **GIVEN** a query "eligibility criteria for breast cancer trials" (contains "eligibility" keyword)
- **WHEN** the clinical intent analyzer processes the query
- **THEN** intent `ELIGIBILITY` is detected with confidence 1.0
- **AND** sections with `section_label="Eligibility Criteria"` or `intent_hint="eligibility"` are boosted by 3.0x
- **AND** eligibility sections rank 1-3 in results

#### Scenario: Multi-intent boosting

- **GIVEN** a query "pembrolizumab dosage and adverse events"
- **WHEN** the clinical intent analyzer processes the query
- **THEN** intents `DOSAGE` (confidence 0.7) and `ADVERSE_EVENTS` (confidence 0.9) are detected
- **AND** dosage sections are boosted by 1.4x (2.0 · 0.7)
- **AND** adverse event sections are boosted by 1.8x (2.0 · 0.9)

#### Scenario: No intent detected no boosting

- **GIVEN** a query "cancer research trends"
- **WHEN** the clinical intent analyzer processes the query
- **THEN** no clinical intent is detected
- **AND** no boosting is applied
- **AND** results ranked by fusion scores only

---

## MODIFIED Requirements

### Requirement: Search API (Modified)

The search API SHALL accept hybrid retrieval parameters (`fusion_method`, `rerank`, `query_intent`) and return per-component scores alongside fused ranking.

**Previous Behavior**: `/v1/search?q=query` returned BM25 results only with single score per result.

**New Behavior**: `/v1/search?q=query&fusion_method=rrf&rerank=true` SHALL return hybrid results with `component_scores` breakdown and fusion metadata.

#### Scenario: Hybrid search request with fusion and reranking

- **GIVEN** a request `/v1/search?q=diabetes&fusion_method=rrf&rerank=true`
- **WHEN** the search is executed
- **THEN** hybrid retrieval runs (BM25 + SPLADE + Dense)
- **AND** RRF fusion combines component rankings
- **AND** cross-encoder reranks top-100
- **AND** response includes:

  ```json
  {
    "results": [
      {
        "doc_id": "PMC123:chunk_5",
        "score": 0.87,
        "component_scores": {
          "bm25": 12.5,
          "splade": 8.3,
          "dense": 0.89,
          "reranker": 0.92
        }
      }
    ],
    "fusion_metadata": {
      "method": "rrf",
      "k": 60,
      "reranked": true,
      "reranker_model": "bge-reranker-base"
    }
  }
  ```

#### Scenario: Component selection via API

- **GIVEN** a request `/v1/search?q=diabetes&components=bm25,dense` (excluding SPLADE)
- **WHEN** the search is executed
- **THEN** only BM25 and Dense components are executed
- **AND** fusion combines BM25 + Dense only
- **AND** response includes `components_used: ["bm25", "dense"]`

---

### Requirement: Search Performance (Modified)

The search performance requirement SHALL accommodate hybrid retrieval latency while maintaining P95 <500ms for fusion ranking.

**Previous Behavior**: P95 <200ms for BM25-only search.

**New Behavior**: P95 <500ms for hybrid retrieval (BM25 + SPLADE + Dense) with RRF fusion, P95 <650ms with reranking.

#### Scenario: Hybrid retrieval latency target

- **GIVEN** a hybrid search with RRF fusion (no reranking)
- **WHEN** 1000 queries are executed under normal load
- **THEN** P95 latency is <500ms
- **AND** component latencies: BM25 P95 <100ms, SPLADE P95 <150ms, Dense P95 <50ms
- **AND** fusion latency: <10ms

#### Scenario: Reranking latency target

- **GIVEN** a hybrid search with RRF fusion and reranking enabled
- **WHEN** 1000 queries are executed under normal load
- **THEN** P95 latency is <650ms
- **AND** reranking latency: +120ms for top-100 results

---

## REMOVED Requirements

### Requirement: BM25-Only Search (Removed)

**Removed**: The requirement for BM25-only search as the primary retrieval method is **REMOVED** in favor of hybrid retrieval as default.

**Reason**: BM25-only achieves only 65% Recall@10, insufficient for biomedical queries with term variation.

**Migration**: BM25 remains as a component in hybrid search, accessible via `components=["bm25"]` parameter for backward compatibility.

---

### Requirement: Ad-Hoc Result Merging (Removed)

**Removed**: The requirement for ad-hoc merging of BM25 and vector results in gateway code is **REMOVED** in favor of standardized fusion methods.

**Reason**: Ad-hoc merging caused 30% variance in relevance across paraphrased queries, lacked explainability.

**Migration**: All result merging delegated to fusion methods (RRF, Weighted) with explicit configuration and per-component score tracking.
