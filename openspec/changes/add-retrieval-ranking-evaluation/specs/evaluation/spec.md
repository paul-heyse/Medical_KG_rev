# Evaluation Capability: Spec Delta

## ADDED Requirements

### Requirement: Recall@K Metric

The system SHALL implement Recall@K metric to measure the percentage of relevant documents retrieved in the top-K results.

**Rationale**: Recall measures coverage, ensuring relevant documents are not missed. Critical for biomedical literature search where high recall is essential.

#### Scenario: Recall@10 calculation

- **GIVEN** a query with 5 relevant documents: [doc1, doc2, doc3, doc4, doc5]
- **AND** retrieved top-10: [doc1, doc6, doc2, doc7, doc8, doc9, doc3, doc10, doc11, doc12]
- **WHEN** Recall@10 is calculated
- **THEN** Recall@10 = 3/5 = 0.60 (60%)
- **BECAUSE** 3 of 5 relevant docs (doc1, doc2, doc3) are in top-10

#### Scenario: Recall@K for multiple K values

- **GIVEN** a query with relevant documents
- **WHEN** Recall is calculated for K=5, K=10, K=20
- **THEN** three metrics are returned: Recall@5, Recall@10, Recall@20
- **AND** Recall@K increases (or stays same) as K increases (monotonic property)

---

### Requirement: nDCG@K Metric

The system SHALL implement Normalized Discounted Cumulative Gain at K (nDCG@K) to measure ranking quality with graded relevance.

**Rationale**: nDCG accounts for position and graded relevance (highly relevant > relevant > somewhat relevant), better than binary Recall for ranking quality.

#### Scenario: nDCG@10 calculation with graded relevance

- **GIVEN** a query with graded relevance labels (0-3 scale):
  - doc1: grade 3 (highly relevant)
  - doc2: grade 2 (relevant)
  - doc3: grade 1 (somewhat relevant)
  - doc4: grade 0 (irrelevant)
- **AND** retrieved top-10: [doc1, doc4, doc2, ...]
- **WHEN** nDCG@10 is calculated using scikit-learn `ndcg_score`
- **THEN** DCG = (2^3 - 1)/log2(2) + (2^0 - 1)/log2(3) + (2^2 - 1)/log2(4) + ...
- **AND** IDCG = DCG for ideal ranking [doc1, doc2, doc3, ...]
- **AND** nDCG@10 = DCG / IDCG ∈ [0, 1]

#### Scenario: nDCG@10 with binary relevance

- **GIVEN** a query with binary relevance (relevant/irrelevant, no grading)
- **WHEN** nDCG@10 is calculated
- **THEN** grades are mapped to 0 (irrelevant) and 1 (relevant)
- **AND** nDCG calculation proceeds as normal

---

### Requirement: Mean Reciprocal Rank (MRR)

The system SHALL implement Mean Reciprocal Rank (MRR) to measure the position of the first relevant result.

**Rationale**: MRR focuses on "first answer" quality, critical for queries where users expect a single relevant result.

#### Scenario: MRR calculation for single query

- **GIVEN** a query with relevant documents [doc1, doc2, doc3]
- **AND** retrieved ranking: [doc5, doc6, doc1, doc7, ...]
- **WHEN** MRR is calculated
- **THEN** first relevant result is doc1 at rank 3
- **AND** Reciprocal Rank = 1/3 = 0.333

#### Scenario: MRR calculation across multiple queries

- **GIVEN** 3 queries with Reciprocal Ranks: 1.0 (rank 1), 0.5 (rank 2), 0.333 (rank 3)
- **WHEN** MRR is calculated across queries
- **THEN** MRR = (1.0 + 0.5 + 0.333) / 3 = 0.611

---

### Requirement: Test Set Management

The system SHALL provide test set management with YAML-based storage, versioning, and query type stratification (exact_term, paraphrase, complex_clinical).

**Rationale**: Systematic test sets enable reproducible evaluation, A/B testing, and regression detection.

#### Scenario: Load test set from YAML

- **GIVEN** a test set file `test_set_v1.yaml` with 50 queries
- **WHEN** the test set is loaded via `load_test_set("test_set_v1")`
- **THEN** 50 queries are loaded with fields: `query_id`, `query_text`, `query_type`, `relevant_docs`
- **AND** each relevant doc has `doc_id` and `grade` (0-3)
- **AND** query types are stratified: 20 exact_term, 15 paraphrase, 15 complex_clinical

#### Scenario: Test set validation

- **GIVEN** a test set loaded from YAML
- **WHEN** the test set is validated
- **THEN** every query has ≥1 relevant document (grade > 0)
- **AND** all graded labels are in valid range (0-3)
- **AND** query IDs are unique

---

### Requirement: Evaluation Runner

The system SHALL provide an evaluation runner that executes all test set queries, calculates metrics (Recall@K, nDCG@K, MRR), and produces aggregate results.

**Rationale**: Automated evaluation enables A/B testing, CI integration, and systematic quality measurement.

#### Scenario: Run evaluation on test set

- **GIVEN** a test set with 50 queries and a retrieval function
- **WHEN** evaluation is run via `evaluate(retrieval_fn, test_set)`
- **THEN** all 50 queries are executed
- **AND** metrics are calculated per query: Recall@5, Recall@10, Recall@20, nDCG@10, MRR
- **AND** aggregate metrics are returned: mean, median, std dev across queries

#### Scenario: Per-component evaluation

- **GIVEN** a hybrid retrieval system with BM25, SPLADE, Dense components
- **WHEN** per-component evaluation is run
- **THEN** metrics are calculated for:
  - BM25-only: Recall@10, nDCG@10
  - SPLADE-only: Recall@10, nDCG@10
  - Dense-only: Recall@10, nDCG@10
  - Hybrid fusion: Recall@10, nDCG@10
- **AND** component contributions are compared

#### Scenario: Evaluation caching

- **GIVEN** an evaluation run with config hash `abc123`
- **WHEN** evaluation is run again with same config
- **THEN** cached results are returned (no re-execution)
- **AND** cache key is `hash(retrieval_config + test_set_version)`

---

### Requirement: A/B Testing Framework

The system SHALL provide an A/B testing framework to compare two retrieval configurations and determine statistical significance.

**Rationale**: Enables data-driven decisions on fusion methods, reranking, boosting rules.

#### Scenario: A/B test RRF vs Weighted fusion

- **GIVEN** two configurations:
  - Config A: `{"fusion_method": "rrf", "rrf_k": 60}`
  - Config B: `{"fusion_method": "weighted", "weights": {"bm25": 0.3, "splade": 0.4, "dense": 0.3}}`
- **WHEN** A/B test is run on test set
- **THEN** metrics are calculated for both configs
- **AND** paired t-test is performed on nDCG@10 values
- **AND** result includes: mean difference, t-statistic, p-value
- **AND** if p < 0.05, difference is statistically significant

#### Scenario: A/B test with/without reranking

- **GIVEN** two configurations:
  - Config A: Fusion only (no reranking)
  - Config B: Fusion + reranking
- **WHEN** A/B test is run on test set
- **THEN** nDCG@10 improvement is measured (Config B vs Config A)
- **AND** statistical significance is determined (paired t-test)
- **AND** latency impact is measured (Config B latency - Config A latency)

---

### Requirement: CI Integration for Regression Detection

The system SHALL integrate evaluation into CI pipeline, failing builds if Recall@10 drops >5% below baseline.

**Rationale**: Prevents quality regressions from being deployed to production.

#### Scenario: CI evaluation on pull request

- **GIVEN** a pull request modifying retrieval code
- **WHEN** CI runs evaluation on test set
- **THEN** current Recall@10 is compared with baseline (main branch)
- **AND** if Recall@10 drops >5%, CI fails with error message
- **AND** if Recall@10 is within 5%, CI passes

#### Scenario: Evaluation report in CI

- **GIVEN** a CI evaluation run
- **WHEN** evaluation completes
- **THEN** a markdown report is generated with:
  - Recall@10 (current vs baseline)
  - nDCG@10 (current vs baseline)
  - Per-query-type breakdown
  - Failing queries (if any)
- **AND** report is posted as PR comment

---

### Requirement: Evaluation REST Endpoint

The system SHALL provide a `/v1/evaluate` endpoint for running evaluations with uploaded test sets.

**Rationale**: Enables ad-hoc evaluation runs without CI, supports experimentation.

#### Scenario: Evaluate with uploaded test set

- **GIVEN** a test set JSON uploaded via `POST /v1/evaluate`
- **WHEN** the evaluation endpoint processes the request
- **THEN** queries are executed against current retrieval system
- **AND** metrics are calculated and returned in response:

  ```json
  {
    "metrics": {
      "recall_at_10": 0.82,
      "ndcg_at_10": 0.79,
      "mrr": 0.85
    },
    "per_query_type": {
      "exact_term": {"recall_at_10": 0.90},
      "paraphrase": {"recall_at_10": 0.78},
      "complex_clinical": {"recall_at_10": 0.75}
    }
  }
  ```

#### Scenario: Evaluate specific component only

- **GIVEN** a request `/v1/evaluate?components=bm25` (BM25-only evaluation)
- **WHEN** evaluation runs
- **THEN** only BM25 component is evaluated
- **AND** metrics reflect BM25-only performance

---

### Requirement: Evaluation Metrics Tracking in Prometheus

The system SHALL emit evaluation metrics to Prometheus for trend analysis and alerting.

**Rationale**: Enables monitoring of retrieval quality over time, alerting on degradation.

#### Scenario: Emit Recall@10 to Prometheus

- **GIVEN** an evaluation run with Recall@10 = 0.82
- **WHEN** evaluation completes
- **THEN** Prometheus gauge `medicalkg_retrieval_recall_at_k{k="10"}` is set to 0.82
- **AND** metric is scraped by Prometheus

#### Scenario: Alert on Recall@10 degradation

- **GIVEN** a Prometheus alert rule: `medicalkg_retrieval_recall_at_k{k="10"} < 0.75`
- **WHEN** Recall@10 drops below 0.75 for >15 minutes
- **THEN** Prometheus alert fires
- **AND** on-call engineer is notified

---

### Requirement: Test Set Refresh Process

The system SHALL provide a quarterly test set refresh process to prevent overfitting and ensure query diversity.

**Rationale**: Test set should evolve with real user queries, prevent gaming metrics.

#### Scenario: Quarterly test set refresh

- **GIVEN** current test set version v1 (50 queries)
- **WHEN** quarterly refresh is triggered
- **THEN** 20 queries are replaced with new queries from production traffic
- **AND** new queries are labeled by 2 domain experts
- **AND** inter-annotator agreement (Cohen's kappa) is verified (κ > 0.6)
- **AND** new test set is versioned as v2

#### Scenario: Held-out validation set

- **GIVEN** a test set of 50 queries
- **WHEN** test set is split for validation
- **THEN** 80% (40 queries) are used for evaluation
- **AND** 20% (10 queries) are held-out for validation
- **AND** held-out set is used to detect overfitting

---

## MODIFIED Requirements

None (Evaluation is a new capability, no existing requirements to modify)

---

## REMOVED Requirements

None (Evaluation is a new capability, no existing requirements to remove)
