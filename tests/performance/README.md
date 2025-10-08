# Gateway Performance Test Suite

The performance suite uses [k6](https://k6.io) and lightweight Python harnesses to validate latency, throughput, and caching guarantees for hybrid retrieval, fusion ranking, and reranking.

## Scenarios

| Script | Purpose | Thresholds |
| ------ | ------- | ---------- |
| `retrieve_latency.js` | Measures REST retrieval latency under light load. | `http_req_duration` P95 < 500 ms |
| `ingest_throughput.js` | Exercises ingestion endpoints with constant arrival rate. | P95 < 800 ms |
| `concurrency.js` | Mixes ingestion and retrieval traffic to validate concurrent processing. | Retrieval P95 < 500 ms, Ingestion P95 < 900 ms |
| `hybrid_suite.js` | Hybrid/rerank/stress/soak coverage with component metrics & cache hit tracking. | Hybrid P95 < 500 ms, Rerank P95 < 650 ms, component latency SLOs from spec |

Legacy `gateway_smoke_test.js` has been removed in favour of the comprehensive `hybrid_suite.js` scenarios.

## Running Tests Locally

```bash
# Ensure the gateway stack is running (docker compose up).
export BASE_URL=http://localhost:8000

# Light-load retrieval check
k6 run tests/performance/retrieve_latency.js

# Hybrid + rerank benchmark with default thresholds
k6 run tests/performance/hybrid_suite.js

# Optional ingestion throughput + concurrency
k6 run tests/performance/ingest_throughput.js
k6 run tests/performance/concurrency.js
```

Override the target gateway by setting `BASE_URL`. When omitted, scripts default to `http://localhost:8000`. The hybrid suite exposes additional knobs via environment variables:

```bash
export HYBRID_DURATION=2m RERANK_DURATION=2m STRESS_ARRIVAL=300
k6 run tests/performance/hybrid_suite.js
```

To trigger the long-running soak scenario without waiting 24 hours, reduce `SOAK_DURATION` (for example `SOAK_DURATION=30m`).

## Python Retrieval Benchmark Harness

The `run_retrieval_benchmarks.py` script issues evaluation queries, computes Recall@K/nDCG@K/MRR via the OpenSpec evaluation runner, and captures per-component latency, cache hit rate, and GPU utilisation snapshots.

```bash
python tests/performance/run_retrieval_benchmarks.py \
  --base-url http://localhost:8000 \
  --tenant-id eval \
  --test-set test_set_v1 \
  --rerank --rerank-model bge-reranker-base \
  --output benchmarks.json
```

The resulting JSON summary includes:

- Aggregate metrics (`recall@10`, `ndcg@10`, `mrr`, etc.).
- Pipeline stage and per-component latency summaries (p50/p95/max).
- Observed reranking cache hit rate and any returned errors.
- GPU/caching Prometheus samples when the `/metrics` endpoint is reachable.

Use `--dataset-root` to load bespoke evaluation sets or integrate tenant-specific gold data.
