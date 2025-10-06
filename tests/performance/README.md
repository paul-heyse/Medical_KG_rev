# Gateway Performance Test Suite

The performance suite uses [k6](https://k6.io) to validate latency, throughput, and concurrency guarantees for the Medical KG gateway.

## Scenarios

| Script | Purpose | Thresholds |
| ------ | ------- | ---------- |
| `retrieve_latency.js` | Measures REST retrieval latency under light load. | `http_req_duration` P95 < 500ms |
| `ingest_throughput.js` | Exercises ingestion endpoints with constant arrival rate. | P95 < 800ms |
| `concurrency.js` | Mixes ingestion and retrieval traffic to validate concurrent processing. | Retrieval P95 < 500ms, Ingestion P95 < 900ms |
| `gateway_smoke_test.js` | Legacy scenario kept for smoke validation. | None |

## Running Tests Locally

```bash
# Ensure the gateway stack is running (docker compose up).
export BASE_URL=http://localhost:8000
k6 run tests/performance/retrieve_latency.js
k6 run tests/performance/ingest_throughput.js
k6 run tests/performance/concurrency.js
```

Override the target gateway by setting `BASE_URL`. When omitted, scripts default to `http://localhost:8000`.
