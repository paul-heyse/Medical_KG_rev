# Implementation Tasks: Ingestion & Orchestration

## 1. Kafka Infrastructure

- [x] 1.1 Add Kafka and Zookeeper to docker-compose.yml
- [x] 1.2 Create Kafka topics (ingest.requests.v1, ingest.results.v1, mapping.events.v1)
- [x] 1.3 Implement KafkaClient wrapper with producer/consumer
- [x] 1.4 Add Kafka health checks
- [x] 1.5 Write Kafka integration tests

## 2. Job Ledger System

- [x] 2.1 Design ledger schema (job_id, doc_key, status, stage, metadata, timestamps)
- [x] 2.2 Implement Ledger class with Redis or PostgreSQL backend
- [x] 2.3 Add state transitions (queued → processing → completed/failed)
- [x] 2.4 Implement idempotency checking
- [x] 2.5 Add ledger query methods
- [x] 2.6 Write ledger tests

## 3. Orchestrator Service

- [x] 3.1 Create Orchestrator class with pipeline definitions
- [x] 3.2 Implement auto-pipeline (metadata → chunk → embed → index)
- [x] 3.3 Implement two-phase pipeline (metadata → PDF → MinerU → postpdf)
- [x] 3.4 Add multi-adapter chaining (OpenAlex → Unpaywall → CORE → MinerU)
- [x] 3.5 Implement job priority queuing
- [x] 3.6 Add retry logic with exponential backoff
- [x] 3.7 Implement dead letter queue for failed jobs
- [x] 3.8 Write orchestrator tests

## 4. Background Workers

- [x] 4.1 Create Worker base class
- [x] 4.2 Implement IngestWorker (consumes ingest.requests)
- [x] 4.3 Implement MappingWorker (consumes mapping.events)
- [x] 4.4 Add graceful shutdown handling
- [x] 4.5 Implement worker health checks
- [x] 4.6 Add worker metrics (jobs processed, failures)
- [x] 4.7 Write worker tests

## 5. Job Status API

- [x] 5.1 Add GET /jobs/{id} endpoint
- [x] 5.2 Implement SSE /jobs/{id}/events streaming
- [x] 5.3 Add job listing endpoint GET /jobs
- [x] 5.4 Implement job cancellation POST /jobs/{id}/cancel
- [x] 5.5 Write job API tests

## 6. Integration & Testing

- [x] 6.1 Create end-to-end orchestration tests
- [x] 6.2 Test two-phase pipeline with sample PDFs
- [x] 6.3 Test multi-adapter enrichment flow
- [x] 6.4 Add chaos testing (random failures)
- [x] 6.5 Performance test with concurrent jobs
