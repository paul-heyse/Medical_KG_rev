# Implementation Tasks: Ingestion & Orchestration

## 1. Kafka Infrastructure

- [ ] 1.1 Add Kafka and Zookeeper to docker-compose.yml
- [ ] 1.2 Create Kafka topics (ingest.requests.v1, ingest.results.v1, mapping.events.v1)
- [ ] 1.3 Implement KafkaClient wrapper with producer/consumer
- [ ] 1.4 Add Kafka health checks
- [ ] 1.5 Write Kafka integration tests

## 2. Job Ledger System

- [ ] 2.1 Design ledger schema (job_id, doc_key, status, stage, metadata, timestamps)
- [ ] 2.2 Implement Ledger class with Redis or PostgreSQL backend
- [ ] 2.3 Add state transitions (queued → processing → completed/failed)
- [ ] 2.4 Implement idempotency checking
- [ ] 2.5 Add ledger query methods
- [ ] 2.6 Write ledger tests

## 3. Orchestrator Service

- [ ] 3.1 Create Orchestrator class with pipeline definitions
- [ ] 3.2 Implement auto-pipeline (metadata → chunk → embed → index)
- [ ] 3.3 Implement two-phase pipeline (metadata → PDF → MinerU → postpdf)
- [ ] 3.4 Add multi-adapter chaining (OpenAlex → Unpaywall → CORE → MinerU)
- [ ] 3.5 Implement job priority queuing
- [ ] 3.6 Add retry logic with exponential backoff
- [ ] 3.7 Implement dead letter queue for failed jobs
- [ ] 3.8 Write orchestrator tests

## 4. Background Workers

- [ ] 4.1 Create Worker base class
- [ ] 4.2 Implement IngestWorker (consumes ingest.requests)
- [ ] 4.3 Implement MappingWorker (consumes mapping.events)
- [ ] 4.4 Add graceful shutdown handling
- [ ] 4.5 Implement worker health checks
- [ ] 4.6 Add worker metrics (jobs processed, failures)
- [ ] 4.7 Write worker tests

## 5. Job Status API

- [ ] 5.1 Add GET /jobs/{id} endpoint
- [ ] 5.2 Implement SSE /jobs/{id}/events streaming
- [ ] 5.3 Add job listing endpoint GET /jobs
- [ ] 5.4 Implement job cancellation POST /jobs/{id}/cancel
- [ ] 5.5 Write job API tests

## 6. Integration & Testing

- [ ] 6.1 Create end-to-end orchestration tests
- [ ] 6.2 Test two-phase pipeline with sample PDFs
- [ ] 6.3 Test multi-adapter enrichment flow
- [ ] 6.4 Add chaos testing (random failures)
- [ ] 6.5 Performance test with concurrent jobs
