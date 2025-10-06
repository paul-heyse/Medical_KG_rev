from __future__ import annotations

import time

from Medical_KG_rev.gateway.sse.manager import EventStreamManager
from Medical_KG_rev.orchestration import KafkaClient, OrchestrationError, Orchestrator
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.orchestrator import (
    INGEST_REQUESTS_TOPIC,
    INGEST_RESULTS_TOPIC,
    MAPPING_EVENTS_TOPIC,
    Pipeline,
    PipelineStage,
)


def create_orchestrator() -> Orchestrator:
    kafka = KafkaClient()
    kafka.create_topics(
        [INGEST_REQUESTS_TOPIC, INGEST_RESULTS_TOPIC, MAPPING_EVENTS_TOPIC, "ingest.deadletter.v1"]
    )
    ledger = JobLedger()
    events = EventStreamManager()
    return Orchestrator(kafka, ledger, events)


def test_submit_job_selects_pipeline() -> None:
    orchestrator = create_orchestrator()
    entry = orchestrator.submit_job(
        tenant_id="tenant",
        dataset="papers",
        item={"id": "1", "document_type": "pdf"},
    )
    assert entry.pipeline == "two-phase"
    assert orchestrator.kafka.pending(INGEST_REQUESTS_TOPIC) == 1


def test_execute_pipeline_emits_mapping_event() -> None:
    orchestrator = create_orchestrator()
    entry = orchestrator.submit_job(
        tenant_id="tenant",
        dataset="papers",
        item={"id": "2", "adapter_chain": True, "open_access": True},
    )
    message = next(orchestrator.kafka.consume(INGEST_REQUESTS_TOPIC))
    orchestrator.execute_pipeline(entry.job_id, message.value)

    mapping_events = list(orchestrator.kafka.consume(MAPPING_EVENTS_TOPIC))
    assert mapping_events and mapping_events[0].value["job_id"] == entry.job_id


def test_pipeline_failure_requeues_with_backoff() -> None:
    orchestrator = create_orchestrator()
    entry = orchestrator.submit_job(
        tenant_id="tenant",
        dataset="papers",
        item={"id": "3"},
    )

    def failing_handler(*_: object, **__: object) -> dict:
        raise RuntimeError("boom")

    orchestrator.pipelines["auto"] = Pipeline(
        name="auto", stages=[PipelineStage("boom", failing_handler)]
    )

    message = next(orchestrator.kafka.consume(INGEST_REQUESTS_TOPIC))
    start = time.time()
    try:
        orchestrator.execute_pipeline(entry.job_id, message.value)
    except OrchestrationError:
        pass

    assert orchestrator.kafka.pending(INGEST_REQUESTS_TOPIC) >= 1
    requeued = orchestrator.kafka.peek(INGEST_REQUESTS_TOPIC)
    assert requeued is not None
    delay = requeued.available_at - start
    assert delay >= 1.0
    assert orchestrator.ledger.get(entry.job_id).attempts == 1
