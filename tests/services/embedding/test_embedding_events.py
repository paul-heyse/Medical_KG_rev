from Medical_KG_rev.services.embedding.events import EmbeddingEventEmitter


def test_event_emitter_publishes_lifecycle_events() -> None:
    class StubKafka:
        def __init__(self) -> None:
            self.topics: dict[str, list[dict[str, object]]] = {}

        def create_topics(self, topics):
            for topic in topics:
                self.topics.setdefault(topic, [])

        def publish(
            self, topic: str, value: dict[str, object], *, key=None, headers=None
        ):
            self.topics.setdefault(topic, []).append(value)

        def consume(self, topic: str):
            while self.topics.get(topic):
                yield type("Message", (), {"value": self.topics[topic].pop(0)})()

    kafka = StubKafka()
    emitter = EmbeddingEventEmitter(kafka=kafka, topic="embedding.test.v1")

    emitter.emit_started(
        tenant_id="tenant-a",
        namespace="single_vector.qwen3.4096.v1",
        provider="vllm",
        batch_size=2,
        correlation_id="corr-123",
    )
    started = next(kafka.consume("embedding.test.v1"))
    assert started.value["type"] == "com.medical-kg.embedding.started"
    assert started.value["data"]["batch_size"] == 2

    emitter.emit_completed(
        tenant_id="tenant-a",
        namespace="single_vector.qwen3.4096.v1",
        provider="vllm",
        correlation_id="corr-123",
        duration_ms=12.5,
        generated=2,
        cache_hits=1,
        cache_misses=1,
    )
    completed = next(kafka.consume("embedding.test.v1"))
    assert completed.value["data"]["embeddings_generated"] == 2
    assert completed.value["data"]["cache_hits"] == 1

    emitter.emit_failed(
        tenant_id="tenant-a",
        namespace="single_vector.qwen3.4096.v1",
        provider="vllm",
        correlation_id="corr-123",
        error_type="TimeoutError",
        message="vLLM timeout",
    )
    failed = next(kafka.consume("embedding.test.v1"))
    assert failed.value["type"] == "com.medical-kg.embedding.failed"
    assert failed.value["data"]["error_type"] == "TimeoutError"
