from __future__ import annotations

import time

from Medical_KG_rev.orchestration.kafka import KafkaClient


def test_create_topics_and_publish_order() -> None:
    client = KafkaClient()
    client.create_topics(["ingest.requests.v1"])

    client.publish("ingest.requests.v1", {"id": 1}, headers={"x-priority": "0"})
    client.publish("ingest.requests.v1", {"id": 2}, headers={"x-priority": "2"})
    client.publish("ingest.requests.v1", {"id": 3}, headers={"x-priority": "1"})

    messages = list(client.consume("ingest.requests.v1"))
    assert [message.value["id"] for message in messages] == [2, 3, 1]


def test_delayed_messages_wait_until_available() -> None:
    client = KafkaClient()
    client.create_topics(["ingest.requests.v1"])

    now = time.time()
    client.publish("ingest.requests.v1", {"id": 1}, available_at=now + 1)
    client.publish("ingest.requests.v1", {"id": 2}, available_at=now)

    first_batch = list(client.consume("ingest.requests.v1"))
    assert [message.value["id"] for message in first_batch] == [2]

    time.sleep(1.1)
    second_batch = list(client.consume("ingest.requests.v1"))
    assert [message.value["id"] for message in second_batch] == [1]


def test_health_flags_toggle() -> None:
    client = KafkaClient()
    assert client.health() == {"kafka": True, "zookeeper": True}

    client.set_health(kafka=False)
    assert client.health()["kafka"] is False

    client.set_health(zookeeper=False)
    assert client.health()["zookeeper"] is False


def test_discard_removes_messages_by_key() -> None:
    client = KafkaClient()
    client.create_topics(["ingest.requests.v1"])
    client.publish("ingest.requests.v1", {"id": 1}, key="job-1")
    client.publish("ingest.requests.v1", {"id": 2}, key="job-2")

    removed = client.discard("ingest.requests.v1", key="job-1")
    assert removed == 1
    remaining = list(client.consume("ingest.requests.v1"))
    assert [message.value["id"] for message in remaining] == [2]
