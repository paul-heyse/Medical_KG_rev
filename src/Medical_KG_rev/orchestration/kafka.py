"""Kafka client façade used for orchestrating ingestion jobs."""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field


@dataclass(order=True)
class KafkaMessage:
    """Internal representation of a Kafka message."""

    sort_key: tuple[int, float] = field(init=False, repr=False)
    topic: str
    value: dict[str, object]
    key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    attempts: int = 0
    available_at: float = field(default_factory=lambda: time.time())

    def __post_init__(self) -> None:
        # Priority is encoded in headers under x-priority. Higher value == higher priority.
        priority = int(self.headers.get("x-priority", "0"))
        self.sort_key = (-priority, self.available_at, self.timestamp)


class KafkaClient:
    """Small in-memory Kafka façade suitable for unit testing."""

    def __init__(self) -> None:
        self._topics: dict[str, list[KafkaMessage]] = defaultdict(list)
        self._health: dict[str, bool] = {"kafka": True, "zookeeper": True}

    # ------------------------------------------------------------------
    # Topic management
    # ------------------------------------------------------------------
    def create_topics(self, topics: Iterable[str]) -> None:
        """Ensure topics exist by initialising their queues."""

        for topic in topics:
            if topic not in self._topics:
                self._topics[topic] = []

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------
    def publish(
        self,
        topic: str,
        value: dict[str, object],
        *,
        key: str | None = None,
        headers: dict[str, str] | None = None,
        available_at: float | None = None,
        attempts: int = 0,
    ) -> KafkaMessage:
        if topic not in self._topics:
            raise ValueError(f"Topic '{topic}' has not been created")
        message = KafkaMessage(
            topic=topic,
            value=value,
            key=key,
            headers=headers or {},
            attempts=attempts,
        )
        if available_at is not None:
            message.available_at = available_at
        message.__post_init__()  # recompute sort key if available_at changed
        heapq.heappush(self._topics[topic], message)
        return message

    def consume(self, topic: str, *, max_messages: int | None = None) -> Iterator[KafkaMessage]:
        if topic not in self._topics:
            raise ValueError(f"Topic '{topic}' has not been created")
        consumed = 0
        now = time.time()
        buffer: list[KafkaMessage] = []
        while self._topics[topic] and (max_messages is None or consumed < max_messages):
            message = heapq.heappop(self._topics[topic])
            if message.available_at > now:
                buffer.append(message)
                break
            consumed += 1
            yield message
        for item in buffer:
            heapq.heappush(self._topics[topic], item)

    def pending(self, topic: str) -> int:
        if topic not in self._topics:
            raise ValueError(f"Topic '{topic}' has not been created")
        return len(self._topics[topic])

    def peek(self, topic: str) -> KafkaMessage | None:
        """Return the next message for a topic without consuming it."""

        if topic not in self._topics:
            raise ValueError(f"Topic '{topic}' has not been created")
        if not self._topics[topic]:
            return None
        return self._topics[topic][0]

    def discard(self, topic: str, *, key: str) -> int:
        if topic not in self._topics:
            raise ValueError(f"Topic '{topic}' has not been created")
        buffer: list[KafkaMessage] = []
        removed = 0
        while self._topics[topic]:
            message = heapq.heappop(self._topics[topic])
            if message.key == key:
                removed += 1
                continue
            buffer.append(message)
        for item in buffer:
            heapq.heappush(self._topics[topic], item)
        return removed

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def set_health(self, *, kafka: bool | None = None, zookeeper: bool | None = None) -> None:
        if kafka is not None:
            self._health["kafka"] = kafka
        if zookeeper is not None:
            self._health["zookeeper"] = zookeeper

    def health(self) -> dict[str, bool]:
        return dict(self._health)


__all__ = ["KafkaClient", "KafkaMessage"]
