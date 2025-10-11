"""Load test script for the vLLM server."""

from __future__ import annotations

import asyncio
import time
from statistics import mean, quantiles

# MinerU VLLM client removed - using Docling VLM services instead


async def single_request(client, request_id: int) -> float:
    """Issue a single chat completion request and return latency in seconds."""
    start = time.time()
    await client.chat_completion(
        messages=[{"role": "user", "content": f"Summarise request {request_id}"}],
        max_tokens=128,
    )
    return time.time() - start


async def load_test(concurrency: int = 10, total_requests: int = 100) -> None:
    """Run a simple fixed-concurrency load test against the vLLM server."""
    client = VLLMClient(base_url="http://localhost:8000")
    async with client:
        durations: list[float] = []
        for batch_start in range(0, total_requests, concurrency):
            batch_end = min(batch_start + concurrency, total_requests)
            batch = range(batch_start, batch_end)
            results = await asyncio.gather(
                *(single_request(client, request_id) for request_id in batch)
            )
            durations.extend(results)

    p50, p95, p99 = quantiles(durations, n=100)[49::46]
    print("\nLoad Test Results:")
    print(f"  Concurrency: {concurrency}")
    print(f"  Total Requests: {total_requests}")
    print(f"  Mean Latency: {mean(durations):.2f}s")
    print(f"  P50 Latency: {p50:.2f}s")
    print(f"  P95 Latency: {p95:.2f}s")
    print(f"  P99 Latency: {p99:.2f}s")
    assert p95 < 10.0, f"P95 latency {p95:.2f}s exceeds 10s threshold"


if __name__ == "__main__":
    asyncio.run(load_test())
