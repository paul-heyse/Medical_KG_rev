"""Utilities for benchmarking sentence segmentation backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

Segment = tuple[int, int, str]
Segmenter = Callable[[str], Iterable[Segment]]


@dataclass(frozen=True, slots=True)
class SegmenterBenchmark:
    """Summary of a benchmarking run for a single sentence segmenter."""

    name: str
    documents: int
    sentences: int
    duration_seconds: float
    throughput_docs_per_second: float
    throughput_sentences_per_second: float


def benchmark_segmenters(
    segmenters: Mapping[str, Segmenter],
    corpus: Sequence[str],
    *,
    repeats: int = 1,
    timer: Callable[[], float] | None = None,
) -> list[SegmenterBenchmark]:
    """Measure throughput for multiple sentence segmenters.

    Args:
        segmenters: Mapping of readable names to callables that accept a text
            string and return sentence spans.
        corpus: Iterable of documents to segment.
        repeats: Number of times to process *corpus* for each segmenter.
        timer: Optional callable returning the current time. Defaults to
            :func:`time.perf_counter`.

    Returns:
        Sorted list of :class:`SegmenterBenchmark` instances ordered by
        descending document throughput.

    """
    if repeats <= 0:
        raise ValueError("repeats must be a positive integer")

    clock = timer or perf_counter
    documents_per_run = len(corpus)
    results: list[SegmenterBenchmark] = []

    for name, segmenter in segmenters.items():
        total_sentences = 0
        start_time = clock()
        for _ in range(repeats):
            for text in corpus:
                try:
                    spans = segmenter(text)
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuntimeError(f"segmenter '{name}' failed while processing input") from exc
                if not isinstance(spans, Iterable):
                    raise TypeError(f"segmenter '{name}' returned non-iterable spans")
                total_sentences += sum(1 for _ in spans)
        duration = max(clock() - start_time, 0.0)
        documents_processed = documents_per_run * repeats
        if duration > 0:
            docs_per_second = documents_processed / duration if documents_processed else 0.0
            sentences_per_second = total_sentences / duration if total_sentences else 0.0
        else:
            docs_per_second = 0.0
            sentences_per_second = 0.0
        results.append(
            SegmenterBenchmark(
                name=name,
                documents=documents_processed,
                sentences=total_sentences,
                duration_seconds=duration,
                throughput_docs_per_second=docs_per_second,
                throughput_sentences_per_second=sentences_per_second,
            )
        )

    results.sort(key=lambda result: result.throughput_docs_per_second, reverse=True)
    return results


__all__ = ["SegmenterBenchmark", "benchmark_segmenters"]
