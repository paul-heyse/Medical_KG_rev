"""Span range helpers used by chunking and highlighting pipelines.

Key Responsibilities:
    - Provide utilities for merging overlapping spans
    - Filter spans relative to bounding regions during extraction workflows

Collaborators:
    - Upstream: Chunking, parsing, and highlighting modules supply span lists
    - Downstream: Outputs feed into rendering and metadata summarisation logic

Side Effects:
    - None; functions return new sequences without mutating inputs

Thread Safety:
    - Thread-safe; purely functional helpers
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from Medical_KG_rev.models import Span

# ==============================================================================
# PUBLIC HELPERS
# ==============================================================================


def merge_overlapping(spans: Sequence[Span]) -> list[Span]:
    """Merge overlapping spans into non-overlapping intervals.

    Args:
        spans: Sequence of spans sorted arbitrarily.

    Returns:
        New list of spans with overlaps coalesced.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: span.start)
    merged: list[Span] = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start <= previous.end:
            end = max(previous.end, current.end)
            merged[-1] = Span(start=previous.start, end=end, text=None)
        else:
            merged.append(current)
    return merged


def spans_within(bounds: Span, spans: Iterable[Span]) -> list[Span]:
    """Return spans that fall entirely within ``bounds``.

    Args:
        bounds: Span denoting the inclusive region of interest.
        spans: Iterable of spans to evaluate against ``bounds``.

    Returns:
        List of spans whose start and end positions lie within ``bounds``.
    """
    return [span for span in spans if span.start >= bounds.start and span.end <= bounds.end]
