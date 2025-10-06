"""Helpers for working with span ranges."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from Medical_KG_rev.models import Span


def merge_overlapping(spans: Sequence[Span]) -> list[Span]:
    """Merge overlapping spans into non-overlapping intervals."""

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
    """Return spans that fall entirely within bounds."""

    return [span for span in spans if span.start >= bounds.start and span.end <= bounds.end]
