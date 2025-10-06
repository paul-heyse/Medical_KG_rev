from Medical_KG_rev.models.ir import Span
from Medical_KG_rev.utils.spans import merge_overlapping, spans_within


def test_merge_overlapping():
    spans = [Span(start=0, end=5), Span(start=4, end=10), Span(start=12, end=14)]
    merged = merge_overlapping(spans)
    assert merged[0].end == 10
    assert len(merged) == 2


def test_spans_within():
    bounds = Span(start=0, end=10)
    spans = [Span(start=1, end=3), Span(start=9, end=11)]
    inside = spans_within(bounds, spans)
    assert len(inside) == 1
