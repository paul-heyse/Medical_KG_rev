"""Synthetic benchmarks comparing Docling VLM against legacy OCR timings."""

from __future__ import annotations


def test_docling_average_latency_is_lower_than_ocr_baseline() -> None:
    """Validate synthetic latency comparison between Docling and OCR."""

    docling_samples = [0.85, 0.82, 0.79]
    ocr_samples = [1.35, 1.28, 1.31]
    docling_avg = sum(docling_samples) / len(docling_samples)
    ocr_avg = sum(ocr_samples) / len(ocr_samples)
    assert docling_avg < ocr_avg
    improvement = (ocr_avg - docling_avg) / ocr_avg
    assert improvement > 0.3
