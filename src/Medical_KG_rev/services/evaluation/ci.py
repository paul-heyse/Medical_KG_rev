"""Helpers for integrating evaluation checks into CI pipelines."""

from __future__ import annotations


def enforce_recall_threshold(
    baseline: float,
    current: float,
    *,
    tolerance: float = 0.05,
) -> None:
    """Raise ``RuntimeError`` if Recall@10 regresses beyond the tolerated drop."""

    if baseline <= 0:
        return
    drop = baseline - current
    if drop <= 0:
        return
    if drop / baseline > tolerance:
        raise RuntimeError(
            f"Recall@10 drop of {drop / baseline:.1%} exceeds tolerance of {tolerance:.0%}"
        )


__all__ = ["enforce_recall_threshold"]
