"""Helpers for integrating evaluation checks into CI pipelines.

This module provides utilities for integrating retrieval evaluation checks
into continuous integration pipelines. It includes functions for enforcing
performance thresholds and detecting regressions in retrieval quality.

Key Responsibilities:
    - Enforce recall performance thresholds in CI
    - Detect performance regressions beyond tolerance
    - Provide clear error messages for CI failures
    - Support configurable tolerance levels

Collaborators:
    - Upstream: CI pipelines, evaluation workflows
    - Downstream: CI failure reporting, performance monitoring

Side Effects:
    - Raises RuntimeError for performance regressions
    - Terminates CI pipeline on threshold violations

Thread Safety:
    - Thread-safe: Stateless functions with no shared state

Performance Characteristics:
    - O(1) threshold checking operations
    - Minimal overhead for CI integration

Example:
-------
    >>> from Medical_KG_rev.services.evaluation.ci import enforce_recall_threshold
    >>> baseline_recall = 0.85
    >>> current_recall = 0.82
    >>> enforce_recall_threshold(baseline_recall, current_recall, tolerance=0.05)

"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================


# ==============================================================================
# CI INTEGRATION FUNCTIONS
# ==============================================================================


def enforce_recall_threshold(
    baseline: float,
    current: float,
    *,
    tolerance: float = 0.05,
) -> None:
    """Raise ``RuntimeError`` if Recall@10 regresses beyond the tolerated drop.

    Args:
    ----
        baseline: Baseline recall performance value.
        current: Current recall performance value.
        tolerance: Maximum tolerated performance drop (default: 0.05).

    Raises:
    ------
        RuntimeError: If performance regression exceeds tolerance.

    """
    if baseline <= 0:
        return
    drop = baseline - current
    if drop <= 0:
        return
    if drop / baseline > tolerance:
        raise RuntimeError(
            f"Recall@10 drop of {drop / baseline:.1%} exceeds tolerance of {tolerance:.0%}"
        )


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = ["enforce_recall_threshold"]
