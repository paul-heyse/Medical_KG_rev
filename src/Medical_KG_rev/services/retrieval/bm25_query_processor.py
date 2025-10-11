"""BM25 query processing contract used by retrieval components.

Key Responsibilities:
    - Define the abstract scoring entry point for BM25 implementations
    - Serve as the shared contract for index-backed and in-memory processors

Collaborators:
    - Upstream: Retrieval services and coordinators requesting BM25 scoring
    - Downstream: Concrete BM25 implementations providing actual scores

Side Effects:
    - None; this module exposes the contract only

Thread Safety:
    - Thread-safe; implementations should ensure safe concurrent use
"""

from __future__ import annotations

from typing import List

# ==============================================================================
# PUBLIC API
# ==============================================================================


def process_query(query: str, documents: List[str]) -> List[float]:
    """Score documents against ``query`` using BM25 logic.

    This placeholder defines the expected interface for BM25 processors and
    should be implemented by retrieval backends that provide actual scoring.

    Args:
        query: Search query to score documents against.
        documents: List of document texts to score.

    Returns:
        List of BM25 scores, one per document in the same order.

    Raises:
        NotImplementedError: Always raised by this placeholder implementation.
        ValueError: If query is empty or documents list is empty.
        RuntimeError: If BM25 processing fails.
    """
    raise NotImplementedError(
        "BM25 query processing not implemented. "
        "This function requires a real BM25 implementation. "
        "Please implement or configure a proper BM25 query processor."
    )


__all__ = ["process_query"]
