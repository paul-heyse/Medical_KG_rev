"""CORE adapter for open research papers.

This module provides adapters for the CORE (COnnecting REpositories) API,
which aggregates open access research papers from repositories worldwide.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from .adapter import COREAdapter


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = ["COREAdapter"]
