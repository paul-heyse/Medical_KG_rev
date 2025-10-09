"""OpenAlex adapter package.

This module provides adapters for the OpenAlex scholarly works API,
which aggregates metadata about academic papers, authors, institutions,
and citations.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from .adapter import OpenAlexAdapter

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = ["OpenAlexAdapter"]
