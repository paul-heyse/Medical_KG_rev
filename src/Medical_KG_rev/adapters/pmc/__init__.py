"""PubMed Central adapter for full-text XML retrieval.

This module provides adapters for the PubMed Central (PMC) API,
which provides access to full-text XML documents from biomedical
and life science journals.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from .adapter import PMCAdapter


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = ["PMCAdapter"]
