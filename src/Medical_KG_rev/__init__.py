"""Medical Knowledge Graph revision - main module.

Key Responsibilities:
    - Provide the main entry point for the Medical KG system
    - Export core functionality and utilities
    - Serve as the root package for all Medical_KG_rev modules

Collaborators:
    - Upstream: External applications import from this module
    - Downstream: All Medical_KG_rev submodules and packages

Side Effects:
    - None: This module is purely functional

Thread Safety:
    - Thread-safe: All functions can be called from multiple threads

Performance Characteristics:
    - Minimal overhead: Simple ping function for health checks

Example:
    >>> from Medical_KG_rev import ping
    >>> ping()
    'pong'
"""


def ping() -> str:
    """Return a simple health check response.

    Args:
        None

    Returns:
        A string indicating the service is alive.

    Raises:
        None

    Example:
        >>> ping()
        'pong'
    """
    return "pong"
