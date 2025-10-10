"""Custom exceptions for the adapter plugin framework.

This module defines custom exception classes for the adapter plugin
framework, providing specific error types for plugin-related failures.

The module defines:
- AdapterPluginError: Base exception for adapter plugin operations

Architecture:
- Simple exception hierarchy for plugin errors
- Inherits from RuntimeError for compatibility
- Provides clear error categorization

Thread Safety:
- Exception classes are thread-safe.

Performance:
- Lightweight exception definitions with minimal overhead.

Examples:
    try:
        plugin.initialize()
    except AdapterPluginError as e:
        logger.error("Plugin initialization failed", exc_info=e)

"""

# IMPORTS
from __future__ import annotations


# EXCEPTION CLASSES
class AdapterPluginError(RuntimeError):
    """Raised when adapter plugin operations fail.

    This exception is raised when adapter plugin operations encounter
    errors during initialization, execution, or cleanup phases.

    Thread Safety:
        Thread-safe exception class.

    Examples:
        try:
            plugin.load_config()
        except AdapterPluginError as e:
            handle_plugin_error(e)

    """


# EXPORTS
__all__ = ["AdapterPluginError"]
