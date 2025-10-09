"""Stage plugin implementations bundled with the orchestration runtime.

This package provides built-in stage plugin implementations. The plugin framework
classes (StagePlugin, StagePluginManager) are imported from the sibling plugins.py module.
"""

from .builtin import CoreStagePlugin, PdfTwoPhasePlugin

# The framework classes live in ../plugins.py (not this __init__.py)
# Python's import system will resolve Medical_KG_rev.orchestration.stages.plugins
# to this package directory, so we cannot directly import from the .py file of the same name.
# Users should import framework classes from Medical_KG_rev.orchestration.stages.plugins
# (this module) which are defined in Medical_KG_rev.orchestration.stages.plugins (the .py file).
# To resolve this, consumers should use the plugin_manager module instead, or import
# directly from the plugins.py file using:
#   from Medical_KG_rev.orchestration.stages import plugins as plugins_file

__all__ = [
    "CoreStagePlugin",
    "PdfTwoPhasePlugin",
]
