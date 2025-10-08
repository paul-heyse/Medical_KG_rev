"""Stage plugin implementations bundled with the orchestration runtime."""

from .builtin import CoreStagePlugin, PdfTwoPhasePlugin

__all__ = ["CoreStagePlugin", "PdfTwoPhasePlugin"]
