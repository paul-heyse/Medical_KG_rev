"""PDF gate stage implementation for conditional pipeline progression."""

from __future__ import annotations

import time
from typing import Any

import structlog
from Medical_KG_rev.config.settings import get_settings
from Medical_KG_rev.orchestration.stages.contracts import (
    GateDecision,
    GateStage,
    PipelineState,
    StageContext,
)
from Medical_KG_rev.orchestration.stages.plugin_manager import StagePluginContext

logger = structlog.get_logger(__name__)


class PdfGateStage(GateStage):
    """Gate stage that controls PDF pipeline progression based on ledger state."""

    def __init__(self, gate_name: str = "pdf-ir-gate") -> None:
        self._gate_name = gate_name
        self._ledger: Any = None
        settings = get_settings()
        self._backend = settings.feature_flags.pdf_processing_backend
        self._ledger_ready_key = (
            "pdf_vlm_ready" if self._backend == "docling_vlm" else "pdf_ir_ready"
        )
        self._state_ready_attr = "vlm_ready" if self._backend == "docling_vlm" else "ir_ready"

    def initialise(self, context: StagePluginContext) -> None:
        """Initialize stage with ledger resources."""
        self._ledger = context.get("ledger")
        if self._ledger is None:
            logger.warning(
                "pdf_gate_stage.no_ledger", message="Ledger not available, gate will always pass"
            )

    def execute(self, ctx: StageContext, state: PipelineState) -> GateDecision:
        """Check if PDF processing is ready to proceed."""
        # Check if we have PDF downloads
        downloads = state.downloads
        if not downloads:
            logger.debug(
                "pdf_gate_stage.no_downloads",
                tenant_id=state.tenant_id,
                gate_name=self._gate_name,
            )
            return GateDecision(
                name=self._gate_name,
                ready=False,
                metadata={"reason": "no_downloads", "timestamp": time.time()},
            )

        # Check ledger state if available
        if self._ledger:
            try:
                # Check if PDFs are marked as downloaded in ledger
                job_id = ctx.job_id or state.job_id
                if job_id:
                    ledger_state = self._ledger.get_job_state(job_id)
                    if ledger_state:
                        pdf_downloaded = ledger_state.get("pdf_downloaded", False)
                        backend_ready = ledger_state.get(self._ledger_ready_key, False)

                        if not pdf_downloaded:
                            logger.debug(
                                "pdf_gate_stage.pdf_not_downloaded",
                                tenant_id=state.tenant_id,
                                job_id=job_id,
                                gate_name=self._gate_name,
                            )
                            return GateDecision(
                                name=self._gate_name,
                                ready=False,
                                metadata={
                                    "reason": "pdf_not_downloaded",
                                    "job_id": job_id,
                                    "timestamp": time.time(),
                                },
                            )

                        if not backend_ready:
                            logger.debug(
                                "pdf_gate_stage.vlm_not_ready"
                                if self._backend == "docling_vlm"
                                else "pdf_gate_stage.pdf_ir_not_ready",
                                tenant_id=state.tenant_id,
                                job_id=job_id,
                                gate_name=self._gate_name,
                            )
                            return GateDecision(
                                name=self._gate_name,
                                ready=False,
                                metadata={
                                    "reason": "pdf_vlm_not_ready"
                                    if self._backend == "docling_vlm"
                                    else "pdf_ir_not_ready",
                                    "job_id": job_id,
                                    "timestamp": time.time(),
                                },
                            )
            except Exception as e:
                logger.warning(
                    "pdf_gate_stage.ledger_error",
                    tenant_id=state.tenant_id,
                    error=str(e),
                    gate_name=self._gate_name,
                )
                # Continue without ledger check

        # Check PDF gate state in pipeline state
        if hasattr(state, "pdf_gate"):
            if not state.pdf_gate.downloaded:
                logger.debug(
                    "pdf_gate_stage.state_not_downloaded",
                    tenant_id=state.tenant_id,
                    gate_name=self._gate_name,
                )
                return GateDecision(
                    name=self._gate_name,
                    ready=False,
                    metadata={
                        "reason": "state_not_downloaded",
                        "timestamp": time.time(),
                    },
                )

            is_ready = getattr(state.pdf_gate, self._state_ready_attr, False)
            if not is_ready:
                logger.debug(
                    "pdf_gate_stage.state_vlm_not_ready"
                    if self._backend == "docling_vlm"
                    else "pdf_gate_stage.state_ir_not_ready",
                    tenant_id=state.tenant_id,
                    gate_name=self._gate_name,
                )
                return GateDecision(
                    name=self._gate_name,
                    ready=False,
                    metadata={
                        "reason": "state_vlm_not_ready"
                        if self._backend == "docling_vlm"
                        else "state_ir_not_ready",
                        "timestamp": time.time(),
                    },
                )

        # All checks passed
        logger.info(
            "pdf_gate_stage.passed",
            tenant_id=state.tenant_id,
            gate_name=self._gate_name,
            download_count=len(downloads),
        )

        return GateDecision(
            name=self._gate_name,
            ready=True,
            metadata={
                "reason": "all_checks_passed",
                "download_count": len(downloads),
                "timestamp": time.time(),
            },
        )


class SimplePdfGateStage(GateStage):
    """Simplified gate stage that only checks for PDF downloads."""

    def __init__(self, gate_name: str = "pdf-gate") -> None:
        self._gate_name = gate_name

    def initialise(self, context: StagePluginContext) -> None:
        """Initialize stage (no external dependencies)."""

    def execute(self, ctx: StageContext, state: PipelineState) -> GateDecision:
        """Check if PDF downloads are available."""
        downloads = state.downloads
        ready = len(downloads) > 0

        logger.debug(
            "simple_pdf_gate_stage.check",
            tenant_id=state.tenant_id,
            gate_name=self._gate_name,
            ready=ready,
            download_count=len(downloads),
        )

        return GateDecision(
            name=self._gate_name,
            ready=ready,
            metadata={
                "reason": "downloads_available" if ready else "no_downloads",
                "download_count": len(downloads),
                "timestamp": time.time(),
            },
        )
