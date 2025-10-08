from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, MutableMapping

import structlog

from Medical_KG_rev.config.settings import MineruSettings

logger = structlog.get_logger(__name__)


def _cuda_available() -> bool:
    """Best-effort probe to confirm CUDA support via PyTorch."""

    try:  # pragma: no cover - torch optional in CI
        import torch
    except Exception:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch runtime guard
        return False


class MineruCliError(RuntimeError):
    """Raised when the MinerU CLI invocation fails."""


@dataclass(slots=True)
class MineruCliInput:
    """Represents a single document to be processed by the CLI."""

    document_id: str
    content: bytes


@dataclass(slots=True)
class MineruCliOutput:
    """Represents a parsed output artefact produced by the CLI."""

    document_id: str
    path: Path


@dataclass(slots=True)
class MineruCliResult:
    outputs: list[MineruCliOutput]
    stdout: str
    stderr: str
    duration_seconds: float


class MineruCliBase:
    """Base class for MinerU CLI integrations."""

    def __init__(self, settings: MineruSettings) -> None:
        self._settings = settings

    def run_batch(
        self,
        batch: Iterable[MineruCliInput],
        *,
        gpu_id: int,
    ) -> MineruCliResult:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


class SubprocessMineruCli(MineruCliBase):
    """Executes the official MinerU CLI using subprocess management."""

    def __init__(
        self,
        settings: MineruSettings,
        *,
        timeout_seconds: int | None = None,
    ) -> None:
        super().__init__(settings)
        self._command = settings.cli_command
        self._timeout = timeout_seconds or settings.workers.timeout_seconds

    def _ensure_command(self) -> None:
        executable = self._command.split()[0]
        if shutil.which(executable) is None:
            raise MineruCliError(f"MinerU CLI '{self._command}' not found in PATH")

    def describe(self) -> str:
        return f"subprocess-cli(command={self._command})"

    def _prepare_environment(self, gpu_id: int) -> MutableMapping[str, str]:
        env: MutableMapping[str, str] = dict(os.environ)
        env.update(self._settings.environment())
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return env

    def _build_command(self, input_dir: Path, output_dir: Path) -> list[str]:
        command = self._command.split()
        command.extend(
            [
                "parse",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--format",
                "json",
                "--vram-limit",
                f"{self._settings.workers.vram_per_worker_gb}G",
            ]
        )
        return command

    def run_batch(
        self,
        batch: Iterable[MineruCliInput],
        *,
        gpu_id: int,
    ) -> MineruCliResult:
        self._ensure_command()
        inputs = list(batch)
        if not inputs:
            return MineruCliResult(outputs=[], stdout="", stderr="", duration_seconds=0.0)
        with tempfile.TemporaryDirectory(prefix="mineru-cli-") as workdir:
            input_dir = Path(workdir, "input")
            output_dir = Path(workdir, "output")
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            for item in inputs:
                path = input_dir / f"{item.document_id}.pdf"
                path.write_bytes(item.content)

            command = self._build_command(input_dir, output_dir)
            logger.bind(command=command, gpu_id=gpu_id).info("mineru.cli.invoke")
            start = time.monotonic()
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=self._prepare_environment(gpu_id),
            )
            duration = time.monotonic() - start
            if proc.returncode != 0:
                logger.bind(
                    returncode=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                ).error("mineru.cli.failed")
                raise MineruCliError(
                    f"MinerU CLI exited with code {proc.returncode}: {proc.stderr.strip()}"
                )

            outputs: list[MineruCliOutput] = []
            for item in inputs:
                output_path = output_dir / f"{item.document_id}.json"
                if not output_path.exists():
                    raise MineruCliError(
                        f"MinerU CLI did not produce output for document '{item.document_id}'"
                    )
                outputs.append(MineruCliOutput(document_id=item.document_id, path=output_path))
            return MineruCliResult(
                outputs=outputs,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_seconds=duration,
            )


class SimulatedMineruCli(MineruCliBase):
    """Deterministic fallback CLI used in environments without GPUs."""

    def describe(self) -> str:
        return "simulated-cli"

    def run_batch(
        self,
        batch: Iterable[MineruCliInput],
        *,
        gpu_id: int,
    ) -> MineruCliResult:
        outputs: list[MineruCliOutput] = []
        stdout_lines = []
        start = time.monotonic()
        for item in batch:
            try:
                text = item.content.decode("utf-8")
            except UnicodeDecodeError as exc:  # pragma: no cover - defensive
                raise MineruCliError("Simulated CLI expects UTF-8 encoded content") from exc
            pages = [page for page in text.split("\f") if page.strip()] or [text]
            reading_order = 0
            tables: list[dict[str, object]] = []
            blocks: list[dict[str, object]] = []
            for page_index, page in enumerate(pages, start=1):
                for raw_line in page.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    reading_order += 1
                    block_id = f"blk-{uuid.uuid4().hex[:8]}"
                    block: dict[str, object] = {
                        "id": block_id,
                        "page": page_index,
                        "type": "table" if "|" in line or "\t" in line else "paragraph",
                        "text": line,
                        "bbox": [0.05, reading_order * 0.02, 0.95, reading_order * 0.02 + 0.015],
                        "confidence": 0.9,
                        "reading_order": reading_order,
                    }
                    if block["type"] == "table":
                        values = [col.strip() for col in line.split("|") if col.strip()]
                        table_id = f"tbl-{uuid.uuid4().hex[:8]}"
                        table_cells = [
                            {
                                "row": 0,
                                "column": index,
                                "content": value,
                                "rowspan": 1,
                                "colspan": 1,
                            }
                            for index, value in enumerate(values)
                        ]
                        tables.append(
                            {
                                "id": table_id,
                                "page": page_index,
                                "caption": None,
                                "bbox": [0.05, 0.05, 0.95, 0.25],
                                "headers": values,
                                "cells": table_cells,
                            }
                        )
                        block["table_id"] = table_id
                    blocks.append(block)
            payload = {
                "document_id": item.document_id,
                "blocks": blocks,
                "tables": tables,
                "figures": [],
                "equations": [],
                "metadata": {"simulated": True},
            }
            fd, path = tempfile.mkstemp(prefix=f"mineru-sim-{item.document_id}-", suffix=".json")
            os.close(fd)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            outputs.append(MineruCliOutput(document_id=item.document_id, path=Path(path)))
            stdout_lines.append(f"simulated:{item.document_id}")

        duration = time.monotonic() - start
        return MineruCliResult(
            outputs=outputs,
            stdout="\n".join(stdout_lines),
            stderr="",
            duration_seconds=duration,
        )


def create_cli(settings: MineruSettings) -> MineruCliBase:
    """Factory that resolves the appropriate CLI implementation."""

    if not _cuda_available():
        message = "CUDA runtime support unavailable for MinerU CLI execution"
        if settings.simulate_if_unavailable:
            logger.bind(reason="cuda-unavailable").warning("mineru.cli.fallback")
            return SimulatedMineruCli(settings)
        raise MineruCliError(message)

    try:
        cli = SubprocessMineruCli(settings)
        cli._ensure_command()
        logger.info("mineru.cli.selected", impl=cli.describe())
        return cli
    except MineruCliError:
        if not settings.simulate_if_unavailable:
            raise
        logger.bind(reason="command-not-found").warning("mineru.cli.fallback")
        return SimulatedMineruCli(settings)


__all__ = [
    "MineruCliBase",
    "MineruCliError",
    "MineruCliInput",
    "MineruCliOutput",
    "MineruCliResult",
    "SimulatedMineruCli",
    "SubprocessMineruCli",
    "create_cli",
]
