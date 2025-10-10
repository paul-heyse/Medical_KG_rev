"""MinerU CLI wrapper and integration layer.

This module provides wrapper classes for integrating with the MinerU CLI,
including subprocess execution, simulated fallback, and factory functions
for CLI selection based on environment capabilities.

Key Components:
    - MineruCliBase: Abstract base class for CLI implementations
    - SubprocessMineruCli: Real CLI execution via subprocess
    - SimulatedMineruCli: Fallback implementation for testing/development
    - Data models: Input/output structures for CLI communication
    - Factory function: CLI selection based on availability

Responsibilities:
    - Execute MinerU CLI commands via subprocess
    - Provide fallback simulation for environments without GPU
    - Handle temporary file management for CLI I/O
    - Manage CLI timeouts and error handling
    - Support batch processing of multiple documents

Collaborators:
    - MinerU service implementation
    - Configuration settings
    - Temporary file system
    - Subprocess execution environment

Side Effects:
    - Creates temporary directories and files
    - Executes external CLI processes
    - Logs CLI execution details

Thread Safety:
    - Thread-safe: Uses temporary directories with unique names
    - Subprocess execution is isolated per call

Performance Characteristics:
    - GPU-accelerated processing via real CLI
    - Fast simulation for development/testing
    - Configurable timeouts prevent hanging processes
    - Batch processing reduces overhead

Example:
    >>> cli = create_cli(settings)
    >>> inputs = [MineruCliInput(document_id="doc1", content=pdf_bytes)]
    >>> result = cli.run_batch(inputs)
    >>> print(f"Processed {len(result.outputs)} documents")

"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import structlog
from Medical_KG_rev.config.settings import MineruSettings

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logger = structlog.get_logger(__name__)


# ==============================================================================
# EXCEPTION CLASSES
# ==============================================================================

class MineruCliError(RuntimeError):
    """Raised when the MinerU CLI invocation fails.

    This exception is raised when the MinerU CLI cannot be executed
    or fails during execution. Common causes include missing CLI
    executable, timeout, or non-zero exit codes.

    Example:
        >>> try:
        ...     cli.run_batch(inputs)
        ... except MineruCliError as e:
        ...     print(f"CLI failed: {e}")

    """


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass(slots=True)
class MineruCliInput:
    """Represents a single document to be processed by the CLI.

    Contains the document identifier and PDF content bytes
    required for MinerU CLI processing.

    Attributes:
        document_id: Unique identifier for the document
        content: PDF content bytes to be processed

    Invariants:
        - document_id is never empty
        - content is not empty

    Example:
        >>> input_item = MineruCliInput(
        ...     document_id="doc_123",
        ...     content=pdf_bytes
        ... )

    """

    document_id: str
    content: bytes


@dataclass(slots=True)
class MineruCliOutput:
    """Represents a parsed output artifact produced by the CLI.

    Contains the document identifier, path, and JSON content from
    MinerU CLI processing.

    Attributes:
        document_id: Unique identifier for the processed document
        path: Path to the generated JSON output file
        json_content: The JSON content as a string (for temp file handling)

    Invariants:
        - document_id is never empty
        - json_content is valid JSON string

    Example:
        >>> output = MineruCliOutput(
        ...     document_id="doc_123",
        ...     path=Path("/tmp/output/doc_123.json"),
        ...     json_content='{"key": "value"}'
        ... )

    """

    document_id: str
    path: Path
    json_content: str


@dataclass(slots=True)
class MineruCliResult:
    """Result of MinerU CLI batch processing operation.

    Contains all outputs, execution details, and timing information
    from a batch processing operation.

    Attributes:
        outputs: List of generated output artifacts
        stdout: Standard output from CLI execution
        stderr: Standard error from CLI execution
        duration_seconds: Processing duration in seconds

    Invariants:
        - duration_seconds >= 0.0
        - outputs list corresponds to input documents

    Example:
        >>> result = MineruCliResult(
        ...     outputs=[output1, output2],
        ...     stdout="Processing completed",
        ...     stderr="",
        ...     duration_seconds=1.5
        ... )

    """

    outputs: list[MineruCliOutput]
    stdout: str
    stderr: str
    duration_seconds: float


# ==============================================================================
# CLI IMPLEMENTATIONS
# ==============================================================================

class MineruCliBase:
    """Base class for MinerU CLI integrations.

    Abstract base class defining the interface for MinerU CLI
    implementations. Provides common functionality and defines
    the contract for CLI execution.

    Attributes:
        _settings: MinerU configuration settings

    Invariants:
        - _settings is never None

    Thread Safety:
        - Thread-safe: Immutable settings reference

    Example:
        >>> class CustomCli(MineruCliBase):
        ...     def run_batch(self, batch):
        ...         # Custom implementation
        ...         pass
        ...     def describe(self):
        ...         return "custom-cli"

    """

    def __init__(self, settings: MineruSettings) -> None:
        """Initialize CLI base with settings.

        Args:
            settings: MinerU configuration settings

        """
        self._settings = settings

    def run_batch(
        self,
        batch: Iterable[MineruCliInput],
    ) -> MineruCliResult:
        """Execute batch processing of documents.

        Args:
            batch: Iterable of documents to process

        Returns:
            CLI result containing outputs and execution details

        Raises:
            NotImplementedError: Must be implemented by subclasses

        """
        raise NotImplementedError

    def describe(self) -> str:
        """Describe the CLI implementation.

        Returns:
            String description of the CLI implementation

        Raises:
            NotImplementedError: Must be implemented by subclasses

        """
        raise NotImplementedError


class SubprocessMineruCli(MineruCliBase):
    """Executes the official MinerU CLI using subprocess management.

    Real implementation that executes the MinerU CLI via subprocess,
    handling temporary file management, command execution, and output
    collection. Supports GPU-accelerated processing through vLLM backend.

    Attributes:
        _command: CLI command to execute
        _timeout: Execution timeout in seconds

    Invariants:
        - _command is never empty
        - _timeout is positive

    Thread Safety:
        - Thread-safe: Uses temporary directories with unique names

    Example:
        >>> cli = SubprocessMineruCli(settings, timeout_seconds=300)
        >>> result = cli.run_batch(inputs)
        >>> print(f"Processed {len(result.outputs)} documents")

    """

    def __init__(
        self,
        settings: MineruSettings,
        *,
        timeout_seconds: int | None = None,
    ) -> None:
        """Initialize subprocess CLI with settings and timeout.

        Args:
            settings: MinerU configuration settings
            timeout_seconds: Optional timeout override

        Note:
            Uses settings timeout if none provided. Command is
            extracted from settings for execution.

        """
        super().__init__(settings)
        self._command = settings.cli_command
        self._timeout = timeout_seconds or settings.cli_timeout_seconds()

    def _ensure_command(self) -> None:
        """Ensure CLI command is available in PATH.

        Raises:
            MineruCliError: If CLI executable not found

        Note:
            Checks if the first component of the command
            is available in the system PATH.

        """
        executable = self._command.split()[0]
        if shutil.which(executable) is None:
            raise MineruCliError(f"MinerU CLI '{self._command}' not found in PATH")

    def describe(self) -> str:
        """Describe the subprocess CLI implementation.

        Returns:
            String description including command

        """
        return f"subprocess-cli(command={self._command})"

    def _build_command(self, input_dir: Path, output_dir: Path) -> list[str]:
        """Build CLI command with input/output directories.

        Args:
            input_dir: Directory containing input PDF files
            output_dir: Directory for output JSON files

        Returns:
            Complete CLI command as list of strings

        Note:
            Adds input/output paths, backend, and vLLM URL to the base command.
            MinerU v2.5.4+ uses: mineru -p <path> -o <output> -b <backend> -u <url>

        """
        command = self._command.split()
        command.extend(
            [
                "--path",
                str(input_dir),
                "--output",
                str(output_dir),
                "--backend",
                self._settings.workers.backend,
                "--url",
                str(self._settings.vllm_server.base_url),
            ]
        )
        return command

    def run_batch(
        self,
        batch: Iterable[MineruCliInput],
    ) -> MineruCliResult:
        """Execute batch processing via subprocess.

        Args:
            batch: Iterable of documents to process

        Returns:
            CLI result with outputs and execution details

        Raises:
            MineruCliError: If CLI execution fails or outputs missing

        Note:
            Creates temporary directories, writes PDF files,
            executes CLI command, and collects outputs.
            Handles timeouts and error conditions.

        """
        self._ensure_command()
        inputs = list(batch)
        if not inputs:
            return MineruCliResult(outputs=[], stdout="", stderr="", duration_seconds=0.0)

        # Process PDFs one at a time - MinerU CLI doesn't batch well
        outputs: list[MineruCliOutput] = []
        all_stdout: list[str] = []
        all_stderr: list[str] = []
        total_duration = 0.0

        for item in inputs:
            with tempfile.TemporaryDirectory(prefix=f"mineru-cli-{item.document_id}-") as workdir:
                input_dir = Path(workdir, "input")
                output_dir = Path(workdir, "output")
                input_dir.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Write single PDF
                pdf_path = input_dir / f"{item.document_id}.pdf"
                pdf_path.write_bytes(item.content)

                command = self._build_command(input_dir, output_dir)
                logger.bind(command=command, document_id=item.document_id).info("mineru.cli.invoke")
                start = time.monotonic()
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    env=dict(os.environ),
                )
                duration = time.monotonic() - start
                total_duration += duration

                if proc.returncode != 0:
                    logger.bind(
                        returncode=proc.returncode,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        document_id=item.document_id,
                    ).error("mineru.cli.failed")
                    raise MineruCliError(
                        f"MinerU CLI exited with code {proc.returncode} for {item.document_id}: {proc.stderr.strip()}"
                    )

                all_stdout.append(proc.stdout)
                all_stderr.append(proc.stderr)

                # MinerU v2.5.4 outputs to: {output_dir}/{document_id}/vlm/{document_id}_content_list.json
                # We use _content_list.json instead of _model.json because it includes:
                # - text_level: heading hierarchy (1=H1, 2=H2, etc.)
                # - Cleaner, flattened structure
                # - Better section clustering information
                output_path = output_dir / item.document_id / "vlm" / f"{item.document_id}_content_list.json"
                if not output_path.exists():
                    logger.bind(
                        output_path=str(output_path),
                        document_id=item.document_id,
                        output_dir_contents=list((output_dir / item.document_id).rglob("*")) if (output_dir / item.document_id).exists() else "dir_not_found"
                    ).error("mineru.cli.output_not_found")
                    raise MineruCliError(
                        f"MinerU CLI did not produce output for document '{item.document_id}' at {output_path}"
                    )
                # Read the JSON content before the temp directory is deleted
                json_content = output_path.read_text(encoding="utf-8")
                outputs.append(MineruCliOutput(
                    document_id=item.document_id,
                    path=output_path,
                    json_content=json_content
                ))

        return MineruCliResult(
            outputs=outputs,
            stdout="\n".join(all_stdout),
            stderr="\n".join(all_stderr),
            duration_seconds=total_duration,
        )


class SimulatedMineruCli(MineruCliBase):
    """Deterministic fallback CLI used in environments without GPUs.

    Simulated implementation that generates mock document structures
    for testing and development environments where GPU processing
    is not available. Creates realistic-looking output without
    requiring actual MinerU CLI installation.

    Attributes:
        None (inherits settings from base class)

    Invariants:
        - Input content must be UTF-8 decodable
        - Generated output follows MinerU format

    Thread Safety:
        - Thread-safe: Uses temporary files with unique names

    Example:
        >>> cli = SimulatedMineruCli(settings)
        >>> result = cli.run_batch(inputs)
        >>> print(f"Simulated processing: {len(result.outputs)} outputs")

    """

    def describe(self) -> str:
        """Describe the simulated CLI implementation.

        Returns:
            String description identifying simulation

        """
        return "simulated-cli"

    def run_batch(
        self,
        batch: Iterable[MineruCliInput],
    ) -> MineruCliResult:
        """Execute simulated batch processing.

        Args:
            batch: Iterable of documents to process

        Returns:
            CLI result with simulated outputs

        Raises:
            MineruCliError: If content is not UTF-8 decodable

        Note:
            Generates mock document structures by parsing
            input content as text and creating blocks,
            tables, and metadata. Outputs are written
            to temporary JSON files.

        """
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

            # Read JSON content for new MineruCliOutput format
            json_content = json.dumps(payload)

            outputs.append(MineruCliOutput(
                document_id=item.document_id,
                path=Path(path),
                json_content=json_content
            ))
            stdout_lines.append(f"simulated:{item.document_id}")

        duration = time.monotonic() - start
        return MineruCliResult(
            outputs=outputs,
            stdout="\n".join(stdout_lines),
            stderr="",
            duration_seconds=duration,
        )


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_cli(settings: MineruSettings) -> MineruCliBase:
    """Factory that resolves the appropriate CLI implementation.

    Attempts to create a real subprocess CLI first, falling back
    to simulated CLI if the MinerU command is not available.
    This allows the system to work in both production (with GPU)
    and development (without GPU) environments.

    Args:
        settings: MinerU configuration settings

    Returns:
        CLI implementation (subprocess or simulated)

    Note:
        Logs the selected implementation for debugging.
        Falls back to simulation if CLI command not found.

    Example:
        >>> cli = create_cli(settings)
        >>> print(f"Using CLI: {cli.describe()}")
        >>> result = cli.run_batch(inputs)

    """
    try:
        cli = SubprocessMineruCli(settings)
        cli._ensure_command()
        logger.info("mineru.cli.selected", impl=cli.describe())
        return cli
    except MineruCliError:
        logger.bind(reason="command-not-found").warning("mineru.cli.fallback")
        return SimulatedMineruCli(settings)


# ==============================================================================
# EXPORTS
# ==============================================================================

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
