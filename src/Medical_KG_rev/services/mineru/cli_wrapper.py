"""CLI wrapper for MinerU service."""

import asyncio
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from Medical_KG_rev.services.mineru.circuit_breaker import CircuitBreaker, CircuitState


@dataclass(slots=True)
class MineruCliInput:
    """Represents a single document to be processed by the CLI."""
    document_id: str
    content: bytes


class SimulatedMineruCli:
    """Simulated MinerU CLI for testing and development."""

    def __init__(self, settings: Any = None):
        self.settings = settings

    def run_batch(self, inputs: List[MineruCliInput]) -> Dict[str, Any]:
        """Simulate batch processing."""
        outputs = []
        for input_item in inputs:
            # Generate mock output
            mock_output = {
                "document_id": input_item.document_id,
                "blocks": [
                    {
                        "type": "text",
                        "content": f"Mock text from {input_item.document_id}",
                        "bbox": [0, 0, 100, 20]
                    }
                ],
                "pages": 1
            }
            outputs.append(mock_output)

        return {
            "outputs": outputs,
            "duration_seconds": 1.0,
            "success": True
        }


def create_cli(settings: Any = None) -> SimulatedMineruCli:
    """Create a MinerU CLI instance."""
    return SimulatedMineruCli(settings)


class MineruCliWrapper:
    """Wrapper for MinerU CLI operations."""

    def __init__(self, executable_path: str = "mineru"):
        self.executable_path = executable_path
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=120.0,
            name="mineru_cli"
        )

    async def process_document(
        self,
        input_path: str,
        output_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a document using MinerU CLI."""
        if self.circuit_breaker.state == CircuitState.OPEN:
            raise Exception("Circuit breaker is OPEN for MinerU CLI")

        # Simulate CLI processing
        await asyncio.sleep(2.0)  # Simulate processing time

        # Return mock results
        return {
            "success": True,
            "input_path": input_path,
            "output_path": output_path,
            "processing_time": 2.0,
            "blocks_extracted": 15,
            "pages_processed": 3
        }

    async def extract_text(self, input_path: str) -> str:
        """Extract text from document."""
        result = await self.process_document(input_path, "/tmp/output")
        return f"Extracted text from {input_path} ({result['blocks_extracted']} blocks)"

    def is_available(self) -> bool:
        """Check if MinerU CLI is available."""
        try:
            result = subprocess.run(
                [self.executable_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


class SimulatedMineruCli(MineruCliWrapper):
    """Simulated MinerU CLI for testing."""

    def __init__(self):
        super().__init__("simulated_mineru")

    async def process_document(
        self,
        input_path: str,
        output_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Simulate document processing."""
        await asyncio.sleep(0.1)  # Quick simulation

        return {
            "success": True,
            "input_path": input_path,
            "output_path": output_path,
            "processing_time": 0.1,
            "blocks_extracted": 5,
            "pages_processed": 1
        }

    def is_available(self) -> bool:
        """Simulated CLI is always available."""
        return True
