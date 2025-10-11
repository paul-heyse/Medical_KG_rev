"""Output parser for MinerU processing results."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


@dataclass
class ParsedBlock:
    """Represents a parsed block from MinerU output."""

    text: str
    bbox: List[float]  # [x1, y1, x2, y2]
    page_number: int
    block_type: str  # e.g., "text", "table", "figure"
    confidence: float
    metadata: Dict[str, Any]


class ParsedDocument(BaseModel):
    """Represents a parsed document from MinerU."""

    document_id: str
    title: Optional[str] = None
    blocks: List[ParsedBlock] = []
    metadata: Dict[str, Any] = {}
    processing_time: float = 0.0

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class MineruOutputParser:
    """Parser for MinerU output files."""

    def __init__(self):
        """Initialize the parser."""
        pass

    def parse_output(self, output_data: Dict[str, Any]) -> ParsedDocument:
        """Parse MinerU output data."""
        document_id = output_data.get("document_id", "unknown")

        blocks = []
        for block_data in output_data.get("blocks", []):
            block = ParsedBlock(
                text=block_data.get("text", ""),
                bbox=block_data.get("bbox", [0, 0, 0, 0]),
                page_number=block_data.get("page_number", 1),
                block_type=block_data.get("block_type", "text"),
                confidence=block_data.get("confidence", 1.0),
                metadata=block_data.get("metadata", {})
            )
            blocks.append(block)

        return ParsedDocument(
            document_id=document_id,
            title=output_data.get("title"),
            blocks=blocks,
            metadata=output_data.get("metadata", {}),
            processing_time=output_data.get("processing_time", 0.0)
        )

    def extract_text(self, parsed_doc: ParsedDocument) -> str:
        """Extract plain text from parsed document."""
        return "\n".join(block.text for block in parsed_doc.blocks)

    def extract_blocks_by_type(self, parsed_doc: ParsedDocument, block_type: str) -> List[ParsedBlock]:
        """Extract blocks of a specific type."""
        return [block for block in parsed_doc.blocks if block.block_type == block_type]
