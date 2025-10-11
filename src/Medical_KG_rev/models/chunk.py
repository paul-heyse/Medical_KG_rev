"""Chunk model for document processing."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a chunk of text from a document."""

    id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="ID of the source document")
    content: str = Field(..., description="Text content of the chunk")
    start_offset: int = Field(..., description="Start offset in the original document")
    end_offset: int = Field(..., description="End offset in the original document")
    chunk_index: int = Field(..., description="Index of this chunk within the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
