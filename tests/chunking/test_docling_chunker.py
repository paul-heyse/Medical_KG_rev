"""Tests for Docling-based chunkers without torch dependencies."""

import pytest

from Medical_KG_rev.chunking.chunkers.docling import DoclingChunker, DoclingVLMChunker
from Medical_KG_rev.models.ir import Block, BlockType
from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMResult


class TestDoclingChunker:
    """Test DoclingChunker without torch dependencies."""

    def test_init(self):
        """Test chunker initialization."""
        chunker = DoclingChunker()
        assert chunker.name == "docling"
        assert chunker.version == "v1"
        assert chunker.segment_type == "docling"

    def test_explain(self):
        """Test chunker explanation."""
        chunker = DoclingChunker(
            min_chunk_size=100,
            max_chunk_size=1000,
            overlap_ratio=0.2,
        )
        explanation = chunker.explain()
        assert explanation["min_chunk_size"] == 100
        assert explanation["max_chunk_size"] == 1000
        assert explanation["overlap_ratio"] == 0.2

    def test_segment_contexts_empty(self):
        """Test segmenting empty contexts."""
        chunker = DoclingChunker()
        segments = list(chunker.segment_contexts([]))
        assert len(segments) == 0

    def test_segment_contexts_single(self):
        """Test segmenting single context."""
        from Medical_KG_rev.chunking.provenance import BlockContext

        chunker = DoclingChunker(min_chunk_size=50, max_chunk_size=200)

        # Create mock context
        block = Block(
            id="test-block",
            type=BlockType.PARAGRAPH,
            text="This is a test paragraph with some content.",
        )
        context = BlockContext(
            block=block,
            text=block.text,
            token_count=10,  # Small token count
        )

        segments = list(chunker.segment_contexts([context]))
        assert len(segments) == 0  # Below min_chunk_size

    def test_segment_contexts_multiple(self):
        """Test segmenting multiple contexts."""
        from Medical_KG_rev.chunking.provenance import BlockContext

        chunker = DoclingChunker(min_chunk_size=50, max_chunk_size=200)

        # Create mock contexts
        contexts = []
        for i in range(5):
            block = Block(
                id=f"test-block-{i}",
                type=BlockType.PARAGRAPH,
                text=f"This is test paragraph {i} with some content that should exceed the minimum chunk size.",
            )
            context = BlockContext(
                block=block,
                text=block.text,
                token_count=20,  # Each context has 20 tokens
            )
            contexts.append(context)

        segments = list(chunker.segment_contexts(contexts))
        assert len(segments) >= 1
        assert all(len(segment.contexts) > 0 for segment in segments)

    def test_overlap_contexts(self):
        """Test overlap context generation."""
        from Medical_KG_rev.chunking.provenance import BlockContext

        chunker = DoclingChunker(max_chunk_size=100, overlap_ratio=0.3)

        # Create mock contexts
        contexts = []
        for i in range(10):
            block = Block(
                id=f"test-block-{i}",
                type=BlockType.PARAGRAPH,
                text=f"Test paragraph {i}",
            )
            context = BlockContext(
                block=block,
                text=block.text,
                token_count=10,
            )
            contexts.append(context)

        overlap = chunker._get_overlap_contexts(contexts)
        assert len(overlap) <= len(contexts)
        # Overlap should be from the end of the original contexts
        if overlap:
            assert overlap[-1] == contexts[-1]


class TestDoclingVLMChunker:
    """Test DoclingVLMChunker without torch dependencies."""

    def test_init(self):
        """Test chunker initialization."""
        chunker = DoclingVLMChunker()
        assert chunker.name == "docling_vlm"
        assert chunker.version == "v1"
        assert chunker.segment_type == "docling_vlm"

    def test_explain(self):
        """Test chunker explanation."""
        chunker = DoclingVLMChunker(
            preserve_structure=True,
            include_tables=False,
            include_figures=True,
        )
        explanation = chunker.explain()
        assert explanation["preserve_structure"] is True
        assert explanation["include_tables"] is False
        assert explanation["include_figures"] is True

    def test_format_table(self):
        """Test table formatting."""
        chunker = DoclingVLMChunker()

        table_data = {
            "headers": ["Name", "Age", "City"],
            "cells": [
                {"row": 0, "column": 0, "text": "Alice"},
                {"row": 0, "column": 1, "text": "25"},
                {"row": 0, "column": 2, "text": "NYC"},
                {"row": 1, "column": 0, "text": "Bob"},
                {"row": 1, "column": 1, "text": "30"},
                {"row": 1, "column": 2, "text": "LA"},
            ],
        }

        formatted = chunker._format_table(table_data)
        assert "Alice" in formatted
        assert "Bob" in formatted
        assert "|" in formatted  # Markdown table format

    def test_format_table_no_headers(self):
        """Test table formatting without headers."""
        chunker = DoclingVLMChunker()

        table_data = {
            "cells": [
                {"row": 0, "column": 0, "text": "Alice"},
                {"row": 0, "column": 1, "text": "25"},
                {"row": 1, "column": 0, "text": "Bob"},
                {"row": 1, "column": 1, "text": "30"},
            ],
        }

        formatted = chunker._format_table(table_data)
        assert "Alice" in formatted
        assert "Bob" in formatted

    def test_chunk_from_docling_result(self):
        """Test chunking from DoclingVLMResult."""
        chunker = DoclingVLMChunker()

        # Create mock DoclingVLMResult
        result = DoclingVLMResult(
            document_id="test-doc",
            text="This is the main text content.\n\nIt has multiple paragraphs.",
            tables=[
                {
                    "headers": ["Name", "Value"],
                    "cells": [
                        {"row": 0, "column": 0, "text": "Test"},
                        {"row": 0, "column": 1, "text": "Value"},
                    ],
                }
            ],
            figures=[
                {
                    "caption": "Test figure caption",
                    "type": "image",
                }
            ],
            metadata={"title": "Test Document"},
        )

        chunks = chunker.chunk_from_docling_result(result, tenant_id="test-tenant")
        assert len(chunks) > 0

        # Check that chunks contain expected content
        chunk_texts = [chunk.text for chunk in chunks]
        assert any("main text content" in text for text in chunk_texts)
        assert any("Test" in text for text in chunk_texts)  # Table content
        assert any("Test figure caption" in text for text in chunk_texts)

    def test_chunk_from_docling_result_no_tables(self):
        """Test chunking from DoclingVLMResult without tables."""
        chunker = DoclingVLMChunker(include_tables=False)

        result = DoclingVLMResult(
            document_id="test-doc",
            text="Simple text content.",
            tables=[
                {
                    "headers": ["Name"],
                    "cells": [{"row": 0, "column": 0, "text": "Test"}],
                }
            ],
            figures=[],
            metadata={},
        )

        chunks = chunker.chunk_from_docling_result(result, tenant_id="test-tenant")
        chunk_texts = [chunk.text for chunk in chunks]

        # Should not contain table content
        assert not any("Test" in text for text in chunk_texts)

    def test_segment_contexts_empty(self):
        """Test segmenting empty contexts."""
        chunker = DoclingVLMChunker()
        segments = list(chunker.segment_contexts([]))
        assert len(segments) == 0

    def test_segment_contexts_by_type(self):
        """Test segmenting contexts by type."""
        from Medical_KG_rev.chunking.provenance import BlockContext

        chunker = DoclingVLMChunker()

        # Create contexts with different types
        contexts = []
        for i, source in enumerate(["docling_vlm", "docling_vlm", "other", "other"]):
            block = Block(
                id=f"test-block-{i}",
                type=BlockType.PARAGRAPH,
                text=f"Test content {i}",
            )
            context = BlockContext(
                block=block,
                text=block.text,
                token_count=10,
            )
            context.metadata = {"source": source}
            contexts.append(context)

        segments = list(chunker.segment_contexts(contexts))
        assert len(segments) == 2  # Two different types
        assert len(segments[0].contexts) == 2  # First two docling_vlm contexts
        assert len(segments[1].contexts) == 2  # Last two other contexts


class TestTorchFreeCompatibility:
    """Test that chunkers work without torch dependencies."""

    def test_no_torch_imports(self):
        """Test that chunkers don't import torch."""
        import sys

        # Clear any existing torch modules
        torch_modules = [name for name in sys.modules.keys() if name.startswith("torch")]
        for module in torch_modules:
            del sys.modules[module]

        # Import chunkers
        from Medical_KG_rev.chunking.chunkers.docling import DoclingChunker, DoclingVLMChunker

        # Verify torch is not imported
        assert "torch" not in sys.modules

        # Test that chunkers work
        docling_chunker = DoclingChunker()
        docling_vlm_chunker = DoclingVLMChunker()

        assert docling_chunker.name == "docling"
        assert docling_vlm_chunker.name == "docling_vlm"

    def test_gpu_semantic_checks_deprecated(self):
        """Test that GPU semantic checks are deprecated."""
        from Medical_KG_rev.chunking.chunkers.semantic import SemanticSplitterChunker
        from Medical_KG_rev.chunking.exceptions import ChunkerConfigurationError

        # Should raise error when gpu_semantic_checks=True
        with pytest.raises(
            ChunkerConfigurationError, match="GPU semantic checks are no longer supported"
        ):
            SemanticSplitterChunker(gpu_semantic_checks=True)

        # Should work when gpu_semantic_checks=False (default)
        chunker = SemanticSplitterChunker(gpu_semantic_checks=False)
        assert chunker.name == "semantic_splitter"
