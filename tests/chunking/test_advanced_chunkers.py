from time import perf_counter

import numpy as np
import pytest

from Medical_KG_rev.chunking.chunkers import (
    ClinicalRoleChunker,
    DiscourseSegmenterChunker,
    GraphRAGChunker,
    GrobidSectionChunker,
    LayoutAwareChunker,
    LLMChapteringChunker,
    SectionAwareChunker,
    SemanticSplitterChunker,
    SlidingWindowChunker,
    TableChunker,
)
from Medical_KG_rev.chunking.pipeline import MultiGranularityPipeline
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section


class StubEncoder:
    def encode(self, sentences, convert_to_numpy=True):
        matrix = np.ones((len(sentences), 8))
        return matrix


def _document() -> Document:
    blocks = [
        Block(
            id="b-1",
            type=BlockType.HEADER,
            text="Introduction to the study",
        ),
        Block(
            id="b-2",
            type=BlockType.PARAGRAPH,
            text="However the treatment group demonstrated improved outcomes compared to control.",
        ),
        Block(
            id="b-3",
            type=BlockType.PARAGRAPH,
            text="Therefore clinicians considered dose adjustments based on safety signals.",
        ),
    ]
    section = Section(id="s-1", title="Study", blocks=blocks)
    return Document(id="doc-1", source="pmc", title="Sample", sections=[section], metadata={"tei_xml": "<TEI><text><body><div type='introduction'><head>Introduction</head></div></body></text></TEI>"})


def test_llm_chaptering_chunker_produces_chunks() -> None:
    chunker = LLMChapteringChunker()
    document = _document()
    chunks = chunker.chunk(document, tenant_id="tenant", granularity="section")
    assert chunks
    assert all(chunk.granularity == "section" for chunk in chunks)


def test_discourse_segmenter_identifies_boundaries() -> None:
    chunker = DiscourseSegmenterChunker()
    chunks = chunker.chunk(_document(), tenant_id="tenant", granularity="paragraph")
    assert len(chunks) >= 2


def test_graph_rag_chunker_clusters_contexts() -> None:
    chunker = GraphRAGChunker()
    chunks = chunker.chunk(_document(), tenant_id="tenant", granularity="section")
    assert chunks
    assert chunks[0].meta["segment_type"] == "graph"


def test_layout_chunker_groups_regions() -> None:
    document = _document()
    chunker = LayoutAwareChunker()
    chunks = chunker.chunk(document, tenant_id="tenant", granularity="section")
    assert chunks
    assert all(chunk.meta["segment_type"] == "layout" for chunk in chunks)


def test_grobid_chunker_requires_tei_metadata() -> None:
    document = _document().model_copy(update={"metadata": {}})
    chunker = GrobidSectionChunker()
    with pytest.raises(Exception):
        chunker.chunk(document, tenant_id="tenant", granularity="section")


def test_multi_granularity_pipeline_runs_parallel() -> None:
    document = _document()
    semantic = SemanticSplitterChunker(encoder=StubEncoder())
    pipeline = MultiGranularityPipeline(
        chunkers=[
            (SectionAwareChunker(), "section"),
            (semantic, "paragraph"),
            (SlidingWindowChunker(target_tokens=60, overlap_ratio=0.2), "window"),
        ]
    )
    chunks = pipeline.chunk(document, tenant_id="tenant")
    assert chunks
    granularities = {chunk.granularity for chunk in chunks}
    assert {"section", "paragraph", "window"}.issubset(granularities)


def test_table_chunker_preserves_atomicity() -> None:
    table_block = Block(
        id="b-table",
        type=BlockType.TABLE,
        text="Heading | Value",
        metadata={"is_table": True},
    )
    doc = Document(
        id="doc-table",
        source="spl",
        title="Tables",
        sections=[Section(id="s", title="Tabular", blocks=[table_block])],
    )
    chunker = TableChunker()
    chunks = chunker.chunk(doc, tenant_id="tenant", granularity="table")
    assert chunks
    assert all(chunk.granularity == "table" for chunk in chunks)


def test_clinical_role_chunker_adds_metadata() -> None:
    document = _document()
    chunker = ClinicalRoleChunker()
    chunks = chunker.chunk(document, tenant_id="tenant", granularity="paragraph")
    assert chunks
    assert any(chunk.meta.get("facet_type") for chunk in chunks)


def test_provenance_metadata_contains_block_ids() -> None:
    document = _document()
    chunker = SectionAwareChunker()
    chunks = chunker.chunk(document, tenant_id="tenant", granularity="section")
    assert chunks
    assert "block_ids" in chunks[0].meta


def test_chunking_performance_under_threshold() -> None:
    document = _document()
    chunker = SemanticSplitterChunker(encoder=StubEncoder())
    start = perf_counter()
    chunker.chunk(document, tenant_id="tenant", granularity="paragraph")
    duration = perf_counter() - start
    assert duration < 0.5
