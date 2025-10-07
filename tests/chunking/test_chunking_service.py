from __future__ import annotations

from pathlib import Path

import pytest

from Medical_KG_rev.chunking import ChunkingService
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.services.retrieval.chunking import ChunkingOptions as RetrievalOptions
from Medical_KG_rev.services.retrieval.chunking import ChunkingService as RetrievalChunkingService


@pytest.fixture()
def simple_config(tmp_path: Path) -> Path:
    config = tmp_path / "chunking.yaml"
    config.write_text(
        """
        default_profile: default
        profiles:
          default:
            enable_multi_granularity: false
            primary:
              strategy: section_aware
              granularity: section
              params:
                target_tokens: 200
        """
    )
    return config


def build_document() -> Document:
    section1 = Section(
        id="sec-1",
        title="Introduction",
        blocks=[
            Block(
                id="b-1",
                type=BlockType.PARAGRAPH,
                text="Introduction text about the study design and rationale.",
            ),
            Block(
                id="b-2",
                type=BlockType.PARAGRAPH,
                text="Further introduction details with eligibility highlights.",
            ),
        ],
    )
    section2 = Section(
        id="sec-2",
        title="Methods",
        blocks=[
            Block(
                id="b-3",
                type=BlockType.PARAGRAPH,
                text="Methods section describing interventions and dosages.",
            ),
        ],
    )
    return Document(
        id="doc-1",
        source="pmc",
        title="Sample Document",
        sections=[section1, section2],
    )


def test_section_chunker(simple_config: Path) -> None:
    document = build_document()
    service = ChunkingService(config_path=simple_config)
    chunks = service.chunk_document(document, tenant_id="tenant-1", source=document.source)
    assert chunks, "expected section chunker to return chunks"
    assert all(chunk.tenant_id == "tenant-1" for chunk in chunks)
    assert {chunk.granularity for chunk in chunks} == {"section"}
    assert chunks[0].chunk_id.startswith(f"{document.id}:section_aware:section:")
    assert chunks[0].meta["token_count"] > 0


def test_retrieval_service_alias_mapping(simple_config: Path) -> None:
    service = RetrievalChunkingService(config_path=simple_config)
    options = RetrievalOptions(strategy="section", max_tokens=256)
    chunks = service.chunk("tenant-1", "doc-legacy", "Heading\n\nBody text.", options)
    assert chunks, "legacy retrieval chunking should return chunks"
    assert chunks[0].granularity == "section"
    assert chunks[0].chunker == "section_aware"
