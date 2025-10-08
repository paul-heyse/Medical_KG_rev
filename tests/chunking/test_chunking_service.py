from __future__ import annotations

from pathlib import Path

import pytest

from Medical_KG_rev.chunking import ChunkingService
from Medical_KG_rev.chunking.exceptions import ChunkingUnavailableError, InvalidDocumentError
from Medical_KG_rev.chunking.registry import ChunkerRegistry
from Medical_KG_rev.chunking.factory import ChunkerFactory
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section


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
                min_tokens: 10
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
def test_chunk_page_number_propagated(simple_config: Path) -> None:
    base = build_document().sections[0]
    blocks = [
        block.model_copy(update={"metadata": {"page_number": 3}})
        for block in base.blocks
    ]
    document = Document(
        id="doc-page",
        source="pmc",
        title="Page Test",
        sections=[base.model_copy(update={"blocks": blocks})],
    )
    service = ChunkingService(config_path=simple_config)
    chunks = service.chunk_document(document, tenant_id="tenant", source=document.source)
    assert chunks
    assert any(chunk.page_no == 3 for chunk in chunks)
    assert any(3 in chunk.meta.get("page_numbers", []) for chunk in chunks)


def test_chunk_text_requires_non_empty(simple_config: Path) -> None:
    service = ChunkingService(config_path=simple_config)
    with pytest.raises(InvalidDocumentError):
        service.chunk_text("tenant", "doc", "  ")


def test_chunking_circuit_breaker_opens(tmp_path: Path) -> None:
    failing_config = tmp_path / "failing.yaml"
    failing_config.write_text(
        """
        default_profile: default
        profiles:
          default:
            enable_multi_granularity: false
            primary:
              strategy: failing
              granularity: section
        """
    )

    class FailingChunker:
        name = "failing"
        version = "v1"

        def chunk(self, document, *, tenant_id, granularity=None, blocks=None):  # noqa: ANN001
            raise MemoryError("simulated OOM")

        def explain(self):
            return {"reason": "test"}

    registry = ChunkerRegistry()
    registry.register("failing", FailingChunker)
    factory = ChunkerFactory(registry)
    service = ChunkingService(
        config_path=failing_config,
        registry_factory=factory,
        failure_threshold=3,
        base_recovery_seconds=1.0,
        max_recovery_seconds=1.0,
    )
    document = build_document()
    for _ in range(3):
        with pytest.raises(MemoryError):
            service.chunk_document(document, tenant_id="tenant", source=document.source)
    with pytest.raises(ChunkingUnavailableError):
        service.chunk_document(document, tenant_id="tenant", source=document.source)


def test_chunking_service_session_cache(simple_config: Path) -> None:
    document = build_document()
    service = ChunkingService(config_path=simple_config)
    assert not service._session_cache
    service.chunk_document(document, tenant_id="tenant", source=document.source)
    assert len(service._session_cache) == 1
    service.chunk_document(document, tenant_id="tenant", source=document.source)
    assert len(service._session_cache) == 1
    service.clear_session_cache()
    assert not service._session_cache
