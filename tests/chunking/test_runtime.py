from __future__ import annotations

from Medical_KG_rev.chunking import (
    ChunkerFactory,
    ChunkerSettings,
    ChunkingRuntime,
    MultiGranularityPipeline,
)
from Medical_KG_rev.chunking.models import ChunkerConfig
from Medical_KG_rev.chunking.chunkers.section import SectionAwareChunker
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section


def build_document() -> Document:
    section = Section(
        id="section-1",
        title="Intro",
        blocks=[
            Block(
                id="block-1",
                type=BlockType.PARAGRAPH,
                text="This is a sample paragraph for chunking tests.",
            )
        ],
    )
    return Document(id="doc", source="pmc", title="Doc", sections=[section])


def test_chunker_factory_caches_instances() -> None:
    factory = ChunkerFactory()
    config = ChunkerConfig(name="sliding_window", params={"target_tokens": 128})
    first = factory.create(config, allow_experimental=True)
    second = factory.create(config, allow_experimental=True)
    assert first.instance is second.instance
    factory.clear_cache()
    third = factory.create(config, allow_experimental=True)
    assert third.instance is not first.instance


def test_runtime_reuses_cached_chunkers() -> None:
    factory = ChunkerFactory()
    runtime = ChunkingRuntime(factory=factory)
    settings = [ChunkerSettings(strategy="sliding_window", params={"target_tokens": 128})]
    session_one = runtime.create_session(
        settings, allow_experimental=True, enable_multi_granularity=False
    )
    session_two = runtime.create_session(
        settings, allow_experimental=True, enable_multi_granularity=False
    )
    first_instance = session_one.plan.entries[0].instance
    second_instance = session_two.plan.entries[0].instance
    assert first_instance is second_instance


def test_pipeline_reuses_prepared_contexts() -> None:
    document = build_document()

    class CountingChunker(SectionAwareChunker):
        def __init__(self) -> None:
            super().__init__(preserve_tables=False)
            self.prepare_calls = 0
            self.with_context_calls = 0

        def prepare_contexts(self, document: Document, *, blocks=None):  # type: ignore[override]
            self.prepare_calls += 1
            return super().prepare_contexts(document, blocks=blocks)

        def chunk_with_contexts(self, *args, **kwargs):  # type: ignore[override]
            self.with_context_calls += 1
            return super().chunk_with_contexts(*args, **kwargs)

    chunker = CountingChunker()
    pipeline = MultiGranularityPipeline(chunkers=[(chunker, "section")])
    chunks = pipeline.chunk(document, tenant_id="tenant")
    assert chunks
    assert chunker.prepare_calls == 1
    assert chunker.with_context_calls == 1
