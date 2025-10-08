from __future__ import annotations

from types import SimpleNamespace

import pytest

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.orchestration.haystack import (
    HaystackChunker,
    HaystackEmbedder,
    HaystackIndexWriter,
    HaystackRetriever,
)
from Medical_KG_rev.orchestration.stages.contracts import (
    EmbeddingBatch,
    EmbeddingVector,
    StageContext,
)


class StubDocumentSplitter:
    def run(self, *, documents):
        return {"documents": list(documents)}


class StubEmbedder:
    def __init__(self) -> None:
        self.calls: list[list] = []

    def run(self, *, documents):
        self.calls.append(documents)
        enriched = []
        for doc in documents:
            enriched.append(
                SimpleNamespace(
                    content=doc.content,
                    meta=doc.meta,
                    embedding=[1.0, 2.0, 3.0],
                )
            )
        return {"documents": enriched}


class StubSparseExpander:
    def __init__(self, payload):
        self.payload = payload
        self.calls: list[list] = []

    def expand(self, documents):
        self.calls.append(list(documents))
        return self.payload


class StubWriter:
    def __init__(self) -> None:
        self.last_payload: list | None = None

    def run(self, *, documents):
        self.last_payload = list(documents)


class StubRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, *, query: str, filters: dict[str, object]):
        self.calls.append((query, filters))
        return {"documents": self.documents}


class StubRanker:
    def __init__(self):
        self.calls: list[list] = []

    def run(self, *, documents):
        self.calls.append(list(documents))
        return {"documents": documents}


@pytest.fixture()
def sample_document() -> Document:
    return Document(
        id="doc-1",
        source="clinical",
        title="Clinical Trial",
        sections=[
            Section(
                id="s1",
                title="Methods",
                blocks=[
                    Block(
                        id="b1",
                        type=BlockType.PARAGRAPH,
                        text="First paragraph of the document.",
                    ),
                    Block(
                        id="b2",
                        type=BlockType.PARAGRAPH,
                        text="Second paragraph provides more detail.",
                    ),
                ],
            )
        ],
    )


def test_chunker_converts_document(sample_document):
    chunker = HaystackChunker(splitter=StubDocumentSplitter())
    ctx = StageContext(tenant_id="tenant-a")

    chunks = chunker.execute(ctx, sample_document)

    assert len(chunks) == 2
    assert chunks[0].chunk_id.endswith(":0000")
    assert chunks[0].meta["block_ids"] == ["b1"]
    assert chunks[1].start_char > chunks[0].end_char - len(chunks[0].body)


def test_embedder_generates_embedding_batch(sample_document):
    chunker = HaystackChunker(splitter=StubDocumentSplitter())
    ctx = StageContext(tenant_id="tenant-b")
    chunks = chunker.execute(ctx, sample_document)
    sparse = StubSparseExpander([
        {"token": 0.5},
        {"token": 0.75},
    ])
    embedder = HaystackEmbedder(embedder=StubEmbedder(), sparse_expander=sparse, require_gpu=False)

    batch = embedder.execute(ctx, chunks)

    assert isinstance(batch, EmbeddingBatch)
    assert batch.model == "qwen-3"
    assert len(batch.vectors) == len(chunks)
    assert batch.vectors[0].metadata["sparse_vector"] == {"token": 0.5}
    assert sparse.calls  # ensure expander invoked


def test_index_writer_delegates_to_writers():
    dense = StubWriter()
    sparse = StubWriter()
    writer = HaystackIndexWriter(dense_writer=dense, sparse_writer=sparse)
    ctx = StageContext(tenant_id="tenant-c")
    batch = EmbeddingBatch(
        vectors=(
            EmbeddingVector(
                id="chunk-1",
                values=(0.1, 0.2, 0.3),
                metadata={"chunk_id": "chunk-1", "text": "body", "doc_id": "doc-1"},
            ),
        ),
        model="qwen-3",
        tenant_id="tenant-c",
    )

    receipt = writer.execute(ctx, batch)

    assert receipt.chunks_indexed == 1
    assert dense.last_payload and sparse.last_payload
    assert dense.last_payload[0].embedding == [0.1, 0.2, 0.3]


def test_retriever_merges_results():
    lexical_doc = SimpleNamespace(id="lex-1", score=0.9, content="Lexical", meta={"source": "lex"})
    dense_doc = SimpleNamespace(id="dense-1", score=0.8, content="Dense", meta={"source": "dense"})
    bm25 = StubRetriever([lexical_doc])
    dense = StubRetriever([dense_doc])
    ranker = StubRanker()
    retriever = HaystackRetriever(bm25_retriever=bm25, dense_retriever=dense, fusion_ranker=ranker)

    results = retriever.retrieve("gene therapy", filters={"tenant_id": "tenant-d"})

    assert {result["id"] for result in results} == {"lex-1", "dense-1"}
    assert bm25.calls and dense.calls
    assert ranker.calls[0][0] is lexical_doc
