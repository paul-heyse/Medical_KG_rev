from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import structlog

from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.orchestration.stages.contracts import (
    ChunkStage,
    EmbedStage,
    EmbeddingBatch,
    EmbeddingVector,
    IndexReceipt,
    IndexStage,
    PipelineState,
    StageContext,
)

logger = structlog.get_logger(__name__)

try:  # pragma: no cover - exercised in integration environments
    from haystack import Document as HaystackDocument  # type: ignore
except ImportError:  # pragma: no cover - fallback for local tooling
    try:
        from haystack.dataclasses import Document as HaystackDocument  # type: ignore
    except ImportError:  # pragma: no cover - unit test fallback

        @dataclass
        class HaystackDocument:  # type: ignore[override]
            id: str | None = None
            content: str = ""
            meta: dict[str, Any] = field(default_factory=dict)
            embedding: Sequence[float] | None = None
            sparse_embedding: dict[str, float] | None = None


try:  # pragma: no cover - optional dependency initialisation
    from haystack.components.preprocessors import DocumentSplitter  # type: ignore
except ImportError:  # pragma: no cover - fallback for tests
    DocumentSplitter = Any  # type: ignore


@dataclass(slots=True)
class _StubResultDocument:
    id: str
    content: str
    score: float
    meta: dict[str, Any]


class _StubRetrieverComponent:
    def __init__(self, kind: str, top_k: int) -> None:
        self._kind = kind
        self._top_k = top_k

    def run(
        self, *, query: str, filters: dict[str, Any] | None = None
    ) -> dict[str, list[_StubResultDocument]]:
        payload_filters = dict(filters or {})
        key = tuple(sorted(payload_filters.items()))
        doc_id = f"{self._kind}-{abs(hash((query, key))) % 10000:04d}"
        document = _StubResultDocument(
            id=doc_id,
            content=f"{self._kind.upper()} result for '{query}'",
            score=0.5,
            meta={"source": f"{self._kind}-stub", "filters": payload_filters},
        )
        return {"documents": [document]}


class _StubRanker:
    def __init__(self, top_k: int) -> None:
        self._top_k = top_k

    def run(self, *, documents: Sequence[Any] | None = None) -> dict[str, list[Any]]:
        docs = list(documents or [])
        docs.sort(key=lambda doc: getattr(doc, "score", 0.0), reverse=True)
        return {"documents": docs[: self._top_k]}


def _ensure_gpu_available(require_gpu: bool) -> None:
    if not require_gpu:
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("GPU support requires the 'torch' package") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled GPU is required for this component")


def _normalise_vector(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


class HaystackChunker(ChunkStage):
    """Convert IR documents into Haystack documents and split them."""

    def __init__(
        self,
        splitter: DocumentSplitter | None = None,
        *,
        chunker_name: str = "haystack.semantic",
        chunker_version: str = "1.0.0",
        granularity: str = "paragraph",
    ) -> None:
        if splitter is None:
            try:  # pragma: no cover - requires haystack runtime
                splitter = DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2)
            except Exception:  # pragma: no cover - fallback when haystack is unavailable
                splitter = SimpleDocumentSplitter(sentence_length=3)
        self._splitter = splitter
        self._chunker_name = chunker_name
        self._chunker_version = chunker_version
        self._granularity = granularity

    def execute(self, ctx: StageContext, state: PipelineState) -> list[Chunk]:
        document = state.require_document()
        haystack_documents: list[HaystackDocument] = []
        full_text: list[str] = []
        offsets: list[tuple[int, dict[str, Any]]] = []
        running_offset = 0
        for section in document.sections:
            title_path = tuple(filter(None, (document.title, section.title)))
            for block in section.blocks:
                text = (block.text or "").strip()
                if not text:
                    continue
                meta = {
                    "block_id": block.id,
                    "section_id": section.id,
                    "title_path": title_path,
                    "document_id": document.id,
                }
                haystack_documents.append(HaystackDocument(content=text, meta=dict(meta)))
                full_text.append(text)
                offsets.append((running_offset, meta))
                running_offset += len(text) + 1  # include newline when reconstructing
        concatenated = "\n".join(full_text)
        if not haystack_documents:
            return []

        result = self._splitter.run(documents=haystack_documents)
        chunks: list[Chunk] = []
        produced = result.get("documents") if isinstance(result, dict) else result
        produced = produced or []
        search_start = 0
        for index, chunk_doc in enumerate(produced):
            content = getattr(chunk_doc, "content", "") or ""
            if not content:
                continue
            start_idx = concatenated.find(content, search_start)
            if start_idx < 0:
                start_idx = concatenated.find(content)
            if start_idx < 0:
                start_idx = 0
            end_idx = start_idx + len(content)
            search_start = end_idx
            block_ids: set[str] = set()
            section_id: str | None = None
            title_path: tuple[str, ...] = ()
            for offset, meta in offsets:
                if offset > end_idx:
                    break
                block_ids.add(meta["block_id"])
                section_id = meta.get("section_id")
                title_path = tuple(meta.get("title_path", ()))
            chunk_meta = dict(getattr(chunk_doc, "meta", {}))
            chunk_meta.setdefault("block_ids", sorted(block_ids))
            chunk_meta.setdefault("coherence_score", getattr(chunk_doc, "score", None))
            chunk_meta.setdefault("section_id", section_id)
            chunk_meta.setdefault("title_path", title_path)
            chunk_id = f"{document.id}:{self._chunker_name}:{index:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.id,
                    tenant_id=ctx.tenant_id,
                    body=content,
                    title_path=title_path,
                    section=section_id,
                    start_char=start_idx,
                    end_char=end_idx,
                    granularity=self._granularity,  # type: ignore[arg-type]
                    chunker=self._chunker_name,
                    chunker_version=self._chunker_version,
                    meta=chunk_meta,
                )
            )
        return chunks


class HaystackSparseExpander:
    """Generate SPLADE-style sparse vectors for retrieval expansion."""

    def __init__(
        self,
        component: Any | None = None,
        *,
        model: str = "naver/splade-cocondenser-ensembledistil",
        require_gpu: bool = True,
    ) -> None:
        _ensure_gpu_available(require_gpu)
        if component is None:  # pragma: no cover - requires haystack runtime
            from haystack.components.embedders import SparseEmbeddingEncoder  # type: ignore

            component = SparseEmbeddingEncoder(model=model, device="cuda")
        self._component = component

    def expand(self, documents: Sequence[HaystackDocument]) -> Sequence[dict[str, float]]:
        result = self._component.run(documents=list(documents))
        vectors = result.get("documents") if isinstance(result, dict) else result
        if not vectors:
            return []
        expansions: list[dict[str, float]] = []
        for doc in vectors:
            sparse = getattr(doc, "sparse_embedding", None) or {}
            expansions.append({str(term): float(weight) for term, weight in sparse.items()})
        return expansions


class HaystackEmbedder(EmbedStage):
    """Wrap the Haystack OpenAI embedder for dense vector generation."""

    def __init__(
        self,
        embedder: Any | None = None,
        *,
        model: str = "qwen-3",
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 16,
        sparse_expander: HaystackSparseExpander | None = None,
        require_gpu: bool = True,
    ) -> None:
        _ensure_gpu_available(require_gpu)
        if embedder is None:  # pragma: no cover - requires haystack runtime
            from haystack.components.embedders import OpenAIDocumentEmbedder  # type: ignore

            embedder = OpenAIDocumentEmbedder(
                model=model,
                api_key=api_key or "dummy",
                base_url=base_url or "http://localhost:8000/v1",
                max_retries=3,
                timeout=60,
                max_batch_size=batch_size,
            )
        self._embedder = embedder
        self._model = model
        self._sparse_expander = sparse_expander

    def execute(self, ctx: StageContext, state: PipelineState) -> EmbeddingBatch:
        chunks = list(state.require_chunks())
        documents = [
            HaystackDocument(
                id=chunk.chunk_id,
                content=chunk.body,
                meta={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "tenant_id": chunk.tenant_id,
                    "granularity": chunk.granularity,
                    "section": chunk.section,
                    "title_path": chunk.title_path,
                    **chunk.meta,
                },
            )
            for chunk in chunks
        ]
        if not documents:
            return EmbeddingBatch(vectors=(), model=self._model, tenant_id=ctx.tenant_id)

        result = self._embedder.run(documents=documents)
        embedded = result.get("documents") if isinstance(result, dict) else result
        embedded = embedded or []

        sparse_vectors: Sequence[dict[str, float]] | None = None
        if self._sparse_expander is not None:
            sparse_vectors = self._sparse_expander.expand(documents)

        vectors: list[EmbeddingVector] = []
        for index, doc in enumerate(embedded):
            embedding = getattr(doc, "embedding", None)
            if embedding is None:
                continue
            chunk_id = doc.meta.get("chunk_id") if isinstance(doc.meta, dict) else None
            chunk_id = chunk_id or getattr(doc, "id", None) or f"chunk-{index}"
            metadata = dict(doc.meta or {})
            if sparse_vectors and index < len(sparse_vectors):
                metadata["sparse_vector"] = sparse_vectors[index]
            vectors.append(
                EmbeddingVector(
                    id=str(chunk_id),
                    values=_normalise_vector(embedding),
                    metadata=metadata,
                )
            )
        return EmbeddingBatch(vectors=tuple(vectors), model=self._model, tenant_id=ctx.tenant_id)


class HaystackIndexWriter(IndexStage):
    """Write embedding payloads to OpenSearch and FAISS in tandem."""

    def __init__(
        self,
        *,
        dense_writer: Any | None = None,
        sparse_writer: Any | None = None,
        opensearch_index: str = "chunks",
        faiss_index: str = "chunks",
    ) -> None:
        if dense_writer is None:  # pragma: no cover - requires haystack runtime
            from haystack_integrations.components.vectorstores.faiss import (  # type: ignore
                FAISSDocumentWriter,
            )

            dense_writer = FAISSDocumentWriter(index_name=faiss_index)
        if sparse_writer is None:  # pragma: no cover - requires haystack runtime
            from haystack_integrations.components.document_stores.opensearch import (  # type: ignore
                OpenSearchDocumentWriter,
            )

            sparse_writer = OpenSearchDocumentWriter(index=opensearch_index)
        self._dense_writer = dense_writer
        self._sparse_writer = sparse_writer
        self._opensearch_index = opensearch_index
        self._faiss_index = faiss_index

    def execute(self, ctx: StageContext, state: PipelineState) -> IndexReceipt:
        batch = state.require_embedding_batch()
        if not batch.vectors:
            return IndexReceipt(chunks_indexed=0, opensearch_ok=True, faiss_ok=True)

        documents: list[HaystackDocument] = []
        for vector in batch.vectors:
            meta = dict(vector.metadata)
            chunk_id = meta.get("chunk_id") or vector.id
            documents.append(
                HaystackDocument(
                    id=str(chunk_id),
                    content=meta.get("text") or meta.get("body") or "",
                    meta=meta,
                    embedding=list(vector.values),
                )
            )

        opensearch_ok = True
        if self._sparse_writer is not None:
            try:
                payload = [doc for doc in documents]
                self._sparse_writer.run(documents=payload)
            except Exception as exc:  # pragma: no cover - error propagation
                opensearch_ok = False
                logger.exception(
                    "opensearch_indexing_failed", index=self._opensearch_index, error=str(exc)
                )
                raise

        faiss_ok = True
        if self._dense_writer is not None:
            try:
                payload = [doc for doc in documents]
                self._dense_writer.run(documents=payload)
            except Exception as exc:  # pragma: no cover - error propagation
                faiss_ok = False
                logger.exception("faiss_indexing_failed", index=self._faiss_index, error=str(exc))
                raise

        return IndexReceipt(
            chunks_indexed=len(documents),
            opensearch_ok=opensearch_ok,
            faiss_ok=faiss_ok,
            metadata={"index": self._opensearch_index, "faiss_index": self._faiss_index},
        )


class HaystackRetriever:
    """Hybrid retriever combining lexical and dense search outputs."""

    def __init__(
        self,
        *,
        bm25_retriever: Any | None = None,
        dense_retriever: Any | None = None,
        fusion_ranker: Any | None = None,
        top_k: int = 20,
    ) -> None:
        if bm25_retriever is None:  # pragma: no cover - requires haystack runtime
            try:
                from haystack_integrations.components.retrievers.opensearch import (  # type: ignore
                    OpenSearchBM25Retriever,
                )

                bm25_retriever = OpenSearchBM25Retriever(top_k=top_k)
            except ImportError:  # pragma: no cover - fallback for tests
                logger.warning("haystack.bm25.stub", top_k=top_k)
                bm25_retriever = _StubRetrieverComponent("bm25", top_k)
        if dense_retriever is None:  # pragma: no cover - requires haystack runtime
            try:
                from haystack_integrations.components.retrievers.faiss import (  # type: ignore
                    FAISSDocumentRetriever,
                )

                dense_retriever = FAISSDocumentRetriever(top_k=top_k)
            except ImportError:  # pragma: no cover - fallback for tests
                logger.warning("haystack.faiss.stub", top_k=top_k)
                dense_retriever = _StubRetrieverComponent("dense", top_k)
        if fusion_ranker is None:  # pragma: no cover - requires haystack runtime
            try:
                from haystack.components.rankers import ReciprocalRankFusionRanker  # type: ignore

                fusion_ranker = ReciprocalRankFusionRanker(top_k=top_k)
            except ImportError:  # pragma: no cover - fallback for tests
                logger.warning("haystack.rank.stub", top_k=top_k)
                fusion_ranker = _StubRanker(top_k)
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._ranker = fusion_ranker
        self._top_k = top_k

    def retrieve(
        self, query: str, *, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        lexical = self._bm25.run(query=query, filters=filters or {})
        dense = self._dense.run(query=query, filters=filters or {})
        lexical_docs = lexical.get("documents") if isinstance(lexical, dict) else lexical
        dense_docs = dense.get("documents") if isinstance(dense, dict) else dense
        fused = self._ranker.run(documents=(lexical_docs or []) + (dense_docs or []))
        documents = fused.get("documents") if isinstance(fused, dict) else fused or []
        results: list[dict[str, Any]] = []
        for doc in documents[: self._top_k]:
            results.append(
                {
                    "id": getattr(doc, "id", None),
                    "score": getattr(doc, "score", None),
                    "content": getattr(doc, "content", None),
                    "meta": dict(getattr(doc, "meta", {}) or {}),
                }
            )
        return results


__all__ = [
    "HaystackChunker",
    "HaystackEmbedder",
    "HaystackIndexWriter",
    "HaystackRetriever",
    "HaystackSparseExpander",
]
