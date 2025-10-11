"""Classical chunking strategies using traditional NLP approaches."""

from __future__ import annotations

from collections.abc import Iterable

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import BlockContext, ProvenanceNormalizer
from ..sentence_splitters import sentence_splitter_factory
from ..tokenization import TokenCounter, default_token_counter


class ClassicalChunker(BaseChunker):
    """Base class for classical chunking strategies."""

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize the classical chunker."""
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        """Chunk a document using classical strategies."""
        contexts = [ctx for ctx in self.normalizer.iter_block_contexts(document) if ctx.text]
        if not contexts:
            return []

        # Use sentence splitter for classical chunking
        splitter = sentence_splitter_factory.create("nltk")
        aggregated_text = " ".join(ctx.text for ctx in contexts)
        sentences = splitter.split(aggregated_text)

        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = self.counter.count_tokens(sentence)
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Create chunk objects
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name="classical",
            chunker_version="v1",
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )

        result_chunks = []
        for chunk_text in chunks:
            # Find the context that corresponds to this chunk
            chunk_contexts = []
            for ctx in contexts:
                if ctx.text in chunk_text:
                    chunk_contexts.append(ctx)

            if chunk_contexts:
                chunk_meta = {
                    "segment_type": "classical",
                    "chunk_size": len(chunk_text),
                    "token_count": self.counter.count_tokens(chunk_text),
                }
                result_chunks.append(assembler.build(chunk_contexts, metadata=chunk_meta))

        return result_chunks

    def explain(self) -> dict[str, object]:
        """Explain the chunking strategy."""
        return {
            "strategy": "classical",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


class LDATopicChunker(BaseChunker):
    """Chunker that uses LDA topic modeling for semantic chunking."""

    def __init__(
        self,
        *,
        num_topics: int = 10,
        passes: int = 20,
        coherence_threshold: float = 0.5,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize the LDA topic chunker."""
        super().__init__()
        try:
            from gensim.corpora import Dictionary
            from gensim.models import LdaModel
        except ImportError as exc:
            raise ChunkerConfigurationError("gensim must be installed for LDATopicChunker") from exc

        self.Dictionary = Dictionary
        self.LdaModel = LdaModel
        self.num_topics = num_topics
        self.passes = passes
        self.coherence_threshold = coherence_threshold
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        """Chunk a document using LDA topic modeling."""
        contexts = [ctx for ctx in self.normalizer.iter_block_contexts(document) if ctx.text]
        if not contexts:
            return []

        # Prepare text for LDA
        texts = [ctx.text.lower().split() for ctx in contexts]
        dictionary = self.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        # Train LDA model
        lda_model = self.LdaModel(
            corpus,
            num_topics=self.num_topics,
            passes=self.passes,
            id2word=dictionary,
        )

        # Group contexts by dominant topic
        topic_groups = {}
        for i, ctx in enumerate(contexts):
            doc_topics = lda_model.get_document_topics(corpus[i])
            dominant_topic = max(doc_topics, key=lambda x: x[1])[0]
            if dominant_topic not in topic_groups:
                topic_groups[dominant_topic] = []
            topic_groups[dominant_topic].append(ctx)

        # Create chunks from topic groups
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name="lda_topic",
            chunker_version="v1",
            granularity=granularity or "topic",
            token_counter=self.counter,
        )

        result_chunks = []
        for topic_id, topic_contexts in topic_groups.items():
            chunk_meta = {
                "segment_type": "lda_topic",
                "topic_id": topic_id,
                "context_count": len(topic_contexts),
            }
            result_chunks.append(assembler.build(topic_contexts, metadata=chunk_meta))

        return result_chunks

    def explain(self) -> dict[str, object]:
        """Explain the chunking strategy."""
        return {
            "strategy": "lda_topic",
            "num_topics": self.num_topics,
            "passes": self.passes,
            "coherence_threshold": self.coherence_threshold,
        }
