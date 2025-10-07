"""Classical lexical and topic segmentation chunkers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..provenance import BlockContext, ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from ..ports import BaseChunker
from ..sentence_splitters import sentence_splitter_factory
from ..adapters.mapping import OffsetMapper


class TextTilingChunker(BaseChunker):
    name = "text_tiling"
    version = "v1"

    def __init__(
        self,
        *,
        w: int = 20,
        k: int = 10,
        smoothing_width: int = 2,
        smoothing_rounds: int = 1,
        token_counter: TokenCounter | None = None,
    ) -> None:
        try:  # pragma: no cover - optional dependency
            from nltk.tokenize.texttiling import TextTilingTokenizer  # type: ignore
        except Exception as exc:
            raise ChunkerConfigurationError(
                "nltk with the punkt dataset is required for TextTilingChunker"
            ) from exc
        self.tokenizer = TextTilingTokenizer(
            w=w, k=k, smoothing_width=smoothing_width, smoothing_rounds=smoothing_rounds
        )
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
        contexts = [
            ctx
            for ctx in self.normalizer.iter_block_contexts(document)
            if ctx.text and not ctx.is_table
        ]
        if not contexts:
            return []
        mapper = OffsetMapper(contexts, token_counter=self.counter)
        segments = self.tokenizer.tokenize(mapper.aggregated_text)
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        cursor = 0
        for segment in segments:
            projection = mapper.project(segment, start_hint=cursor)
            cursor = projection.end_offset
            if not projection.contexts:
                continue
            chunks.append(
                assembler.build(
                    projection.contexts,
                    metadata={"segment_type": "lexical", "algorithm": "text_tiling"},
                )
            )
        return chunks

    def explain(self) -> dict[str, object]:
        return {"algorithm": "TextTiling"}


class C99Chunker(BaseChunker):
    name = "c99"
    version = "v1"

    def __init__(
        self,
        *,
        block_size: int = 12,
        step: int = 6,
        similarity_window: int = 3,
        smooth_width: int = 2,
        cutoff: float = 0.35,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.block_size = block_size
        self.step = step
        self.similarity_window = similarity_window
        self.smooth_width = smooth_width
        self.cutoff = cutoff
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def _similarity_matrix(self, contexts: list[BlockContext]) -> np.ndarray:
        try:  # pragma: no cover - optional dependency
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        except Exception as exc:
            raise ChunkerConfigurationError(
                "scikit-learn must be installed for C99Chunker"
            ) from exc
        texts = [ctx.text for ctx in contexts]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        matrix = (tfidf * tfidf.T).toarray()
        np.fill_diagonal(matrix, 1.0)
        return matrix

    def _smooth(self, matrix: np.ndarray) -> np.ndarray:
        if self.smooth_width <= 1:
            return matrix
        kernel = np.ones((self.smooth_width, self.smooth_width))
        kernel /= kernel.size
        padded = np.pad(matrix, self.smooth_width // 2)
        smoothed = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                region = padded[i : i + self.smooth_width, j : j + self.smooth_width]
                smoothed[i, j] = float(np.sum(region * kernel))
        return smoothed

    def _depth_scores(self, matrix: np.ndarray) -> list[float]:
        n = matrix.shape[0]
        scores: list[float] = []
        for i in range(1, n - 1):
            left = matrix[max(0, i - self.similarity_window) : i, max(0, i - self.similarity_window) : i]
            right = matrix[i : min(n, i + self.similarity_window), i : min(n, i + self.similarity_window)]
            left_mean = float(np.mean(left)) if left.size else 0.0
            right_mean = float(np.mean(right)) if right.size else 0.0
            scores.append((left_mean + right_mean) / 2)
        return scores

    def _select_boundaries(self, scores: list[float]) -> list[int]:
        boundaries: list[int] = []
        avg = float(np.mean(scores)) if scores else 0.0
        for idx, score in enumerate(scores, start=1):
            if score < avg * self.cutoff:
                boundaries.append(idx)
        if scores:
            boundaries.append(len(scores) + 1)
        return boundaries

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        contexts = [
            ctx
            for ctx in self.normalizer.iter_block_contexts(document)
            if ctx.text and not ctx.is_table
        ]
        if not contexts:
            return []
        matrix = self._smooth(self._similarity_matrix(contexts))
        scores = self._depth_scores(matrix)
        boundaries = self._select_boundaries(scores)
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        start = 0
        for boundary in boundaries:
            window = contexts[start:boundary]
            if window:
                chunks.append(
                    assembler.build(
                        window,
                        metadata={"segment_type": "lexical", "algorithm": "c99"},
                    )
                )
            start = boundary
        return chunks

    def explain(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "similarity_window": self.similarity_window,
            "smooth_width": self.smooth_width,
            "cutoff": self.cutoff,
        }


class BayesSegChunker(BaseChunker):
    name = "bayes_seg"
    version = "v1"

    def __init__(
        self,
        *,
        n_components: int = 5,
        min_tokens: int = 120,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.n_components = n_components
        self.min_tokens = min_tokens
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def _fit_model(self, contexts: list[BlockContext]) -> list[int]:
        try:  # pragma: no cover - optional dependency
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.mixture import BayesianGaussianMixture  # type: ignore
        except Exception as exc:
            raise ChunkerConfigurationError(
                "scikit-learn must be installed for BayesSegChunker"
            ) from exc
        texts = [ctx.text for ctx in contexts]
        vectorizer = TfidfVectorizer(stop_words="english")
        features = vectorizer.fit_transform(texts).toarray()
        model = BayesianGaussianMixture(
            n_components=min(self.n_components, len(contexts)),
            weight_concentration_prior_type="dirichlet_process",
            max_iter=100,
        )
        labels = model.fit_predict(features)
        return labels.tolist()

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        contexts = [
            ctx
            for ctx in self.normalizer.iter_block_contexts(document)
            if ctx.text and not ctx.is_table
        ]
        if not contexts:
            return []
        labels = self._fit_model(contexts)
        boundaries: list[int] = []
        for idx in range(1, len(labels)):
            if labels[idx] != labels[idx - 1]:
                boundaries.append(idx)
        if len(labels) not in boundaries:
            boundaries.append(len(labels))
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        start = 0
        pending: list[BlockContext] = []
        for boundary in boundaries:
            pending.extend(contexts[start:boundary])
            token_total = sum(ctx.token_count for ctx in pending)
            if token_total < self.min_tokens and boundary != boundaries[-1]:
                start = boundary
                continue
            if pending:
                chunks.append(
                    assembler.build(
                        list(pending),
                        metadata={
                            "segment_type": "lexical",
                            "algorithm": "bayes_seg",
                            "components": self.n_components,
                        },
                    )
                )
            pending = []
            start = boundary
        if pending:
            chunks.append(
                assembler.build(
                    list(pending),
                    metadata={
                        "segment_type": "lexical",
                        "algorithm": "bayes_seg",
                        "components": self.n_components,
                    },
                )
            )
        return chunks

    def explain(self) -> dict[str, object]:
        return {"n_components": self.n_components, "min_tokens": self.min_tokens}


class LDATopicChunker(BaseChunker):
    name = "lda_topic"
    version = "v1"

    def __init__(
        self,
        *,
        num_topics: int = 8,
        passes: int = 2,
        coherence_threshold: float = 0.3,
        token_counter: TokenCounter | None = None,
    ) -> None:
        try:  # pragma: no cover - optional dependency
            from gensim.corpora import Dictionary  # type: ignore
            from gensim.models import LdaModel  # type: ignore
        except Exception as exc:
            raise ChunkerConfigurationError(
                "gensim must be installed for LDATopicChunker"
            ) from exc
        self.Dictionary = Dictionary
        self.LdaModel = LdaModel
        self.num_topics = num_topics
        self.passes = passes
        self.coherence_threshold = coherence_threshold
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def _tokenize(self, text: str) -> list[str]:
        splitter = sentence_splitter_factory("nltk")
        tokens: list[str] = []
        for sentence in splitter.split(text):
            tokens.extend(token.lower() for token in sentence.split())
        return tokens

    def _topic_assignments(self, contexts: list[BlockContext]) -> list[int]:
        corpus_tokens = [self._tokenize(ctx.text) for ctx in contexts]
        dictionary = self.Dictionary(corpus_tokens)
        bow = [dictionary.doc2bow(tokens) for tokens in corpus_tokens]
        if not bow:
            return [0 for _ in contexts]
        lda = self.LdaModel(
            corpus=bow,
            id2word=dictionary,
            num_topics=min(self.num_topics, len(corpus_tokens)),
            passes=self.passes,
        )
        assignments: list[int] = []
        for vector in bow:
            topic_dist = lda.get_document_topics(vector, minimum_probability=0.0)
            topic_dist.sort(key=lambda item: item[1], reverse=True)
            assignments.append(topic_dist[0][0] if topic_dist else 0)
        return assignments

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        contexts = [
            ctx
            for ctx in self.normalizer.iter_block_contexts(document)
            if ctx.text and not ctx.is_table
        ]
        if not contexts:
            return []
        topics = self._topic_assignments(contexts)
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )
        segments: list[tuple[list[BlockContext], int]] = []
        start = 0
        last_topic = topics[0] if topics else 0
        for idx in range(1, len(topics)):
            if topics[idx] != last_topic:
                segment_contexts = contexts[start:idx]
                if segment_contexts:
                    segments.append((segment_contexts, int(last_topic)))
                start = idx
                last_topic = topics[idx]
        tail = contexts[start:]
        if tail:
            segments.append((tail, int(last_topic)))
        merged_segments: list[tuple[list[BlockContext], int]] = []
        for context_list, topic in segments:
            if merged_segments and topic == merged_segments[-1][1]:
                merged_segments[-1][0].extend(context_list)
            else:
                merged_segments.append((list(context_list), topic))
        chunks: list[Chunk] = []
        for context_list, topic in merged_segments:
            chunks.append(
                assembler.build(
                    context_list,
                    metadata={
                        "segment_type": "topic",
                        "algorithm": "lda",
                        "topic": topic,
                    },
                )
            )
        return chunks

    def explain(self) -> dict[str, object]:
        return {
            "num_topics": self.num_topics,
            "passes": self.passes,
            "coherence_threshold": self.coherence_threshold,
        }

