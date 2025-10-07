"""Semantic splitter chunker based on embedding coherence."""

from __future__ import annotations

from math import inf
from typing import Iterable

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..provenance import BlockContext, ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from ..ports import BaseChunker


def _resolve_encoder(
    model_name: str,
    *,
    gpu_semantic_checks: bool,
    encoder: object | None,
) -> object:
    if encoder is not None:
        return encoder
    if SentenceTransformer is None:
        raise ChunkerConfigurationError(
            "sentence-transformers must be installed for semantic chunkers"
        )
    encoder = SentenceTransformer(model_name)
    if gpu_semantic_checks:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("GPU semantic checks requested but CUDA is not available")
        encoder = encoder.to("cuda")
    return encoder


class SemanticSplitterChunker(BaseChunker):
    name = "semantic_splitter"
    version = "v1"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        tau_coh: float = 0.82,
        min_tokens: int = 200,
        gpu_semantic_checks: bool = False,
        encoder: object | None = None,
    ) -> None:
        encoder = _resolve_encoder(
            model_name,
            gpu_semantic_checks=gpu_semantic_checks,
            encoder=encoder,
        )
        self.counter = token_counter or default_token_counter()
        self.model = encoder
        self.tau_coh = tau_coh
        self.min_tokens = min_tokens
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
        embeddings = self._encode(contexts)
        boundaries = self._find_boundaries(contexts, embeddings)
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
                    assembler.build(window, metadata={"segment_type": "semantic"})
                )
            start = boundary
        tail = contexts[start:]
        if tail:
            chunks.append(assembler.build(tail, metadata={"segment_type": "semantic"}))
        return chunks

    def explain(self) -> dict[str, object]:
        return {"tau_coh": self.tau_coh, "min_tokens": self.min_tokens}

    def _encode(self, contexts: list[BlockContext]) -> np.ndarray:
        sentences = [ctx.text for ctx in contexts]
        if not sentences:
            return np.empty((0, 1))
        encode = getattr(self.model, "encode", None)
        if encode is None:
            raise ChunkerConfigurationError("Encoder does not expose an encode() method")
        result = encode(sentences, convert_to_numpy=True)  # type: ignore[arg-type]
        return np.asarray(result)

    def _find_boundaries(self, contexts: list[BlockContext], embeddings: np.ndarray) -> list[int]:
        if embeddings.size == 0:
            return [len(contexts)]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.clip(norms, a_min=1e-9, a_max=inf)
        sims = np.sum(normalized[1:] * normalized[:-1], axis=1)
        boundaries = []
        token_budget = 0
        for idx, (ctx, sim) in enumerate(zip(contexts[1:], sims, strict=False), start=1):
            token_budget += ctx.token_count
            if token_budget >= self.min_tokens and sim < self.tau_coh:
                boundaries.append(idx)
                token_budget = 0
        boundaries.append(len(contexts))
        return boundaries


class SemanticClusterChunker(BaseChunker):
    name = "semantic_cluster"
    version = "v1"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        clusterer: str = "agglomerative",
        distance_threshold: float = 0.35,
        min_cluster_size: int = 3,
        gpu_semantic_checks: bool = False,
        encoder: object | None = None,
    ) -> None:
        self.counter = token_counter or default_token_counter()
        self.model = _resolve_encoder(
            model_name,
            gpu_semantic_checks=gpu_semantic_checks,
            encoder=encoder,
        )
        self.clusterer = clusterer
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def _encode(self, contexts: list[BlockContext]) -> np.ndarray:
        sentences = [ctx.text for ctx in contexts]
        if not sentences:
            return np.empty((0, 1))
        encode = getattr(self.model, "encode", None)
        if encode is None:
            raise ChunkerConfigurationError("Encoder does not expose an encode() method")
        return np.asarray(encode(sentences, convert_to_numpy=True))  # type: ignore[arg-type]

    def _cluster(self, embeddings: np.ndarray) -> list[int]:
        if embeddings.size == 0:
            return []
        if self.clusterer == "hdbscan":
            try:  # pragma: no cover - optional dependency
                import hdbscan  # type: ignore
            except Exception:
                self.clusterer = "agglomerative"
            else:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    metric="euclidean",
                )
                return clusterer.fit_predict(embeddings).tolist()
        try:  # pragma: no cover - optional dependency
            from sklearn.cluster import AgglomerativeClustering  # type: ignore
        except Exception as exc:
            raise ChunkerConfigurationError(
                "scikit-learn must be installed for SemanticClusterChunker"
            ) from exc
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
        )
        labels = clusterer.fit_predict(embeddings)
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
        embeddings = self._encode(contexts)
        labels = self._cluster(embeddings)
        if not labels:
            labels = [0] * len(contexts)
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
        last_label = labels[0]
        for idx, label in enumerate(labels):
            if label != last_label:
                window = contexts[start:idx]
                if window:
                    chunks.append(
                        assembler.build(
                            window,
                            metadata={
                                "segment_type": "semantic_cluster",
                                "cluster": int(last_label),
                            },
                        )
                    )
                start = idx
                last_label = label
        tail = contexts[start:]
        if tail:
            chunks.append(
                assembler.build(
                    tail,
                    metadata={
                        "segment_type": "semantic_cluster",
                        "cluster": int(last_label),
                    },
                )
            )
        return chunks

    def explain(self) -> dict[str, object]:
        return {
            "clusterer": self.clusterer,
            "distance_threshold": self.distance_threshold,
            "min_cluster_size": self.min_cluster_size,
        }


class GraphPartitionChunker(BaseChunker):
    name = "graph_partition"
    version = "v1"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.55,
        algorithm: str = "louvain",
        gpu_semantic_checks: bool = False,
        encoder: object | None = None,
    ) -> None:
        self.counter = token_counter or default_token_counter()
        self.model = _resolve_encoder(
            model_name,
            gpu_semantic_checks=gpu_semantic_checks,
            encoder=encoder,
        )
        self.similarity_threshold = similarity_threshold
        self.algorithm = algorithm
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def _encode(self, contexts: list[BlockContext]) -> np.ndarray:
        sentences = [ctx.text for ctx in contexts]
        if not sentences:
            return np.empty((0, 1))
        encode = getattr(self.model, "encode", None)
        if encode is None:
            raise ChunkerConfigurationError("Encoder does not expose an encode() method")
        return np.asarray(encode(sentences, convert_to_numpy=True))  # type: ignore[arg-type]

    def _similarity_graph(self, embeddings: np.ndarray):
        try:  # pragma: no cover - optional dependency
            import networkx as nx  # type: ignore
        except Exception as exc:
            raise ChunkerConfigurationError(
                "networkx must be installed for GraphPartitionChunker"
            ) from exc
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        sim_matrix = normalized @ normalized.T
        graph = nx.Graph()
        for idx in range(sim_matrix.shape[0]):
            graph.add_node(idx)
        for i in range(sim_matrix.shape[0]):
            for j in range(i + 1, sim_matrix.shape[0]):
                weight = float(sim_matrix[i, j])
                if weight >= self.similarity_threshold:
                    graph.add_edge(i, j, weight=weight)
        return graph

    def _partition(self, graph) -> list[int]:
        try:  # pragma: no cover - optional dependency
            import networkx as nx  # type: ignore
        except Exception as exc:
            raise ChunkerConfigurationError(
                "networkx must be installed for GraphPartitionChunker"
            ) from exc
        if self.algorithm == "louvain" and hasattr(
            nx.algorithms.community, "louvain_communities"
        ):
            communities = list(
                nx.algorithms.community.louvain_communities(  # type: ignore[attr-defined]
                    graph
                )
            )
        else:
            communities = list(
                nx.algorithms.community.greedy_modularity_communities(graph)
            )
        labels = [0] * graph.number_of_nodes()
        for community_id, nodes in enumerate(communities):
            for node in nodes:
                labels[int(node)] = community_id
        return labels

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
        embeddings = self._encode(contexts)
        if embeddings.size == 0:
            return []
        graph = self._similarity_graph(embeddings)
        labels = self._partition(graph)
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
        last_label = labels[0]
        for idx, label in enumerate(labels):
            if label != last_label:
                window = contexts[start:idx]
                if window:
                    chunks.append(
                        assembler.build(
                            window,
                            metadata={
                                "segment_type": "graph_partition",
                                "community": int(last_label),
                            },
                        )
                    )
                start = idx
                last_label = label
        tail = contexts[start:]
        if tail:
            chunks.append(
                assembler.build(
                    tail,
                    metadata={
                        "segment_type": "graph_partition",
                        "community": int(last_label),
                    },
                )
            )
        return chunks

    def explain(self) -> dict[str, object]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "algorithm": self.algorithm,
        }
