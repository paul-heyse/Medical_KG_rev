from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import inf
from typing import Any

import numpy as np

from ..base import EmbeddingContextualChunker
from ..exceptions import ChunkerConfigurationError
from ..provenance import BlockContext
from ..segmentation import Segment
from ..tokenization import TokenCounter


class SemanticSplitterChunker(EmbeddingContextualChunker):
    name = "semantic_splitter"
    version = "v1"
    segment_type = "semantic"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        tau_coh: float = 0.82,
        min_tokens: int = 200,
        gpu_semantic_checks: bool = False,  # Deprecated parameter, kept for compatibility
        encoder: object | None = None,
    ) -> None:
        super().__init__(
            token_counter=token_counter,
            model_name=model_name,
            gpu_semantic_checks=gpu_semantic_checks,
            encoder=encoder,
        )
        self.tau_coh = tau_coh
        self.min_tokens = min_tokens

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        context_list = list(contexts)
        if not context_list:
            return []
        embeddings = self.encode_contexts(context_list)
        boundaries = self._find_boundaries(context_list, embeddings)
        segments: list[Segment] = []
        start = 0
        for boundary in boundaries:
            window = context_list[start:boundary]
            if window:
                segments.append(Segment(contexts=list(window)))
            start = boundary
        tail = context_list[start:]
        if tail:
            segments.append(Segment(contexts=list(tail)))
        return segments

    def explain(self) -> dict[str, object]:
        return {"tau_coh": self.tau_coh, "min_tokens": self.min_tokens}

    def _find_boundaries(
        self, contexts: Sequence[BlockContext], embeddings: np.ndarray
    ) -> list[int]:
        if embeddings.size == 0:
            return [len(contexts)]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.clip(norms, a_min=1e-9, a_max=inf)
        sims = np.sum(normalized[1:] * normalized[:-1], axis=1)
        boundaries: list[int] = []
        token_budget = 0
        for idx, (ctx, sim) in enumerate(zip(contexts[1:], sims, strict=False), start=1):
            token_budget += ctx.token_count
            if token_budget >= self.min_tokens and sim < self.tau_coh:
                boundaries.append(idx)
                token_budget = 0
        boundaries.append(len(contexts))
        return boundaries


class SemanticClusterChunker(EmbeddingContextualChunker):
    name = "semantic_cluster"
    version = "v1"
    segment_type = "semantic_cluster"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        clusterer: str = "agglomerative",
        distance_threshold: float = 0.35,
        min_cluster_size: int = 3,
        gpu_semantic_checks: bool = False,  # Deprecated parameter, kept for compatibility
        encoder: object | None = None,
    ) -> None:
        super().__init__(
            token_counter=token_counter,
            model_name=model_name,
            gpu_semantic_checks=gpu_semantic_checks,
            encoder=encoder,
        )
        self.clusterer = clusterer
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        context_list = list(contexts)
        if not context_list:
            return []
        embeddings = self.encode_contexts(context_list)
        labels = self._cluster(embeddings)
        if not labels:
            labels = [0] * len(context_list)
        segments: list[Segment] = []
        start = 0
        last_label = labels[0]
        for idx, label in enumerate(labels):
            if label != last_label:
                window = context_list[start:idx]
                if window:
                    segments.append(
                        Segment(
                            contexts=list(window),
                            metadata={"cluster": int(last_label)},
                        )
                    )
                start = idx
                last_label = label
        tail = context_list[start:]
        if tail:
            segments.append(Segment(contexts=list(tail), metadata={"cluster": int(last_label)}))
        return segments

    def explain(self) -> dict[str, object]:
        return {
            "clusterer": self.clusterer,
            "distance_threshold": self.distance_threshold,
            "min_cluster_size": self.min_cluster_size,
        }

    def _cluster(self, embeddings: np.ndarray) -> list[int]:
        if embeddings.size == 0:
            return []
        if self.clusterer == "hdbscan":
            try:  # pragma: no cover - optional dependency
                import hdbscan
            except Exception:
                self.clusterer = "agglomerative"
            else:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    metric="euclidean",
                )
                result = clusterer.fit_predict(embeddings)
                return result.tolist() if hasattr(result, "tolist") else list(result)
        try:  # pragma: no cover - optional dependency
            from sklearn.cluster import AgglomerativeClustering
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


class GraphPartitionChunker(EmbeddingContextualChunker):
    name = "graph_partition"
    version = "v1"
    segment_type = "graph_partition"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.55,
        algorithm: str = "louvain",
        gpu_semantic_checks: bool = False,  # Deprecated parameter, kept for compatibility
        encoder: object | None = None,
    ) -> None:
        super().__init__(
            token_counter=token_counter,
            model_name=model_name,
            gpu_semantic_checks=gpu_semantic_checks,
            encoder=encoder,
        )
        self.similarity_threshold = similarity_threshold
        self.algorithm = algorithm

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        context_list = list(contexts)
        if not context_list:
            return []
        embeddings = self.encode_contexts(context_list)
        if embeddings.size == 0:
            return []
        graph = self._similarity_graph(embeddings)
        labels = self._partition(graph)
        segments: list[Segment] = []
        start = 0
        last_label = labels[0]
        for idx, label in enumerate(labels):
            if label != last_label:
                window = context_list[start:idx]
                if window:
                    segments.append(
                        Segment(
                            contexts=list(window),
                            metadata={"community": int(last_label)},
                        )
                    )
                start = idx
                last_label = label
        tail = context_list[start:]
        if tail:
            segments.append(Segment(contexts=list(tail), metadata={"community": int(last_label)}))
        return segments

    def explain(self) -> dict[str, object]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "algorithm": self.algorithm,
        }

    def _similarity_graph(self, embeddings: np.ndarray) -> Any:
        try:  # pragma: no cover - optional dependency
            import networkx as nx
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

    def _partition(self, graph: Any) -> list[int]:
        try:  # pragma: no cover - optional dependency
            import networkx as nx
        except Exception as exc:
            raise ChunkerConfigurationError(
                "networkx must be installed for GraphPartitionChunker"
            ) from exc
        if self.algorithm == "louvain" and hasattr(nx.algorithms.community, "louvain_communities"):
            communities = list(nx.algorithms.community.louvain_communities(graph))
        else:
            communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
        labels = [0] * graph.number_of_nodes()
        for community_id, nodes in enumerate(communities):
            for node in nodes:
                labels[int(node)] = community_id
        return labels
