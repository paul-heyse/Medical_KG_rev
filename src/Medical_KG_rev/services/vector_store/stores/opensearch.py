"""Simulated OpenSearch-compatible vector store with hybrid search support."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import time

import numpy as np

from ..errors import BackendUnavailableError, NamespaceNotFoundError
from ..models import (
    CompressionPolicy,
    HealthStatus,
    IndexParams,
    RebuildReport,
    SnapshotInfo,
    VectorMatch,
    VectorQuery,
    VectorRecord,
)
from ..types import VectorStorePort
from .memory import InMemoryVectorStore


_SPECIAL_FILTER_KEYS = {"lexical_query", "mode", "vector_weight"}


@dataclass(slots=True)
class _NamespaceState:
    params: IndexParams
    compression: CompressionPolicy
    metadata: dict[str, Mapping[str, object]] = field(default_factory=dict)
    lexical_tokens: dict[str, set[str]] = field(default_factory=dict)
    named_vectors: Mapping[str, IndexParams] | None = None
    engine: str = "lucene"
    trained: bool = False
    encoder: str | None = None
    rank_profiles: dict[str, Mapping[str, object]] = field(default_factory=dict)


class OpenSearchKNNStore(VectorStorePort):
    """Hybrid vector store that mimics the OpenSearch k-NN plugin."""

    def __init__(self, *, default_engine: str = "lucene") -> None:
        self._delegate = InMemoryVectorStore()
        self._default_engine = default_engine
        self._states: dict[tuple[str, str], _NamespaceState] = {}

    # ------------------------------------------------------------------
    # VectorStorePort API
    # ------------------------------------------------------------------
    def create_or_update_collection(
        self,
        *,
        tenant_id: str,
        namespace: str,
        params: IndexParams,
        compression: CompressionPolicy,
        metadata: Mapping[str, object] | None = None,
        named_vectors: Mapping[str, IndexParams] | None = None,
    ) -> None:
        self._delegate.create_or_update_collection(
            tenant_id=tenant_id,
            namespace=namespace,
            params=params,
            compression=compression,
            metadata=metadata,
            named_vectors=named_vectors,
        )
        state = self._states.get((tenant_id, namespace))
        engine = str((metadata or {}).get("engine", self._default_engine)).lower()
        encoder = (metadata or {}).get("encoder")
        if state:
            state.params = params
            state.compression = compression
            state.named_vectors = named_vectors
            state.engine = engine
            if encoder:
                state.encoder = str(encoder)
            return
        state = _NamespaceState(
            params=params,
            compression=compression,
            named_vectors=named_vectors,
            engine=engine,
        )
        if encoder:
            state.encoder = str(encoder)
        profiles = (metadata or {}).get("rank_profiles")
        if isinstance(profiles, Mapping):
            for name, config in profiles.items():
                if isinstance(config, Mapping):
                    state.rank_profiles[name] = dict(config)
        self._states[(tenant_id, namespace)] = state

    def list_collections(self, *, tenant_id: str) -> Sequence[str]:
        return self._delegate.list_collections(tenant_id=tenant_id)

    def upsert(
        self,
        *,
        tenant_id: str,
        namespace: str,
        records: Sequence[VectorRecord],
    ) -> None:
        state = self._ensure_state(tenant_id, namespace)
        self._delegate.upsert(
            tenant_id=tenant_id,
            namespace=namespace,
            records=records,
        )
        for record in records:
            state.metadata[record.vector_id] = dict(record.metadata)
            document = str(record.metadata.get("text", ""))
            tokens = self._tokenise(document)
            if tokens:
                state.lexical_tokens[record.vector_id] = tokens
            elif record.vector_id in state.lexical_tokens:
                state.lexical_tokens.pop(record.vector_id, None)

    def query(
        self,
        *,
        tenant_id: str,
        namespace: str,
        query: VectorQuery,
    ) -> Sequence[VectorMatch]:
        state = self._ensure_state(tenant_id, namespace)
        self._ensure_trained(state, namespace)

        filters = dict(query.filters or {})
        lexical_query = str(filters.pop("lexical_query", "")) if "lexical_query" in filters else ""
        mode = str(filters.pop("mode", "hybrid" if lexical_query else "vector")).lower()
        vector_weight = float(filters.pop("vector_weight", 0.5))
        vector_weight = min(max(vector_weight, 0.0), 1.0)

        vector_filters = {
            key: value for key, value in filters.items() if key not in _SPECIAL_FILTER_KEYS
        }
        vector_matches: list[VectorMatch] = []
        if mode in {"vector", "hybrid"}:
            vector_matches = list(
                self._delegate.query(
                    tenant_id=tenant_id,
                    namespace=namespace,
                    query=VectorQuery(
                        values=query.values,
                        top_k=max(query.top_k, 1),
                        filters=None,
                        vector_name=query.vector_name,
                    ),
                )
            )
            vector_matches = [
                match
                for match in vector_matches
                if self._match_filters(state.metadata.get(match.vector_id, {}), vector_filters)
            ]

        lexical_scores: dict[str, float] = {}
        if lexical_query and mode in {"lexical", "hybrid"}:
            lexical_scores = self._lexical_scores(state, lexical_query, vector_filters)

        if mode == "lexical":
            return self._build_from_scores(state, lexical_scores, query.top_k)
        if not vector_matches and not lexical_scores:
            return []
        if not lexical_scores:
            return vector_matches[: query.top_k]

        combined = self._combine_scores(state, vector_matches, lexical_scores, vector_weight)
        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[: query.top_k]

    def delete(
        self,
        *,
        tenant_id: str,
        namespace: str,
        vector_ids: Sequence[str],
    ) -> int:
        state = self._ensure_state(tenant_id, namespace)
        removed = self._delegate.delete(
            tenant_id=tenant_id,
            namespace=namespace,
            vector_ids=vector_ids,
        )
        for vector_id in vector_ids:
            state.metadata.pop(vector_id, None)
            state.lexical_tokens.pop(vector_id, None)
        return removed

    def create_snapshot(
        self,
        *,
        tenant_id: str,
        namespace: str,
        destination: str,
        include_payloads: bool = True,
    ) -> SnapshotInfo:
        snapshot = self._delegate.create_snapshot(
            tenant_id=tenant_id,
            namespace=namespace,
            destination=destination,
            include_payloads=include_payloads,
        )
        path = Path(snapshot.path)
        data = json.loads(path.read_text())
        state = self._states.get((tenant_id, namespace))
        if state:
            data["opensearch"] = {
                "engine": state.engine,
                "rank_profiles": {name: dict(cfg) for name, cfg in state.rank_profiles.items()},
                "lexical_tokens": {
                    key: sorted(tokens) for key, tokens in state.lexical_tokens.items()
                },
                "trained": state.trained,
            }
        path.write_text(json.dumps(data, sort_keys=True))
        stats = path.stat()
        metadata = dict(snapshot.metadata or {})
        if state:
            metadata["engine"] = state.engine
        return SnapshotInfo(
            namespace=namespace,
            path=str(path),
            size_bytes=stats.st_size,
            created_at=time.time(),
            metadata=metadata,
        )

    def restore_snapshot(
        self,
        *,
        tenant_id: str,
        namespace: str,
        source: str,
        overwrite: bool = False,
    ) -> RebuildReport:
        report = self._delegate.restore_snapshot(
            tenant_id=tenant_id,
            namespace=namespace,
            source=source,
            overwrite=overwrite,
        )
        state = self._ensure_state(tenant_id, namespace)
        payload = json.loads(Path(source).read_text())
        extra = payload.get("opensearch", {})
        if extra:
            state.engine = extra.get("engine", state.engine)
            state.trained = bool(extra.get("trained", state.trained))
            lexical_tokens = extra.get("lexical_tokens", {})
            state.lexical_tokens = {key: set(values) for key, values in lexical_tokens.items()}
            profiles = extra.get("rank_profiles", {})
            state.rank_profiles = {name: dict(cfg) for name, cfg in profiles.items()}
        return RebuildReport(
            namespace=namespace,
            rebuilt=report.rebuilt,
            details={"engine": state.engine, "documents": len(state.metadata)},
        )

    def rebuild_index(
        self,
        *,
        tenant_id: str,
        namespace: str,
        force: bool = False,
    ) -> RebuildReport:
        state = self._ensure_state(tenant_id, namespace)
        if state.engine == "faiss":
            state.trained = True
        return RebuildReport(
            namespace=namespace,
            rebuilt=True,
            details={"engine": state.engine, "trained": state.trained},
        )

    def check_health(
        self,
        *,
        tenant_id: str,
        namespace: str | None = None,
    ) -> Mapping[str, HealthStatus]:
        def build_status(ns: str, state: _NamespaceState) -> HealthStatus:
            healthy = state.engine != "faiss" or state.trained
            return HealthStatus(
                name=ns,
                healthy=healthy,
                details={
                    "engine": state.engine,
                    "documents": len(state.metadata),
                },
            )

        if namespace is not None:
            state = self._states.get((tenant_id, namespace))
            if not state:
                return {namespace: HealthStatus(name=namespace, healthy=False, details={})}
            return {namespace: build_status(namespace, state)}
        statuses: dict[str, HealthStatus] = {}
        for (tenant, ns), state in self._states.items():
            if tenant != tenant_id:
                continue
            statuses[ns] = build_status(ns, state)
        return statuses

    # ------------------------------------------------------------------
    # OpenSearch specific helpers
    # ------------------------------------------------------------------
    def capabilities(self) -> Mapping[str, object]:
        return {
            "supports_hybrid": True,
            "engines": ["lucene", "faiss"],
            "compression": ["none", "int8", "fp16", "pq"],
            "requires_training": [
                key for key, state in self._states.items() if state.engine == "faiss"
            ],
        }

    def train_index(
        self,
        *,
        tenant_id: str,
        namespace: str,
        samples: Sequence[Sequence[float]],
        encoder: str | None = None,
    ) -> Mapping[str, Any]:
        state = self._ensure_state(tenant_id, namespace)
        if not samples:
            raise ValueError("training samples must not be empty")
        matrix = np.asarray(samples, dtype=float)
        centroid = matrix.mean(axis=0)
        state.trained = True
        if encoder:
            state.encoder = encoder
        state.rank_profiles["trained"] = {"centroid": centroid.tolist()}
        return {"centroid": centroid.tolist(), "samples": int(matrix.shape[0])}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_state(self, tenant_id: str, namespace: str) -> _NamespaceState:
        state = self._states.get((tenant_id, namespace))
        if not state:
            raise NamespaceNotFoundError(namespace, tenant_id=tenant_id)
        return state

    def _ensure_trained(self, state: _NamespaceState, namespace: str) -> None:
        if state.engine != "faiss":
            return
        if not state.trained:
            raise BackendUnavailableError(
                f"Namespace '{namespace}' requires training before querying the FAISS engine"
            )

    def _tokenise(self, value: str) -> set[str]:
        return {token for token in value.lower().split() if token}

    def _match_filters(self, metadata: Mapping[str, object], filters: Mapping[str, object]) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if metadata.get(key) != expected:
                return False
        return True

    def _lexical_scores(
        self,
        state: _NamespaceState,
        query: str,
        filters: Mapping[str, object],
    ) -> dict[str, float]:
        tokens = self._tokenise(query)
        if not tokens:
            return {}
        scores: dict[str, float] = {}
        for vector_id, doc_tokens in state.lexical_tokens.items():
            metadata = state.metadata.get(vector_id, {})
            if not self._match_filters(metadata, filters):
                continue
            overlap = tokens.intersection(doc_tokens)
            if not overlap:
                continue
            scores[vector_id] = len(overlap) / len(tokens)
        return scores

    def _build_from_scores(
        self,
        state: _NamespaceState,
        scores: Mapping[str, float],
        top_k: int,
    ) -> list[VectorMatch]:
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        matches: list[VectorMatch] = []
        for vector_id, score in ranked[:top_k]:
            matches.append(
                VectorMatch(
                    vector_id=vector_id,
                    score=float(score),
                    metadata=state.metadata.get(vector_id, {}),
                )
            )
        return matches

    def _combine_scores(
        self,
        state: _NamespaceState,
        vector_matches: Sequence[VectorMatch],
        lexical_scores: Mapping[str, float],
        vector_weight: float,
    ) -> list[VectorMatch]:
        vector_norm = self._normalise({match.vector_id: match.score for match in vector_matches})
        lexical_norm = self._normalise(dict(lexical_scores))
        combined: dict[str, float] = defaultdict(float)
        for vector_id, score in vector_norm.items():
            combined[vector_id] += vector_weight * score
        for vector_id, score in lexical_norm.items():
            combined[vector_id] += (1.0 - vector_weight) * score
        results: list[VectorMatch] = []
        for vector_id, score in combined.items():
            metadata = next(
                (match.metadata for match in vector_matches if match.vector_id == vector_id),
                state.metadata.get(vector_id, {}),
            )
            results.append(VectorMatch(vector_id=vector_id, score=float(score), metadata=metadata))
        if not results:
            return []
        return results

    def _normalise(self, scores: Mapping[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_value = max(scores.values())
        if max_value <= 0:
            return {key: 0.0 for key in scores}
        return {key: value / max_value for key, value in scores.items()}


__all__ = ["OpenSearchKNNStore"]
