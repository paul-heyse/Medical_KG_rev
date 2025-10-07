"""Advanced and discourse oriented chunkers."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

from Medical_KG_rev.models.ir import BlockType, Document

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import BlockContext, ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter


@dataclass(slots=True)
class _Community:
    nodes: list[int]
    score: float


class DiscourseSegmenterChunker(BaseChunker):
    """Chunker that approximates EDU segmentation using rhetorical connectives."""

    name = "discourse_segmenter"
    version = "v1"

    def __init__(
        self,
        *,
        connectives: Sequence[str] | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.connectives = [
            "however",
            "therefore",
            "moreover",
            "in contrast",
            "furthermore",
            "meanwhile",
            "additionally",
            "consequently",
        ]
        if connectives:
            self.connectives.extend(connective.lower() for connective in connectives)
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
        contexts = list(self.normalizer.iter_block_contexts(document))
        if not contexts:
            return []
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )
        boundaries = self._detect_boundaries(contexts)
        chunks: list[Chunk] = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=False):
            window = contexts[start:end]
            if not window:
                continue
            metadata = {"segment_type": "discourse", "connectives": self.connectives[:5]}
            chunks.append(assembler.build(window, metadata=metadata))
        return chunks

    def explain(self) -> dict[str, object]:
        return {"connectives": list(self.connectives)}

    def _detect_boundaries(self, contexts: Sequence[BlockContext]) -> list[int]:
        boundaries = [0]
        token_budget = 0
        for index, ctx in enumerate(contexts[1:], start=1):
            token_budget += ctx.token_count
            lowered = ctx.text.lower()
            if ctx.block.type == BlockType.HEADER:
                boundaries.append(index)
                token_budget = 0
                continue
            if any(lowered.startswith(connective) for connective in self.connectives):
                boundaries.append(index)
                token_budget = 0
                continue
            if token_budget >= 220:
                boundaries.append(index)
                token_budget = 0
        boundaries.append(len(contexts))
        return sorted(set(boundaries))


class GrobidSectionChunker(BaseChunker):
    """Chunker that aligns MinerU output with Grobid TEI XML sections."""

    name = "grobid_section"
    version = "v1"

    def __init__(self, *, token_counter: TokenCounter | None = None) -> None:
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
        tei_xml = str(document.metadata.get("tei_xml", ""))
        if not tei_xml.strip():
            raise ChunkerConfigurationError("Document does not provide TEI XML metadata")
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as exc:  # pragma: no cover - invalid TEI is rare
            raise ChunkerConfigurationError("Invalid TEI XML payload") from exc
        sequence = list(self.normalizer.iter_block_contexts(document))
        if not sequence:
            return []
        section_map = self._tei_section_titles(root)
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "section",
            token_counter=self.counter,
        )
        grouped: dict[str, list[BlockContext]] = defaultdict(list)
        for ctx in sequence:
            key = section_map.get(ctx.section_title.lower(), ctx.section_title.lower())
            grouped[key].append(ctx)
        chunks: list[Chunk] = []
        for title, contexts in grouped.items():
            metadata = {"segment_type": "grobid", "tei_title": title}
            chunks.append(assembler.build(contexts, metadata=metadata))
        return chunks

    def explain(self) -> dict[str, object]:
        return {"strategy": "grobid-tei-alignment"}

    def _tei_section_titles(self, root: ET.Element) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for div in root.findall(".//{*}div"):
            type_attr = div.attrib.get("type") or "section"
            head = div.findtext("{*}head") or type_attr
            mapping[head.lower()] = head
        return mapping


class LayoutAwareChunker(BaseChunker):
    """Chunker that groups blocks using layout metadata from DocTR/Docling."""

    name = "layout_aware"
    version = "v1"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        overlap_threshold: float = 0.3,
    ) -> None:
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)
        self.overlap_threshold = overlap_threshold

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        contexts = list(self.normalizer.iter_block_contexts(document))
        if not contexts:
            return []
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "section",
            token_counter=self.counter,
        )
        groups = self._group_by_layout(contexts)
        chunks: list[Chunk] = []
        for _, bucket in sorted(groups.items()):
            metadata = {"segment_type": "layout", "region_count": len(bucket)}
            chunks.append(assembler.build(bucket, metadata=metadata))
        return chunks

    def explain(self) -> dict[str, object]:
        return {"overlap_threshold": self.overlap_threshold}

    def _group_by_layout(self, contexts: Sequence[BlockContext]) -> dict[str, list[BlockContext]]:
        buckets: dict[str, list[BlockContext]] = defaultdict(list)
        for ctx in contexts:
            region_id = str(ctx.block.metadata.get("layout_region") or ctx.section.id)
            buckets[region_id].append(ctx)
        return buckets


class GraphRAGChunker(BaseChunker):
    """Chunker that builds a lexical graph and emits community based chunks."""

    name = "graph_rag"
    version = "v1"

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.18,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
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
        graph = self._build_graph(contexts)
        communities = self._community_detection(graph)
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "section",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        for community in communities:
            window = [contexts[index] for index in community.nodes]
            summary = window[0].text.split(".", 1)[0]
            metadata = {
                "segment_type": "graph",
                "community_score": round(community.score, 3),
                "nodes": len(window),
                "summary": summary,
            }
            chunks.append(assembler.build(window, metadata=metadata))
        return chunks

    def explain(self) -> dict[str, object]:
        return {"similarity_threshold": self.similarity_threshold}

    def _build_graph(self, contexts: Sequence[BlockContext]) -> dict[int, set[int]]:
        graph: dict[int, set[int]] = defaultdict(set)
        vectors = [self._vectorize(ctx.text) for ctx in contexts]
        for i, left in enumerate(vectors):
            for j in range(i + 1, len(vectors)):
                right = vectors[j]
                score = self._cosine(left, right)
                if score >= self.similarity_threshold:
                    graph[i].add(j)
                    graph[j].add(i)
        return graph

    def _community_detection(self, graph: dict[int, set[int]]) -> list[_Community]:
        seen: set[int] = set()
        communities: list[_Community] = []
        for node in sorted(graph):
            if node in seen:
                continue
            queue: deque[int] = deque([node])
            members: list[int] = []
            while queue:
                current = queue.popleft()
                if current in seen:
                    continue
                seen.add(current)
                members.append(current)
                for neighbor in graph.get(current, set()):
                    if neighbor not in seen:
                        queue.append(neighbor)
            if not members:
                continue
            score = sum(len(graph.get(member, set())) for member in members) / max(len(members), 1)
            communities.append(_Community(nodes=sorted(members), score=score))
        communities.sort(key=lambda item: (item.score, len(item.nodes)), reverse=True)
        return communities

    def _vectorize(self, text: str) -> dict[str, float]:
        counts: dict[str, float] = defaultdict(float)
        for token in text.lower().split():
            if len(token) <= 2:
                continue
            counts[token.strip(".,;:()[]")] += 1.0
        norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
        return {token: value / norm for token, value in counts.items()}

    def _cosine(self, left: dict[str, float], right: dict[str, float]) -> float:
        keys = set(left) & set(right)
        if not keys:
            return 0.0
        return sum(left[key] * right[key] for key in keys)


__all__ = [
    "DiscourseSegmenterChunker",
    "GrobidSectionChunker",
    "LayoutAwareChunker",
    "GraphRAGChunker",
]
