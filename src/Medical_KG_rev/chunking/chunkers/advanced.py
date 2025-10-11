"""Advanced chunkers that build on contextual chunking abstractions."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

from Medical_KG_rev.models.ir import Block, BlockType, Document

from ..base import ContextualChunker
from ..exceptions import ChunkerConfigurationError
from ..provenance import BlockContext
from ..segmentation import Segment
from ..tokenization import TokenCounter



@dataclass(slots=True)
class _Community:
    nodes: list[int]
    score: float


class DiscourseSegmenterChunker(ContextualChunker):
    """Chunker that approximates EDU segmentation using rhetorical connectives."""

    name = "discourse_segmenter"
    version = "v1"
    default_granularity = "paragraph"
    segment_type = "discourse"

    def __init__(
        self,
        *,
        connectives: Sequence[str] | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        super().__init__(token_counter=token_counter)
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

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        sequence = list(contexts)
        if not sequence:
            return []
        boundaries = self._detect_boundaries(sequence)
        segments: list[Segment] = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=False):
            window = sequence[start:end]
            if not window:
                continue
            segments.append(
                Segment(
                    contexts=list(window),
                    metadata={"connectives": list(self.connectives[:5])},
                )
            )
        return segments

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


class GrobidSectionChunker(ContextualChunker):
    """Chunker that aligns Docling VLM output with Grobid TEI XML sections."""

    name = "grobid_section"
    version = "v1"
    default_granularity = "section"
    segment_type = "grobid"
    include_tables = True

    def __init__(self, *, token_counter: TokenCounter | None = None) -> None:
        super().__init__(token_counter=token_counter)

    def segment_document(
        self,
        document: Document,
        contexts: Sequence[BlockContext],
        *,
        blocks: Sequence[Block] | None = None,
    ) -> Iterable[Segment]:
        metadata = document.metadata or {}
        tei_xml = str(metadata.get("tei_xml", ""))
        if not tei_xml.strip():
            raise ChunkerConfigurationError("Document does not provide TEI XML metadata")
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as exc:  # pragma: no cover - invalid TEI is rare
            raise ChunkerConfigurationError("Invalid TEI XML payload") from exc
        section_map = self._tei_section_titles(root)
        grouped: dict[str, list[BlockContext]] = defaultdict(list)
        for ctx in contexts:
            key = ctx.section_title.lower()
            normalized = section_map.get(key, key)
            grouped[normalized].append(ctx)
        segments: list[Segment] = []
        for title, bucket in grouped.items():
            if not bucket:
                continue
            segments.append(
                Segment(
                    contexts=list(bucket),
                    metadata={"tei_title": title},
                )
            )
        return segments

    def segment_contexts(self, contexts: Sequence[BlockContext]) -> Iterable[Segment]:
        if not contexts:
            return []
        return [Segment(contexts=list(contexts))]

    def explain(self) -> dict[str, object]:
        return {"strategy": "grobid-tei-alignment"}

    def _tei_section_titles(self, root: ET.Element) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for div in root.findall(".//{*}div"):
            type_attr = div.attrib.get("type") or "section"
            head = div.findtext("{*}head") or type_attr
            mapping[head.lower()] = head
        return mapping


class LayoutAwareChunker(ContextualChunker):
    """Chunker that groups blocks using layout metadata from DocTR/Docling."""

    name = "layout_aware"
    version = "v1"
    default_granularity = "section"
    segment_type = "layout"
    include_tables = True

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        overlap_threshold: float = 0.3,
    ) -> None:
        super().__init__(token_counter=token_counter)
        self.overlap_threshold = overlap_threshold

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        grouped = self._group_by_layout(list(contexts))
        segments: list[Segment] = []
        for region_id, bucket in sorted(grouped.items()):
            if not bucket:
                continue
            metadata = {
                "region_id": region_id,
                "region_count": len(bucket),
                "overlap_threshold": self.overlap_threshold,
            }
            segments.append(Segment(contexts=list(bucket), metadata=metadata))
        return segments

    def explain(self) -> dict[str, object]:
        return {"overlap_threshold": self.overlap_threshold}

    def _group_by_layout(self, contexts: Sequence[BlockContext]) -> dict[str, list[BlockContext]]:
        buckets: dict[str, list[BlockContext]] = defaultdict(list)
        for ctx in contexts:
            metadata = ctx.block.metadata or {}
            overlap = float(metadata.get("layout_overlap", 1.0))
            if overlap < self.overlap_threshold:
                continue
            region_id = str(metadata.get("layout_region") or ctx.section.id)
            buckets[region_id].append(ctx)
        return buckets


class GraphRAGChunker(ContextualChunker):
    """Chunker that builds a lexical graph and emits community based chunks."""

    name = "graph_rag"
    version = "v1"
    default_granularity = "section"
    segment_type = "graph"

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.18,
        token_counter: TokenCounter | None = None,
    ) -> None:
        super().__init__(token_counter=token_counter)
        self.similarity_threshold = similarity_threshold

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        sequence = [ctx for ctx in contexts if ctx.text]
        if not sequence:
            return []
        graph = self._build_graph(sequence)
        communities = self._community_detection(graph)
        segments: list[Segment] = []
        for community in communities:
            window = [sequence[index] for index in community.nodes if 0 <= index < len(sequence)]
            if not window:
                continue
            summary = window[0].text.split(".", 1)[0]
            metadata = {
                "community_score": round(community.score, 3),
                "nodes": len(window),
                "summary": summary,
            }
            segments.append(Segment(contexts=list(window), metadata=metadata))
        return segments

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
    "GraphRAGChunker",
    "GrobidSectionChunker",
    "LayoutAwareChunker",
]
