"""Information extraction microservice using an LLM template approach."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import structlog

from ..gpu.manager import GpuManager

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class ExtractionSpan:
    label: str
    text: str
    start: int
    end: int
    confidence: float


@dataclass(slots=True)
class ExtractionResult:
    document_id: str
    kind: str
    spans: List[ExtractionSpan] = field(default_factory=list)
    raw_response: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractionInput:
    tenant_id: str
    document_id: str
    text: str
    kind: str


PicoSchema = {
    "population": ["population", "participants", "patients"],
    "intervention": ["intervention", "treatment"],
    "comparison": ["comparison", "control"],
    "outcome": ["outcome", "result"],
}


class _LLMClient:
    """Lightweight template-driven LLM stub for deterministic tests."""

    def __init__(self, gpu: GpuManager) -> None:
        self.gpu = gpu
        self._prompt_cache: Dict[str, str] = {}

    def generate(self, *, prompt: str, text: str) -> Dict[str, object]:
        cache_key = f"{prompt}:{hash(text)}"
        if cache_key in self._prompt_cache:
            return json.loads(self._prompt_cache[cache_key])
        with self.gpu.device_session("extraction", warmup=True):
            # Build a deterministic pseudo-response by matching heuristics.
            result: Dict[str, List[Dict[str, object]]] = {}
            lowered = text.lower()
            for label, keywords in PicoSchema.items():
                matches: List[Dict[str, object]] = []
                for keyword in keywords:
                    for match in re.finditer(keyword, lowered):
                        span = {
                            "text": text[match.start() : match.end()],
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.7,
                        }
                        matches.append(span)
                if matches:
                    result[label] = matches
            payload = json.dumps(result)
            self._prompt_cache[cache_key] = payload
            return json.loads(payload)


class ExtractionService:
    """Runs LLM extraction flows with span grounding validation."""

    def __init__(self, gpu: GpuManager) -> None:
        self.gpu = gpu
        self.llm = _LLMClient(gpu)

    def _validate_spans(self, text: str, spans: Iterable[Dict[str, object]]) -> List[ExtractionSpan]:
        validated: List[ExtractionSpan] = []
        for span in spans:
            start = int(span.get("start", -1))
            end = int(span.get("end", -1))
            if start < 0 or end <= start or end > len(text):
                logger.warning("extraction.span.invalid", span=span)
                continue
            snippet = text[start:end]
            if snippet != span.get("text"):
                logger.warning("extraction.span.mismatch", expected=snippet, actual=span.get("text"))
                continue
            validated.append(
                ExtractionSpan(
                    label=span.get("label", ""),
                    text=snippet,
                    start=start,
                    end=end,
                    confidence=float(span.get("confidence", 0.5)),
                )
            )
        return validated

    def extract(self, request: ExtractionInput) -> ExtractionResult:
        logger.info("extraction.run", document_id=request.document_id, kind=request.kind)
        template = {
            "pico": "Identify PICO elements with start/end offsets.",
            "adverse-event": "Extract adverse events and severities.",
        }.get(request.kind, "generic extraction")
        raw = self.llm.generate(prompt=template, text=request.text)

        spans: List[ExtractionSpan] = []
        for label, matches in raw.items():
            validated = self._validate_spans(
                request.text,
                ({**match, "label": label} for match in matches),  # type: ignore[arg-type]
            )
            spans.extend(validated)

        logger.info(
            "extraction.completed",
            document_id=request.document_id,
            kind=request.kind,
            spans=len(spans),
        )
        return ExtractionResult(
            document_id=request.document_id,
            kind=request.kind,
            spans=spans,
            raw_response=raw,
        )


class ExtractionGrpcService:
    """Async gRPC servicer bridging extraction results to protobuf responses."""

    def __init__(self, service: ExtractionService) -> None:
        self.service = service

    async def Extract(self, request, context):  # type: ignore[override]
        extraction_request = ExtractionInput(
            tenant_id=request.tenant_id,
            document_id=request.document_id,
            text=request.text,
            kind=request.kind,
        )
        result = self.service.extract(extraction_request)

        from Medical_KG_rev.proto.gen import extraction_pb2  # type: ignore import-error

        reply = extraction_pb2.ExtractResponse(
            document_id=result.document_id,
            kind=result.kind,
            raw_json=json.dumps(result.raw_response),
        )
        for span in result.spans:
            message = reply.spans.add()
            message.label = span.label
            message.text = span.text
            message.start = span.start
            message.end = span.end
            message.confidence = span.confidence
        return reply
