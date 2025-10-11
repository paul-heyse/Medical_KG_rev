"""FastAPI server exposing the Docling VLM service over HTTP."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from Medical_KG_rev.config.settings import get_settings

from .docling_vlm_service import DoclingVLMResult, DoclingVLMService


class ProcessRequest(BaseModel):
    document_id: str
    content_base64: str | None = None
    pdf_path: str | None = None

    def ensure_pdf(self) -> Path:
        if self.pdf_path:
            path = Path(self.pdf_path)
            if not path.exists():
                raise HTTPException(status_code=400, detail="PDF path does not exist")
            return path
        if not self.content_base64:
            raise HTTPException(
                status_code=400, detail="Either pdf_path or content_base64 must be supplied"
            )
        try:
            payload = base64.b64decode(self.content_base64)
        except Exception as exc:  # pragma: no cover - FastAPI validates base64
            raise HTTPException(status_code=400, detail="Invalid base64 payload") from exc
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp.write(payload)
        temp.flush()
        temp.close()
        return Path(temp.name)


class ProcessResponse(BaseModel):
    document_id: str
    text: str
    tables: list[dict[str, Any]]
    figures: list[dict[str, Any]]
    metadata: dict[str, Any]

    @classmethod
    def from_result(cls, result: DoclingVLMResult) -> ProcessResponse:
        return cls(
            document_id=result.document_id,
            text=result.text,
            tables=result.tables,
            figures=result.figures,
            metadata=result.metadata,
        )


def create_app() -> FastAPI:
    settings = get_settings()
    config = settings.docling_vlm.as_config()
    service = DoclingVLMService(config=config, eager=True)
    app = FastAPI(title="Docling VLM Service", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.post("/process", response_model=ProcessResponse)
    def process(request: ProcessRequest) -> ProcessResponse:
        pdf_path = request.ensure_pdf()
        try:
            result = service.process_pdf(str(pdf_path), document_id=request.document_id)
        finally:
            if not request.pdf_path:
                pdf_path.unlink(missing_ok=True)
        return ProcessResponse.from_result(result)

    @app.post("/warmup")
    def warmup() -> dict[str, Any]:
        # Re-create the service to trigger warmup to reuse built-in logic.
        warmed = DoclingVLMService(config=config, eager=True)
        return warmed.health()

    return app


def main() -> None:  # pragma: no cover - manual invocation helper
    import uvicorn

    uvicorn.run(
        "Medical_KG_rev.services.parsing.docling_vlm_server:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
