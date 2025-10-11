"""FastAPI application shim for the Docling VLM service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMResult


def create_app() -> FastAPI:
    """Return a minimal FastAPI app that exposes a stub endpoint."""

    app = FastAPI(title="Docling VLM (stub)")

    @app.post("/v1/process", response_model=DoclingVLMResult)
    async def process_pdf() -> DoclingVLMResult:  # pragma: no cover - simple stub
        raise HTTPException(status_code=501, detail="Docling VLM service is not available")

    return app


if __name__ == "__main__":  # pragma: no cover - local debugging helper
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
