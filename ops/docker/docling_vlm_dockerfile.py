"""Utility helpers for maintaining the Docling VLM Dockerfile."""

from __future__ import annotations

import argparse
from pathlib import Path

DOCKERFILE_NAME = "Dockerfile.docling-vlm"
DOCKERFILE_RELATIVE = Path(__file__).with_name(DOCKERFILE_NAME)

DOCKERFILE_TEMPLATE = """# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MK_DOCLING_VLM_CACHE=/models/docling-vlm \
    HUGGINGFACE_HUB_CACHE=/models/docling-vlm \
    TRANSFORMERS_CACHE=/models/docling-vlm \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        git-lfs \
        curl \
        ffmpeg \
        libglib2.0-0 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install --system

RUN mkdir -p ${MK_DOCLING_VLM_CACHE}

COPY requirements.txt requirements-dev.txt pyproject.toml poetry.lock* ./

RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi && \
    pip install "docling[vlm]>=2.0.0" "docling-core>=2.0.0" "vllm>=0.11.0"

COPY . .

RUN pip install .

COPY ops/docker/docling_vlm_setup.py /docker-entrypoint.d/docling_vlm_setup.py
RUN python /docker-entrypoint.d/docling_vlm_setup.py --ensure-cache --warmup

EXPOSE 8000

ENTRYPOINT ["uvicorn", "Medical_KG_rev.services.parsing.docling_vlm_server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
"""


def write_dockerfile(path: Path | None = None) -> Path:
    """Write the canonical Docling VLM Dockerfile to disk."""
    destination = Path(path) if path else DOCKERFILE_RELATIVE
    destination.write_text(DOCKERFILE_TEMPLATE, encoding="utf-8")
    return destination


def check_dockerfile(path: Path | None = None) -> bool:
    """Return True when the on-disk Dockerfile matches the template."""
    destination = Path(path) if path else DOCKERFILE_RELATIVE
    if not destination.exists():
        return False
    return destination.read_text(encoding="utf-8") == DOCKERFILE_TEMPLATE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the Dockerfile matches the recorded template",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCKERFILE_RELATIVE,
        help="Optional output path when writing the Dockerfile",
    )
    args = parser.parse_args()

    if args.check:
        ok = check_dockerfile(args.output)
        if not ok:
            print(f"Dockerfile at {args.output} is out of date", flush=True)
            return 1
        print("Dockerfile is up to date", flush=True)
        return 0

    destination = write_dockerfile(args.output)
    print(f"Wrote Docling VLM Dockerfile to {destination}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
