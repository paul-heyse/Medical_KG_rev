# Use a CUDA-enabled base image so Docling and vLLM can utilise the GPU
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libglib2.0-0 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

ENV MK_DOCLING_VLM_CACHE=/models/docling-vlm \
    HUGGINGFACE_HUB_CACHE=/models/docling-vlm \
    TRANSFORMERS_CACHE=/models/docling-vlm \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN mkdir -p /models/docling-vlm

COPY pyproject.toml poetry.lock* requirements.txt requirements-dev.txt ./

RUN pip install --upgrade pip && \
    pip install --upgrade "vllm>=0.11.0" && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi && \
    pip install "docling[vlm]>=2.0.0" "docling-core>=2.0.0"

COPY . .

RUN pip install .

EXPOSE 8000

CMD ["uvicorn", "Medical_KG_rev.gateway.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
