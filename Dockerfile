ARG CUDA_IMAGE_TAG=12.1.1-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_IMAGE_TAG} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git-lfs \
        libgl1 \
        libglib2.0-0 \
        software-properties-common && \
    git lfs install --system && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv && \
    python3.12 -m ensurepip && \
    python3.12 -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

RUN mkdir -p /models/gemma3-12b/cache
ENV DOCLING_VLM_MODEL_PATH=/models/gemma3-12b \
    HUGGINGFACE_HUB_CACHE=/models/gemma3-12b/cache \
    TRANSFORMERS_CACHE=/models/gemma3-12b/cache \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"

COPY pyproject.toml poetry.lock* requirements.txt requirements-dev.txt ./

RUN python3.12 -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then python3.12 -m pip install -r requirements.txt; fi && \
    if [ -f requirements-dev.txt ]; then python3.12 -m pip install -r requirements-dev.txt; fi

COPY . .

RUN python3.12 -m pip install .

EXPOSE 8000

CMD ["python3.12", "-m", "uvicorn", "Medical_KG_rev.gateway.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
