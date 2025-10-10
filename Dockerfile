FROM python:3.12-slim AS base

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
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV HUGGINGFACE_HUB_CACHE=/models/gemma3-12b \
    TRANSFORMERS_CACHE=/models/gemma3-12b \
    DOCLING_VLM_MODEL_DIR=/models/gemma3-12b

RUN mkdir -p /models/gemma3-12b

COPY pyproject.toml poetry.lock* requirements.txt requirements-dev.txt ./

RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi

COPY . .

RUN pip install .

EXPOSE 8000

CMD ["uvicorn", "Medical_KG_rev.gateway.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
