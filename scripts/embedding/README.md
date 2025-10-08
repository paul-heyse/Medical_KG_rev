# Embedding Utility Scripts

This directory contains helper utilities for provisioning the GPU-only
embedding stack described in the `add-embeddings-representation`
OpenSpec change. The scripts are safe to run on developer workstations
and CI hosts; they surface actionable error messages when optional
dependencies such as `huggingface_hub` or `pyserini` are unavailable.

## Available scripts

- `download_models.py` – orchestrates downloading Qwen3 dense and SPLADE
  sparse models, mirroring the runbooks listed in the change tasks.
- `setup_environment.sh` – convenience wrapper that installs production
  dependencies with `pip` and then runs the environment verifier.  This script
  is idempotent and safe to run repeatedly in CI or provisioning pipelines.
- `verify_environment.py` – checks for GPU availability, validates the
  vLLM Docker catalog under `ops/vllm/`, and surfaces dependency
  versions before starting the embedding microservice.

Run the scripts with `python -m scripts.embedding.<name>`.
