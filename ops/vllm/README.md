# vLLM Model Images

This directory stores the Docker build definitions for every vLLM-backed
GPU embedding model that the platform supports. Each model receives its
own Dockerfile so that we can build, version, and publish images in
parallel without having to edit a shared template.

## Layout

- `catalog.json` – machine-readable catalog listing the models, their
  canonical Dockerfile names, and the published image tags.
- `Dockerfile.<model>` – per-model Dockerfile that copies the materialised
  weights from `models/<model>/` into the container image and starts the
  vLLM OpenAI-compatible server with appropriate defaults.

To add a new vLLM-capable model:

1. Materialise the model weights under `models/<model-name>/`.
2. Copy `Dockerfile.qwen3-embedding-8b` as a starting point and adjust
   the model metadata or runtime flags.
3. Append an entry to `catalog.json` with the Dockerfile name and the
   published container tag.
4. Update deployment manifests (`docker-compose.yml`, Kubernetes
   kustomizations, etc.) to reference the new image.

All GPU-serving embeddings **must** ship as Docker images in this
directory. We do not install `vllm` via `pip` anywhere in the runtime
codebase.
