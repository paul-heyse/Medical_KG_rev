# GPU Microservices Deployment Guide

This document outlines how to deploy the MinerU, Embedding, and Extraction GPU microservices.

## Prerequisites

- NVIDIA GPU with supported CUDA drivers (12.1+)
- Docker 24+ with the NVIDIA Container Toolkit configured
- Access to the project container registry

## Building the Image

```bash
docker build -f ops/Dockerfile.gpu -t medicalkg/gpu-services:latest .
```

## Running Locally

```bash
docker run --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 7000:7000 -p 7001:7001 -p 7002:7002 \
  medicalkg/gpu-services:latest
```

The container exposes the following gRPC endpoints:

- `7000`: MinerU PDF processing
- `7001`: Embedding generation
- `7002`: LLM extraction

## Health and Readiness

The container health check executes `python -m Medical_KG_rev.scripts.healthcheck --service gpu`.
This fails fast when CUDA devices are unavailable or misconfigured.

Each gRPC service registers a [gRPC Health Checking Protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md) endpoint.
Clients should poll the health service before submitting work.

## Observability

- Prometheus metrics exported via the default registry (`gpu_service_utilization_percent`, `gpu_service_memory_megabytes`)
- OpenTelemetry spans emitted for each RPC via `UnaryUnaryTracingInterceptor`

## Shutdown

The launcher script (`Medical_KG_rev.scripts.serve_gpu`) listens for `SIGTERM`/`SIGINT` and
performs graceful shutdown of all running gRPC servers.
