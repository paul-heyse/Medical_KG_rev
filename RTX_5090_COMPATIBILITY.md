# RTX 5090 Compatibility Issue

## The Problem

Your system has an **NVIDIA GeForce RTX 5090**, which uses the Blackwell architecture with CUDA capability `sm_120`.

The vLLM Docker image (v0.11.0) was built with PyTorch that only supports up to `sm_90` (RTX 4090, H100), so it **cannot** run on your GPU.

Error:

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

## Solutions

### Option 1: Use Latest vLLM (Recommended to Try)

The latest vLLM might have RTX 5090 support:

```bash
# I've updated docker-compose.yml to use vllm/vllm-openai:latest

# Try it:
docker compose down
docker compose up -d vllm-server
docker compose logs -f vllm-server
```

If you see the same error, proceed to Option 2 or 3.

### Option 2: Use External API (OpenAI, etc.)

Use a hosted API service instead of local GPU:

```bash
# Set environment variables
export MK_MINERU__VLLM_SERVER__BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your-api-key-here"

# Run the demo
python download_and_process_random_papers.py
```

**Pros:**

- ✅ Works immediately
- ✅ No GPU required
- ✅ Access to latest models

**Cons:**

- ❌ Costs money per request
- ❌ Data sent externally

### Option 3: Use Simulated Mode (For Testing)

Run without GPU to test the pipeline:

```bash
# Just run - it automatically falls back to simulation
python download_and_process_random_papers.py --samples 5
```

**Pros:**

- ✅ Tests pipeline logic
- ✅ No GPU/Docker needed
- ✅ Free

**Cons:**

- ❌ Cannot process real PDFs
- ❌ Not production-ready

### Option 4: Build vLLM with RTX 5090 Support

Build vLLM from source with CUDA 12.6+ and PyTorch 2.6+:

```bash
# This requires:
# - CUDA 12.6 or newer
# - PyTorch 2.6 with sm_120 support
# - Building from vLLM source

# See: https://docs.vllm.ai/en/latest/getting_started/installation.html
```

This is advanced and time-consuming.

### Option 5: Use a Different GPU

If you have access to:

- RTX 4090 (sm_89)
- RTX 4080 (sm_89)
- A100/H100 (sm_80/sm_90)
- Any older NVIDIA GPU

The current vLLM image will work fine.

## Current Status

I've updated `docker-compose.yml` to use `vllm/vllm-openai:latest` which might support RTX 5090.

Try running:

```bash
docker compose up -d vllm-server
docker compose logs -f vllm-server
```

If it still fails with the same CUDA capability error, I recommend **Option 2 (External API)** for immediate results, or **Option 3 (Simulated Mode)** for free testing.

## Why This Happened

The RTX 5090 just launched (2025), and Docker images take time to rebuild with newer PyTorch versions. This is a temporary compatibility gap that will be resolved as vLLM releases new images.

## Recommendation

**For demonstration purposes:**

1. Try latest vLLM first (already configured)
2. If that fails, use simulated mode: `python download_and_process_random_papers.py`

**For production:**

1. Use external API service (OpenAI, Anthropic, etc.)
2. Or wait for vLLM to release RTX 5090-compatible images
3. Or build vLLM from source with CUDA 12.6+
