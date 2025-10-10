# MinerU + vLLM Model Options

## The Issue

The original configuration used `Qwen/Qwen2.5-VL-7B-Instruct` (a vision-language model), but this requires:

- Very recent vLLM version with VL support
- Significant GPU memory (24GB+)
- Special configuration for multimodal processing

## Options

### Option 1: Use Text-Only Model (Recommended for Demo)

**Model**: `Qwen/Qwen2.5-7B-Instruct` (text-only, no VL)

**Pros**:

- ✅ Fully supported by vLLM v0.11.0
- ✅ Smaller memory footprint (~14GB)
- ✅ Faster inference
- ✅ Easier to run

**Cons**:

- ❌ Cannot process images directly
- ❌ MinerU won't have vision capabilities

**Configuration**:

```yaml
vllm-server:
  image: vllm/vllm-openai:v0.11.0
  command:
    - --model
    - Qwen/Qwen2.5-7B-Instruct  # Text-only
    - --gpu-memory-utilization
    - "0.85"
    - --max-model-len
    - "16384"
```

### Option 2: Use Simulated CLI (No GPU Required)

**For testing the pipeline without GPU**:

Just run the demo without starting vLLM. MinerU will automatically fall back to simulated mode.

**Pros**:

- ✅ No GPU required
- ✅ Fast for testing pipeline logic
- ✅ No Docker dependencies

**Cons**:

- ❌ Cannot process real PDFs
- ❌ Only works with text files
- ❌ Not production-ready

### Option 3: Use External API Service

**Use a hosted service** like OpenAI, Anthropic, or other providers:

Modify the VLLMClient to point to an external API:

```bash
export MK_MINERU__VLLM_SERVER__BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your-key-here"
```

**Pros**:

- ✅ No local GPU required
- ✅ Access to latest models
- ✅ Scalable

**Cons**:

- ❌ Costs money per request
- ❌ Data sent to external service
- ❌ Requires API key

### Option 4: Upgrade to Latest vLLM (Advanced)

Use a newer vLLM image with VL support:

```yaml
vllm-server:
  image: vllm/vllm-openai:latest  # or v0.9.x
  command:
    - --model
    - Qwen/Qwen2.5-VL-7B-Instruct
    - --trust-remote-code
```

**Pros**:

- ✅ Full vision-language capabilities
- ✅ Can process images

**Cons**:

- ❌ Requires testing/validation
- ❌ May have breaking changes
- ❌ Higher memory requirements

## Recommendation

**For the demo**, I've switched to **Option 1: Text-Only Model** (`Qwen/Qwen2.5-7B-Instruct`).

This will:

- ✅ Work immediately with your current setup
- ✅ Process PDFs successfully
- ✅ Demonstrate the full pipeline
- ✅ Extract text blocks, tables, structure

The text-only model is sufficient for demonstrating that:

- Docker integration works
- MinerU connects to vLLM
- PDFs are processed end-to-end
- Structured content is extracted

## Next Steps

1. **Try the text-only model first** (already configured)
2. **If you need VL capabilities**, consider Option 4 with latest vLLM
3. **For production**, evaluate based on your use case

## Current Configuration

I've updated `docker-compose.yml` to use the text-only model. To start:

```bash
# Remove old containers
docker compose down

# Start with new configuration
docker compose up -d vllm-server

# Wait 2-3 minutes for model download
docker compose logs -f vllm-server

# Run demo
python download_and_process_random_papers.py
```
