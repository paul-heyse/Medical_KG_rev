# How to Run the MinerU + vLLM Demo

## ⚠️ Important: Start Docker Services First

The demo requires Docker services to be running. Without them, MinerU falls back to simulation mode and cannot process real PDFs.

## Step-by-Step Instructions

### 1. Start vLLM Server

```bash
# Start the vLLM server (this will take 2-5 minutes to load the model)
docker compose up -d vllm-server

# Watch the logs to see when it's ready
docker compose logs -f vllm-server
```

**Wait for this message:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Press `Ctrl+C` to exit the logs.

### 2. Verify vLLM is Healthy

```bash
# Check the health endpoint
curl http://localhost:8000/health

# Should return: OK or similar response

# Or check container status
docker compose ps
# Should show vllm-server as "running (healthy)"
```

### 3. Run the Integration Test (Optional but Recommended)

```bash
# This validates the setup before processing PDFs
python scripts/test_mineru_vllm_integration.py
```

**Expected output:**

```
✓ Settings loaded successfully
✓ vLLM server is reachable
✓ vLLM server reports healthy status
✓ VLLMClient initialized successfully
✓ All critical tests passed!
```

### 4. Run the Demo

```bash
# Process 10 papers (default)
python download_and_process_random_papers.py

# Or specify how many
python download_and_process_random_papers.py --samples 20
```

## What You Should See

### Successful Output

```
======================================================================
END-TO-END PDF PROCESSING WITH MINERU + VLLM DOCKER
======================================================================

🏥 STEP 0: Validating vLLM Server Health
----------------------------------------------------------------------
🔍 Checking vLLM server health at http://localhost:8000...
✅ vLLM server is healthy and ready

✅ vLLM server is ready for PDF processing

📚 STEP 1: Fetching Papers from OpenAlex
----------------------------------------------------------------------
✅ Fetched 10 papers

📥 STEP 3: Downloading PDFs
----------------------------------------------------------------------
✅ Successfully downloaded 7 PDFs

⚙️  STEP 4: Processing PDFs with MinerU + vLLM
----------------------------------------------------------------------
[1/7]
🔄 Processing PDF with MinerU + vLLM: W2162544110.pdf
   📄 Title: Developing and evaluating complex...
   📊 PDF size: 0.18 MB
   ⚙️  Sending to MinerU processor...
   ✅ Processing complete in 4.52s
   📊 Extracted: 127 blocks, 3 tables, 5 figures
   💾 Results saved to: W2162544110_processed.json

...

🎉 SUCCESS! End-to-end MinerU + vLLM Docker integration verified!
   ✅ Downloaded PDFs from OpenAlex
   ✅ Processed PDFs with MinerU
   ✅ MinerU connected to vLLM server
   ✅ vLLM provided GPU-accelerated inference
   ✅ Extracted structured content (blocks, tables, etc.)
```

## Common Issues

### Issue: "Cannot connect to vLLM server"

```
❌ Cannot connect to vLLM server: Connection refused
```

**Solution:**

```bash
# Check if vLLM is running
docker compose ps

# If not running, start it
docker compose up -d vllm-server

# Wait for it to be healthy (2-5 minutes)
docker compose logs -f vllm-server
```

### Issue: "Simulated CLI expects UTF-8 encoded content"

This means Docker services aren't running. Follow Step 1 above.

### Issue: vLLM Server Not Starting

```bash
# Check for errors in logs
docker compose logs vllm-server

# Common causes:
# 1. No GPU available
nvidia-smi  # Verify GPU is accessible

# 2. Insufficient GPU memory (need 24GB+)
# 3. Model download failed (check internet connection)
```

### Issue: Model Download is Slow

The first time you run vLLM, it needs to download the Qwen2.5-VL-7B-Instruct model (10+ GB).

```bash
# Check download progress
docker compose logs vllm-server | grep -i download

# This is normal and only happens once
# The model is cached in ~/.cache/huggingface
```

## Stopping the Services

```bash
# Stop all services
docker compose down

# Or just stop vLLM
docker compose stop vllm-server

# Stop and remove everything (including cached data)
docker compose down -v
```

## Quick Reference

```bash
# Start services
docker compose up -d vllm-server

# Check status
docker compose ps
curl http://localhost:8000/health

# Run demo
python download_and_process_random_papers.py

# View logs
docker compose logs -f vllm-server

# Stop services
docker compose down
```

## Need Help?

1. **Validate setup**: `./scripts/validate_docker_setup.sh`
2. **Test integration**: `python scripts/test_mineru_vllm_integration.py`
3. **Check logs**: `docker compose logs vllm-server`
4. **Review docs**: `docs/devops/mineru-vllm-docker-setup.md`

---

**Remember**: Always start Docker services BEFORE running the demo! 🚀
