#!/bin/bash
set -euo pipefail

VLLM_URL="${VLLM_SERVER_URL:-http://localhost:8000}"

# Test health endpoint
curl -f "${VLLM_URL}/health" >/dev/null

echo "vLLM health endpoint responded successfully"

# Test OpenAI-compatible chat completion
curl -sS -X POST "${VLLM_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "temperature": 0.0
  }' | jq '.choices[0].message.content'
