#!/bin/bash
echo "=========================================="
echo "Starting MinerU + vLLM Demo Environment"
echo "=========================================="
echo ""

echo "1. Starting vLLM server..."
docker compose up -d vllm-server

echo ""
echo "2. Waiting for vLLM to be healthy (this may take 2-5 minutes)..."
echo "   (Press Ctrl+C if you want to skip waiting)"
echo ""

# Wait for health check
timeout 300 bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do 
    echo "   Still waiting for vLLM server to start..."
    sleep 10
done' && echo "   ✅ vLLM server is healthy!" || echo "   ⚠️  Timeout waiting for vLLM"

echo ""
echo "3. Checking container status..."
docker compose ps

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test integration:"
echo "     python scripts/test_mineru_vllm_integration.py"
echo ""
echo "  2. Run the demo:"
echo "     python download_and_process_random_papers.py"
echo ""
echo "To view logs:"
echo "  docker compose logs -f vllm-server"
echo ""
