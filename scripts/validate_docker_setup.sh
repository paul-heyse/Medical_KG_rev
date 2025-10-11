#!/bin/bash
# validate_docker_setup.sh - Validate Docker setup for MinerU + vLLM integration

set -euo pipefail

echo "=========================================="
echo "Docker Setup Validation for MinerU + vLLM"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
ERRORS=0
WARNINGS=0

# Helper functions
error() {
    echo -e "${RED}✗ $1${NC}"
    ((ERRORS++))
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((WARNINGS++))
}

info() {
    echo "ℹ $1"
}

# Check 1: Docker installed
echo "1. Checking Docker installation..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d ' ' -f3 | cut -d ',' -f1)
    success "Docker is installed (version: $DOCKER_VERSION)"
else
    error "Docker is not installed"
    info "Install Docker from: https://docs.docker.com/get-docker/"
fi

# Check 2: Docker Compose installed
echo ""
echo "2. Checking Docker Compose installation..."
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short)
    success "Docker Compose is installed (version: $COMPOSE_VERSION)"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version | cut -d ' ' -f3 | cut -d ',' -f1)
    success "Docker Compose is installed (version: $COMPOSE_VERSION)"
    info "Consider upgrading to Docker Compose V2 (docker compose)"
else
    error "Docker Compose is not installed"
    info "Install Docker Compose from: https://docs.docker.com/compose/install/"
fi

# Check 3: NVIDIA Docker runtime (for GPU support)
echo ""
echo "3. Checking NVIDIA Docker runtime..."
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    success "NVIDIA Docker runtime is working"
else
    warning "NVIDIA Docker runtime test failed"
    info "GPU support may not be available"
    info "Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Check 4: Required files exist
echo ""
echo "4. Checking required configuration files..."
FILES=(
    "docker-compose.yml"
    "ops/docker/Dockerfile.mineru-worker"
    "config/mineru.yaml"
    "requirements.txt"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        success "Found: $file"
    else
        error "Missing: $file"
    fi
done

# Check 5: Docker Compose file syntax
echo ""
echo "5. Validating docker-compose.yml syntax..."
if docker compose -f docker-compose.yml config &> /dev/null; then
    success "docker-compose.yml syntax is valid"
else
    error "docker-compose.yml has syntax errors"
    info "Run: docker compose config"
fi

# Check 6: Check for existing containers
echo ""
echo "6. Checking for existing containers..."
if docker ps -a | grep -q "vllm-server\|mineru-worker"; then
    warning "Found existing MinerU or vLLM containers"
    info "You may want to stop/remove them before starting fresh:"
    info "  docker compose down"
fi

# Check 7: Check available disk space
echo ""
echo "7. Checking available disk space..."
AVAILABLE_GB=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -gt 50 ]; then
    success "Sufficient disk space available (${AVAILABLE_GB}GB)"
elif [ "$AVAILABLE_GB" -gt 20 ]; then
    warning "Limited disk space (${AVAILABLE_GB}GB) - may need more for models"
else
    error "Insufficient disk space (${AVAILABLE_GB}GB) - need at least 20GB"
fi

# Check 8: Verify HuggingFace cache directory
echo ""
echo "8. Checking HuggingFace cache directory..."
HF_CACHE="${HOME}/.cache/huggingface"
if [ -d "$HF_CACHE" ]; then
    success "HuggingFace cache directory exists: $HF_CACHE"
    CACHE_SIZE=$(du -sh "$HF_CACHE" 2>/dev/null | cut -f1)
    info "Cache size: $CACHE_SIZE"
else
    warning "HuggingFace cache directory does not exist"
    info "It will be created automatically when models are downloaded"
fi

# Check 9: Network configuration
echo ""
echo "9. Checking Docker network configuration..."
if docker network ls | grep -q "medical-kg-network"; then
    info "Network 'medical-kg-network' already exists"
    success "Network configuration looks good"
else
    info "Network 'medical-kg-network' will be created by docker-compose"
    success "Network configuration is valid"
fi

# Check 10: Python environment (if running locally)
echo ""
echo "10. Checking Python environment (optional for testing)..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d ' ' -f2)
    success "Python is installed (version: $PYTHON_VERSION)"

    if python3 -c "import httpx" 2>/dev/null; then
        success "httpx package is available for testing"
    else
        warning "httpx package not found (needed for test script)"
        info "Install: pip install httpx"
    fi
else
    warning "Python3 not found (optional for running test scripts)"
fi

# Summary
echo ""
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    success "All checks passed! Ready to start Docker services."
    echo ""
    info "Next steps:"
    echo "  1. Start services: docker compose up -d"
    echo "  2. Check logs: docker compose logs -f vllm-server mineru-worker"
    echo "  3. Test integration: python scripts/test_mineru_vllm_integration.py"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    warning "All critical checks passed with $WARNINGS warning(s)"
    echo ""
    info "You can proceed, but review warnings above"
    info "Next steps:"
    echo "  1. Start services: docker compose up -d"
    echo "  2. Check logs: docker compose logs -f vllm-server mineru-worker"
    exit 0
else
    error "Validation failed with $ERRORS error(s) and $WARNINGS warning(s)"
    echo ""
    info "Please fix the errors above before starting services"
    exit 1
fi
