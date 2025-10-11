#!/bin/bash
# Start torch isolation architecture services

set -e

echo "Starting Medical KG Torch Isolation Architecture..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q "nvidia"; then
    echo "Warning: NVIDIA Docker runtime not detected. GPU services may not work properly."
    echo "Please install nvidia-docker2 and restart Docker."
fi

# Create network if it doesn't exist
docker network create medical-kg-network 2>/dev/null || true

# Start infrastructure services first
echo "Starting infrastructure services..."
docker-compose -f ops/docker/docker-compose.torch-isolation.yml up -d neo4j opensearch kafka zookeeper redis prometheus grafana

# Wait for infrastructure to be ready
echo "Waiting for infrastructure services to be ready..."
sleep 30

# Start GPU services
echo "Starting GPU services..."
docker-compose -f ops/docker/docker-compose.torch-isolation.yml up -d gpu-services embedding-services reranking-services

# Wait for GPU services to be ready
echo "Waiting for GPU services to be ready..."
sleep 60

# Start main gateway
echo "Starting main gateway..."
docker-compose -f ops/docker/docker-compose.torch-isolation.yml up -d gateway

# Wait for gateway to be ready
echo "Waiting for gateway to be ready..."
sleep 30

# Check service health
echo "Checking service health..."
docker-compose -f ops/docker/docker-compose.torch-isolation.yml ps

echo ""
echo "Torch Isolation Architecture started successfully!"
echo ""
echo "Services:"
echo "  - Main Gateway: http://localhost:8000"
echo "  - Neo4j: http://localhost:7474"
echo "  - OpenSearch: http://localhost:9200"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000"
echo ""
echo "GPU Services:"
echo "  - GPU Services: gpu-services:50051"
echo "  - Embedding Services: embedding-services:50051"
echo "  - Reranking Services: reranking-services:50051"
echo ""
echo "To view logs: docker-compose -f ops/docker/docker-compose.torch-isolation.yml logs -f"
echo "To stop: docker-compose -f ops/docker/docker-compose.torch-isolation.yml down"
