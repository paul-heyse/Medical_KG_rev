#!/bin/bash
# Health check script for torch isolation architecture

set -e

echo "Checking Medical KG Torch Isolation Architecture Health..."
echo ""

# Function to check HTTP endpoint
check_http() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}

    echo -n "Checking $name... "
    if response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null); then
        if [ "$response" = "$expected_status" ]; then
            echo "✓ OK ($response)"
        else
            echo "✗ FAILED ($response, expected $expected_status)"
        fi
    else
        echo "✗ FAILED (connection error)"
    fi
}

# Function to check gRPC endpoint
check_grpc() {
    local name=$1
    local host=$2
    local port=$3

    echo -n "Checking $name... "
    if python3 -c "
import grpc
from grpc_health.v1 import health_pb2_grpc, health_pb2
try:
    channel = grpc.insecure_channel('$host:$port')
    stub = health_pb2_grpc.HealthStub(channel)
    response = stub.Check(health_pb2.HealthCheckRequest(service=''))
    if response.status == health_pb2.HealthCheckResponse.SERVING:
        print('✓ OK (SERVING)')
    else:
        print('✗ FAILED (status:', response.status, ')')
except Exception as e:
    print('✗ FAILED (', str(e), ')')
" 2>/dev/null; then
        true
    else
        echo "✗ FAILED (grpc check error)"
    fi
}

# Check infrastructure services
echo "Infrastructure Services:"
check_http "Neo4j" "http://localhost:7474"
check_http "OpenSearch" "http://localhost:9200"
check_http "Prometheus" "http://localhost:9090"
check_http "Grafana" "http://localhost:3000"

echo ""
echo "GPU Services:"
check_grpc "GPU Services" "localhost" "50051"
check_grpc "Embedding Services" "localhost" "50051"
check_grpc "Reranking Services" "localhost" "50051"

echo ""
echo "Main Gateway:"
check_http "Gateway Health" "http://localhost:8000/health"

echo ""
echo "Health check complete!"
echo ""
echo "To view detailed logs:"
echo "docker-compose -f ops/docker/docker-compose.torch-isolation.yml logs -f"
