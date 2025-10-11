#!/bin/bash
# Stop torch isolation architecture services

set -e

echo "Stopping Medical KG Torch Isolation Architecture..."

# Stop all services
docker-compose -f ops/docker/docker-compose.torch-isolation.yml down

echo "All services stopped successfully!"
echo ""
echo "To remove volumes (WARNING: This will delete all data):"
echo "docker-compose -f ops/docker/docker-compose.torch-isolation.yml down -v"
