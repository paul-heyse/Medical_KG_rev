#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for local services to become ready..."
timeout=120
interval=5
elapsed=0

check_endpoint() {
  local url=$1
  if curl -sSf "$url" >/dev/null; then
    return 0
  fi
  return 1
}

while (( elapsed < timeout )); do
  if check_endpoint "http://localhost:8000/openapi.json"; then
    echo "Gateway is reachable"
    exit 0
  fi
  sleep "$interval"
  elapsed=$((elapsed + interval))
done

echo "Services did not become ready in time" >&2
exit 1
