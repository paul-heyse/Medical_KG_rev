#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[rollback][%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$1"
}

log "=== ROLLBACK: Standardized Embeddings ==="

log "Scaling down new embedding services"
kubectl scale deployment/vllm-embedding --replicas=0 || true
kubectl scale deployment/pyserini-splade --replicas=0 || true

log "Re-enabling legacy embedding deployment"
kubectl scale deployment/legacy-embedding --replicas=3 || true

if git rev-parse --verify HEAD >/dev/null 2>&1; then
  log "Reverting embedding commits"
  git revert --no-edit "$1"
fi

log "Undoing embedding-service rollout"
kubectl rollout undo deployment/embedding-service || true

log "Restoring OpenSearch mappings"
curl -fsSL -X PUT "http://opensearch:9200/chunks/_mapping" -H 'Content-Type: application/json' -d @legacy-mapping.json || true

log "=== ROLLBACK COMPLETE ==="
log "Target RTO: 15 minutes"
