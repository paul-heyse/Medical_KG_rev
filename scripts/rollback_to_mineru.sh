#!/usr/bin/env bash
set -euo pipefail

# This script restores the MinerU processing stack when Docling VLM must be rolled back.
# It flips the PDF processing feature flag, reapplies MinerU Kubernetes manifests, and
# restarts dependent services. Use only during emergency response after verifying
# rollback conditions in docs/guides/rollback_procedures.md.

if [[ "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: scripts/rollback_to_mineru.sh [--dry-run]

Prerequisites:
  * kubectl configured for the active cluster
  * helm available for service restarts
  * access to secrets managing PDF processing backends

Steps performed:
  1. Toggle feature flag PDF_PROCESSING_BACKEND=mineru
  2. Apply archived MinerU manifests under ops/k8s/overlays/mineru-rollback
  3. Restart gateway, orchestration, and parsing deployments

Use --dry-run to see the actions without executing mutating commands.
USAGE
  exit 0
fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

run() {
  if $DRY_RUN; then
    echo "[dry-run] $*"
  else
    echo "[exec] $*"
    eval "$@"
  fi
}

# 1. Flip feature flag via config map/secret update
FEATURE_FLAG_SECRET=${FEATURE_FLAG_SECRET:-doc-processing-feature-flags}
NAMESPACE=${NAMESPACE:-doc-processing}
run "kubectl -n ${NAMESPACE} patch secret ${FEATURE_FLAG_SECRET} --type merge -p '{\"stringData\":{\"PDF_PROCESSING_BACKEND\":\"mineru\"}}'"

# 2. Apply archived MinerU manifests (kept for rollback only)
MANIFEST_PATH=${MANIFEST_PATH:-ops/k8s/overlays/mineru-rollback}
if [[ ! -d "$MANIFEST_PATH" ]]; then
  echo "MinerU rollback manifests missing at $MANIFEST_PATH" >&2
  exit 1
fi
run "kubectl apply -k ${MANIFEST_PATH}"

# 3. Restart dependent services to ensure MinerU containers are running
declare -a deployments=(gateway pdf-orchestrator mineru-parser)
for deployment in "${deployments[@]}"; do
  run "kubectl -n ${NAMESPACE} rollout restart deployment/${deployment}"
  run "kubectl -n ${NAMESPACE} rollout status deployment/${deployment} --timeout=180s"
done

echo "Rollback completed. Monitor alerts and ledger state before resuming ingestion."
