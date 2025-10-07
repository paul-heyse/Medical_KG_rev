#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: scripts/setup_mineru.sh [--models-dir DIR]

Downloads MinerU models and primes the local cache so that the CLI can run
without hitting the network during service start-up. The script is idempotent
and may be executed multiple times.

Environment variables:
  MINERU_MODELS_DIR   Override the location where MinerU stores model weights.
  MINERU_CLI_COMMAND  Custom MinerU CLI command (defaults to `mineru`).
USAGE
  exit 0
fi

MODELS_DIR="${MINERU_MODELS_DIR:-${1:-$HOME/.cache/mineru/models}}"
CLI_COMMAND="${MINERU_CLI_COMMAND:-mineru}"

mkdir -p "${MODELS_DIR}"

echo "[mineru] priming model cache in ${MODELS_DIR}" >&2

if ! command -v "${CLI_COMMAND}" >/dev/null 2>&1; then
  echo "[mineru] CLI '${CLI_COMMAND}' not found in PATH" >&2
  exit 1
fi

"${CLI_COMMAND}" --version >/dev/null 2>&1 || {
  echo "[mineru] unable to execute '${CLI_COMMAND} --version'" >&2
  exit 1
}

# MinerU lazily downloads models during the first CLI invocation. We trigger a
# dry run with an empty document so that required weights are cached.
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

touch "${TMP_DIR}/empty.pdf"

"${CLI_COMMAND}" parse \
  --input "${TMP_DIR}" \
  --output "${TMP_DIR}/out" \
  --format json \
  --vram-limit 1G \
  --models "${MODELS_DIR}" \
  >/dev/null 2>&1 || true

echo "[mineru] setup complete"
