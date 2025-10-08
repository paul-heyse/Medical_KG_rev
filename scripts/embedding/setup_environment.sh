#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="${ROOT_DIR}/requirements.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "requirements.txt not found at ${REQ_FILE}" >&2
  exit 1
fi

echo "[setup] Installing production dependencies"
python -m pip install --upgrade pip
python -m pip install -r "${REQ_FILE}"

echo "[setup] Verifying GPU environment"
python -m scripts.embedding.verify_environment
