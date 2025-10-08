#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install_chunking_dependencies.sh [options]

Install the optional chunking/parsing dependencies inside a dedicated virtual
environment and verify that the required resources (including the configured
Hugging Face tokenizer) are available. This script is intended to simplify
deployment of the new chunking stack in fresh environments.

Options:
  --venv PATH         Target virtual environment directory (default: .venv-chunking)
  --check-only        Only run the dependency check (requires an existing venv)
  --full-requirements Install all packages from requirements.txt instead of the
                       minimal chunking set.
  --hf-model NAME     Hugging Face model/tokenizer to download (defaults to
                       MEDICAL_KG_SENTENCE_MODEL when set).
  -h, --help          Show this message.
EOF
}

VENV_DIR=".venv-chunking"
CHECK_ONLY=0
FULL_REQUIREMENTS=0
HF_MODEL="${MEDICAL_KG_SENTENCE_MODEL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --full-requirements)
      FULL_REQUIREMENTS=1
      shift
      ;;
    --hf-model)
      HF_MODEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ${CHECK_ONLY} -eq 0 ]]; then
  echo "[chunking] Creating virtual environment at ${VENV_DIR}" >&2
  python -m venv "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found at ${VENV_DIR}." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

PYTHON_VERSION="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
PYTHON_MINOR="${PYTHON_VERSION##*.}"

if [[ ${CHECK_ONLY} -eq 0 ]]; then
  echo "[chunking] Upgrading pip" >&2
  python -m pip install --upgrade pip setuptools wheel

  if [[ ${FULL_REQUIREMENTS} -eq 1 ]]; then
    echo "[chunking] Installing project requirements" >&2
    if ! python -m pip install -r requirements.txt; then
      echo "[chunking] requirements.txt installation failed. Check Python compatibility for pinned packages." >&2
    fi
  else
    echo "[chunking] Installing chunking dependencies" >&2
    python -m pip install \
      langchain-text-splitters==0.2.0 \
      llama-index-core==0.10.0 \
      syntok==1.4.4 \
      tiktoken==0.6.0 \
      transformers==4.38.0 \
      pydantic==2.6.4

    if [[ ${PYTHON_MINOR} -lt 12 ]]; then
      python -m pip install 'unstructured[local-inference]==0.12.0'
    else
      echo "[chunking] Skipping unstructured==0.12.0 (requires Python <3.12)." >&2
      echo "[chunking] Install from a Python 3.11 runtime for production or select a compatible version." >&2
    fi
  fi

  if [[ -n "${HF_MODEL}" ]]; then
    echo "[chunking] Downloading Hugging Face tokenizer: ${HF_MODEL}" >&2
    python - <<PY
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("${HF_MODEL}", use_fast=True)
PY
  else
    echo "[chunking] Skipping Hugging Face tokenizer download; provide --hf-model or set MEDICAL_KG_SENTENCE_MODEL." >&2
  fi
fi

echo "[chunking] Verifying dependencies" >&2
python scripts/check_chunking_dependencies.py

echo "[chunking] Completed dependency installation" >&2
