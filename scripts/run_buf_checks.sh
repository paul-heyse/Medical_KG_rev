#!/usr/bin/env bash
set -euo pipefail

if ! command -v buf >/dev/null 2>&1; then
  echo "buf CLI not available" >&2
  exit 1
fi

buf lint
buf breaking --against .git#branch=main || true
