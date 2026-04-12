#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"

required_scripts=(
  "tools/build_fasa_dataset.py"
  "tools/build_fasa_a_outputs.py"
  "tools/a_outputs_audit.py"
)

echo "[fasa-preflight] repo_root=$REPO_ROOT"
echo "[fasa-preflight] python=$PYTHON_BIN"

for rel in "${required_scripts[@]}"; do
  abs="$REPO_ROOT/$rel"
  if [[ ! -f "$abs" ]]; then
    echo "[fasa-preflight] missing: $rel"
    exit 2
  fi
done

for rel in "${required_scripts[@]}"; do
  abs="$REPO_ROOT/$rel"
  echo "[fasa-preflight] smoke --help: $rel"
  "$PYTHON_BIN" "$abs" --help >/dev/null
done

echo "[fasa-preflight] OK"
