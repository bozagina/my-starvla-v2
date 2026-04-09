#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-myserver}"
REMOTE_REPO="${REMOTE_REPO:-/2025233147/zzq_0317/starVLA}"
CKPT_ROOT="$REMOTE_REPO/results/Checkpoints"

ssh "$HOST" bash -lc "'
set -euo pipefail
latest=\$(ls -td \"$CKPT_ROOT\"/*/ 2>/dev/null | head -n 1)
if [ -z \"\${latest:-}\" ]; then
  echo \"No checkpoint run dir found under $CKPT_ROOT\"
  exit 1
fi
echo \"Latest run dir: \$latest\"
echo \"Tailing: \$latest/train.log\"
tail -n 200 -F \"\$latest/train.log\"
'"
