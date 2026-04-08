#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$REPO_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
star_vla_python="${STAR_VLA_PYTHON:-$(which python)}"
your_ckpt="${YOUR_CKPT:-$REPO_ROOT/results/Checkpoints/libero_qwen25_latest/checkpoints/steps_10000_pytorch_model.pt}"
gpu_id="${GPU_ID:-0}"
port="${PORT:-5694}"

CUDA_VISIBLE_DEVICES="$gpu_id" "$star_vla_python" deployment/model_server/server_policy.py \
  --ckpt_path "$your_ckpt" \
  --port "$port" \
  --use_bf16
