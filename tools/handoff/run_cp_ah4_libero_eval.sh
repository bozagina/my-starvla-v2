#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CP-AH-4 LIBERO Evaluation Script
#
# Runs LIBERO eval for a given checkpoint.
# Usage:
#   bash run_cp_ah4_libero_eval.sh <CKPT_PATH> [EVAL_TAG]
#
# The script:
#   1. Starts a policy server on a free port (background)
#   2. Waits for the server to be ready
#   3. Runs eval_libero.py against the server
#   4. Kills the server
#   5. Saves results to results/LiberoEval/<EVAL_TAG>/
###############################################################################

CKPT_PATH="${1:?Usage: $0 <CKPT_PATH> [EVAL_TAG]}"
EVAL_TAG="${2:-eval_$(date +%Y%m%d_%H%M%S)}"

_saved_args=("$@")
set --
source /2025233147/miniconda3/bin/activate
conda activate /2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/vdpm/.venv_vdpm_fullflow
set -- "${_saved_args[@]}"

REPO_ROOT=/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA
cd "$REPO_ROOT"

export LIBERO_HOME=/2025233147/zzq/LIBERO
export LIBERO_CONFIG_PATH=/2025233147/zzq/LIBERO
export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
NUM_TRIALS="${LIBERO_NUM_TRIALS:-20}"
PORT="${LIBERO_PORT:-5700}"
GPU_SERVER="${LIBERO_GPU:-0}"
GPU_EVAL=""

EVAL_DIR="${REPO_ROOT}/results/LiberoEval/${EVAL_TAG}"
mkdir -p "${EVAL_DIR}"
cp "$0" "${EVAL_DIR}/"

echo "=============================================="
echo "  CP-AH-4 LIBERO Evaluation"
echo "=============================================="
echo "CKPT:       ${CKPT_PATH}"
echo "EVAL_TAG:   ${EVAL_TAG}"
echo "TASK_SUITE: ${TASK_SUITE}"
echo "NUM_TRIALS: ${NUM_TRIALS}"
echo "PORT:       ${PORT}"
echo "EVAL_DIR:   ${EVAL_DIR}"
echo "=============================================="

# --- Step 1: Start policy server ---
echo "[$(date)] Starting policy server on GPU ${GPU_SERVER}, port ${PORT} ..."
CUDA_VISIBLE_DEVICES=${GPU_SERVER} stdbuf -oL python -u deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port ${PORT} \
    --use_bf16 \
    > "${EVAL_DIR}/server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

cleanup() {
    echo "[$(date)] Cleaning up server (PID=${SERVER_PID}) ..."
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
}
trap cleanup EXIT

# --- Step 2: Wait for server to be ready ---
echo "[$(date)] Waiting for server to be ready ..."
MAX_WAIT=300
WAITED=0
while ! grep -q "server running" "${EVAL_DIR}/server.log" 2>/dev/null; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "ERROR: Server process died"
        tail -30 "${EVAL_DIR}/server.log"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "ERROR: Server did not start within ${MAX_WAIT}s"
        echo "Server log tail:"
        tail -30 "${EVAL_DIR}/server.log"
        exit 1
    fi
    echo "  ... waited ${WAITED}s"
done
echo "[$(date)] Server is ready (waited ${WAITED}s)"

# --- Step 3: Run LIBERO eval ---
echo "[$(date)] Running LIBERO evaluation ..."
FOLDER_NAME=$(echo "${CKPT_PATH}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
VIDEO_OUT="${EVAL_DIR}/videos"

python examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT_PATH}" \
    --args.host "127.0.0.1" \
    --args.port "${PORT}" \
    --args.task-suite-name "${TASK_SUITE}" \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.video-out-path "${VIDEO_OUT}" \
    --args.use-state \
    --args.expected-state-dim 0 \
    --args.seed 7 \
    --args.log-payload-every-n-steps 20 \
    --args.repeat-infer-debug-times 1 \
    2>&1 | tee "${EVAL_DIR}/eval.log"

EVAL_EXIT=${PIPESTATUS[0]}
echo "[$(date)] Eval exit code: ${EVAL_EXIT}"

# --- Step 4: Extract summary ---
echo ""
echo "=============================================="
echo "  RESULTS SUMMARY"
echo "=============================================="
if grep -i "success\|average\|score\|rate" "${EVAL_DIR}/eval.log" | tail -20; then
    :
else
    echo "(no summary lines found; check eval.log)"
fi

echo ""
echo "Full logs: ${EVAL_DIR}/eval.log"
echo "Server logs: ${EVAL_DIR}/server.log"

exit ${EVAL_EXIT}
