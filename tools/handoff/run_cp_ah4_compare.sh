#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CP-AH-4 Comparison: A-only baseline vs Corrective Flow
#
# Runs LIBERO eval for both models and compares Success@LIBERO.
# Both models trained for 500 steps (apples-to-apples comparison).
###############################################################################

REPO_ROOT=/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA
SCRIPT="${REPO_ROOT}/tools/handoff/run_cp_ah4_libero_eval.sh"

BASELINE_CKPT="${REPO_ROOT}/results/Checkpoints/ah4_v093_500step_20260414_115012/final_model/pytorch_model.pt"
CF_CKPT="${REPO_ROOT}/results/Checkpoints/cp_smoke_500step_20260415_042824/final_model/pytorch_model.pt"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${REPO_ROOT}/results/LiberoEval/cp_ah4_compare_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo "=============================================="
echo "  CP-AH-4 Comparison Run"
echo "  Timestamp: ${TIMESTAMP}"
echo "=============================================="
echo ""

# --- Phase 1: A-only baseline ---
echo "========== Phase 1: A-only Baseline =========="
echo "Checkpoint: ${BASELINE_CKPT}"
if [ -f "${BASELINE_CKPT}" ]; then
    bash "${SCRIPT}" "${BASELINE_CKPT}" "cp_ah4_baseline_${TIMESTAMP}" 2>&1 | tee "${RESULTS_DIR}/baseline_full.log"
    BASELINE_EXIT=${PIPESTATUS[0]}
    echo "Baseline eval exit code: ${BASELINE_EXIT}"
else
    echo "ERROR: Baseline checkpoint not found: ${BASELINE_CKPT}"
    BASELINE_EXIT=1
fi

echo ""
echo "========== Phase 2: Corrective Flow =========="
echo "Checkpoint: ${CF_CKPT}"
if [ -f "${CF_CKPT}" ]; then
    bash "${SCRIPT}" "${CF_CKPT}" "cp_ah4_cf_${TIMESTAMP}" 2>&1 | tee "${RESULTS_DIR}/cf_full.log"
    CF_EXIT=${PIPESTATUS[0]}
    echo "CF eval exit code: ${CF_EXIT}"
else
    echo "ERROR: CF checkpoint not found: ${CF_CKPT}"
    CF_EXIT=1
fi

echo ""
echo "=============================================="
echo "  COMPARISON COMPLETE"
echo "=============================================="
echo "Baseline exit: ${BASELINE_EXIT}"
echo "CF exit:       ${CF_EXIT}"
echo "Results dir:   ${RESULTS_DIR}"
echo ""
echo "To extract Success@LIBERO scores:"
echo "  grep -i 'success\|average.*rate' ${RESULTS_DIR}/baseline_full.log"
echo "  grep -i 'success\|average.*rate' ${RESULTS_DIR}/cf_full.log"
