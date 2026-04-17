#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CP-AH-4-v1 / CP-AH-5 Baseline Training — Fusion mode WITHOUT CF
#
# Used for both:
#   CP-AH-5: RNG alignment baseline (single seed, compare with CF-no-infer)
#   CP-AH-4-v1: Multi-seed baseline (3 seeds × 20 trials × 10 tasks)
#
# Usage:
#   SEED=42 bash tools/handoff/run_cp_rtc_fusion_baseline_train.sh
###############################################################################

source /2025233147/miniconda3/bin/activate
conda activate /2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/vdpm/.venv_vdpm_fullflow

REPO_ROOT=/2025233147/zzq_0317/starVLA
cd "$REPO_ROOT"

SEED="${SEED:-42}"

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export WANDB_MODE=disabled
export HF_ENABLE_PARALLEL_LOADING=true
export HF_PARALLEL_LOADING_WORKERS=8
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

PRETRAINED_CKPT="/2025233147/zzq/SpatialVLA_llava3d/checkpoints_to_transfer/2d_vlm_no_geo_30k/checkpoints/steps_30000_pytorch_model.pt"

RUN_ID="cp_ah4_fusion_baseline_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${REPO_ROOT}/results/Checkpoints/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

CORRECTION_JSONL="${REPO_ROOT}/results/PseudoLabels/p3_0b_outcome_v093/correction_dataset_with_a_outputs.jsonl"

echo "=== CP-AH-4-v1 Fusion Baseline (NO CF, seed=${SEED}) ==="
echo "RUN_ID: ${RUN_ID}"
echo "PRETRAINED: ${PRETRAINED_CKPT}"
echo "SEED: ${SEED}"
echo "STEPS: 30000→33000 (3000 additional)"
echo "a_module.mode: fusion"
echo "corrective_flow: DISABLED"
echo "====================================================="

stdbuf -oL -eL accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --gradient_accumulation_steps 4 \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_pi_from30k.yaml \
  --framework.name QwenPI \
  --framework.a_module.mode fusion \
  --framework.a_module.standalone.input_key a_outputs \
  --framework.a_module.standalone.allow_pseudo_labels_fallback false \
  --framework.a_module.standalone.fallback_to_lite false \
  --framework.a_module.standalone.strict_missing true \
  --framework.a_module.vdpm_device cuda:0 \
  --datasets.vla_data.data_root_dir /2025233147/zzq/SpatialVLA_llava3d/playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.per_device_batch_size 4 \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vla_data.correction_dataset_jsonl "${CORRECTION_JSONL}" \
  --datasets.vla_data.correction_supervision_enabled true \
  --datasets.vla_data.correction_supervision_filter_to_index true \
  --trainer.pretrained_checkpoint "${PRETRAINED_CKPT}" \
  --trainer.optional_loss_hooks.enabled true \
  --trainer.optional_loss_hooks.a_loss.enabled true \
  --trainer.optional_loss_hooks.a_loss.risk_weight 1.0 \
  --trainer.optional_loss_hooks.a_loss.trigger_weight 1.0 \
  --trainer.optional_loss_hooks.a_loss.embed_weight 1.0 \
  --trainer.optional_loss_hooks.a_loss.consist_weight 0.1 \
  --trainer.optional_loss_hooks.corrective_loss.enabled true \
  --trainer.optional_loss_hooks.corrective_loss.delta_norm_weight 1.0 \
  --trainer.optional_loss_hooks.corrective_loss.correction_mask_weight 1.0 \
  --trainer.optional_loss_hooks.corrective_loss.region_prior_weight 0.5 \
  --trainer.max_train_steps 33000 \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 5 \
  --trainer.eval_interval 33000 \
  --trainer.gradient_accumulation_steps 4 \
  --seed "${SEED}" \
  --run_root_dir "${REPO_ROOT}/results/Checkpoints" \
  --run_id "${RUN_ID}" \
  --wandb_project starvla_qwen \
  2>&1 | tee -a "${OUTPUT_DIR}/train.raw.log"

train_exit=${PIPESTATUS[0]}
tr '\r' '\n' < "${OUTPUT_DIR}/train.raw.log" > "${OUTPUT_DIR}/train.log"
echo "Training exit code: ${train_exit}"
echo "Logs: ${OUTPUT_DIR}/train.log"
exit ${train_exit}
