#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CP-BUILD RTC Smoke Test — Fusion mode + CorrectiveFlowHead (RTC v1.1)
#
# 500-step smoke test to validate code correctness before multi-seed training.
# BD-7: self-attention only (use_cross_attention=false)
# BD-8: trigger/region NOT input to CF model (RTC paradigm)
# ENG-3: CF RNG alignment via independent Generator
###############################################################################

source /2025233147/miniconda3/bin/activate
conda activate /2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/vdpm/.venv_vdpm_fullflow

REPO_ROOT=/2025233147/zzq_0317/starVLA
cd "$REPO_ROOT"

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

RUN_ID="cp_rtc_smoke_500step_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${REPO_ROOT}/results/Checkpoints/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

CORRECTION_JSONL="${REPO_ROOT}/results/PseudoLabels/p3_0b_outcome_v093/correction_dataset_with_a_outputs.jsonl"

echo "=== CP-BUILD RTC Smoke Test (Fusion + CF RTC v1.1, 500 steps) ==="
echo "RUN_ID: ${RUN_ID}"
echo "a_module.mode: fusion"
echo "corrective_flow: enabled, RTC paradigm, self-attn only"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "================================================================="

stdbuf -oL -eL accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --gradient_accumulation_steps 4 \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_pi.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm /2025233147/zzq/SpatialVLA_llava3d/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct \
  --framework.a_module.mode fusion \
  --framework.a_module.standalone.input_key a_outputs \
  --framework.a_module.standalone.allow_pseudo_labels_fallback false \
  --framework.a_module.standalone.fallback_to_lite false \
  --framework.a_module.standalone.strict_missing true \
  --framework.a_module.vdpm_device cuda:0 \
  --framework.corrective_flow.enabled true \
  --framework.corrective_flow.num_layers 4 \
  --framework.corrective_flow.hidden_dim 1024 \
  --framework.corrective_flow.num_heads 16 \
  --framework.corrective_flow.region_loss_weight 0.5 \
  --framework.corrective_flow.use_cross_attention false \
  --datasets.vla_data.data_root_dir /2025233147/zzq/SpatialVLA_llava3d/playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.per_device_batch_size 2 \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vla_data.correction_dataset_jsonl "${CORRECTION_JSONL}" \
  --datasets.vla_data.correction_supervision_enabled true \
  --datasets.vla_data.correction_supervision_filter_to_index true \
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
  --trainer.optional_loss_hooks.corrective_flow.enabled true \
  --trainer.optional_loss_hooks.corrective_flow.key corrective_flow_loss \
  --trainer.optional_loss_hooks.corrective_flow.metric_name loss/corrective_flow \
  --trainer.optional_loss_hooks.corrective_flow.scale 0.5 \
  --trainer.optional_loss_hooks.corrective_flow.noise_scale 0.05 \
  --trainer.max_train_steps 500 \
  --trainer.save_interval 100 \
  --trainer.logging_frequency 5 \
  --trainer.eval_interval 500 \
  --trainer.gradient_accumulation_steps 4 \
  --seed 42 \
  --run_root_dir "${REPO_ROOT}/results/Checkpoints" \
  --run_id "${RUN_ID}" \
  --wandb_project starvla_qwen \
  2>&1 | tee -a "${OUTPUT_DIR}/train.raw.log"

train_exit=${PIPESTATUS[0]}
tr '\r' '\n' < "${OUTPUT_DIR}/train.raw.log" > "${OUTPUT_DIR}/train.log"
echo "Training exit code: ${train_exit}"
echo "Logs: ${OUTPUT_DIR}/train.log"
exit ${train_exit}
