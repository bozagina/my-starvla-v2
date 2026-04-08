#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$REPO_ROOT"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

framework_name="${FRAMEWORK_NAME:-QwenPI}"
freeze_module_list="${FREEZE_MODULE_LIST:-}"
base_vlm="${BASE_VLM:-/2025233147/zzq/SpatialVLA_llava3d/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct}"
config_yaml="${CONFIG_YAML:-$REPO_ROOT/starVLA/config/training/starvla_train_pi_qwen25.yaml}"
libero_data_root="${LIBERO_DATA_ROOT:-}"
data_mix="${DATA_MIX:-libero_all}"
run_root_dir="${RUN_ROOT_DIR:-$REPO_ROOT/results/Checkpoints}"
seed="${SEED:-42}"
per_device_bs="${PER_DEVICE_BS:-4}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-4}"
timestamp="$(date +"%Y%m%d_%H%M%S")"
run_id="libero_qwen25_${framework_name}_s${seed}_${timestamp}"

if [[ -z "$libero_data_root" ]]; then
  echo "[ERROR] LIBERO_DATA_ROOT is required."
  echo "[ERROR] Please export LIBERO_DATA_ROOT=/absolute/path/to/LEROBOT_LIBERO_DATA"
  exit 2
fi
if [[ ! -d "$libero_data_root" ]]; then
  echo "[ERROR] LIBERO_DATA_ROOT does not exist: $libero_data_root"
  exit 2
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_ENABLE_PARALLEL_LOADING=true
export HF_PARALLEL_LOADING_WORKERS=8
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
raw_log_file="${output_dir}/train.raw.log"
log_file="${output_dir}/train.log"
cp "$0" "${output_dir}/"

freeze_args=()
if [[ -n "${freeze_module_list}" ]]; then
  freeze_args=(--trainer.freeze_modules "${freeze_module_list}")
fi

stdbuf -oL -eL accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --gradient_accumulation_steps "${grad_accum_steps}" \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.a_module.mode standalone \
  --framework.a_module.standalone.strict_missing true \
  --framework.a_module.standalone.fallback_to_lite false \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_bs}" \
  --datasets.vla_data.video_backend torchvision_av \
  "${freeze_args[@]}" \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps "${grad_accum_steps}" \
  --seed "${seed}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starvla_qwen \
  2>&1 | tee -a "${raw_log_file}"

train_exit=${PIPESTATUS[0]}
tr '\r' '\n' < "${raw_log_file}" > "${log_file}"
echo "Saved raw log to: ${raw_log_file}"
echo "Saved normalized log to: ${log_file}"
exit ${train_exit}
