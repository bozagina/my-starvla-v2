#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$REPO_ROOT"

export LIBERO_HOME="${LIBERO_HOME:-/2025233147/zzq/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$LIBERO_HOME}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-$(which python)}"

export PYTHONPATH="${PYTHONPATH:-}:$LIBERO_HOME"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

host="${HOST:-127.0.0.1}"
base_port="${BASE_PORT:-5694}"
unnorm_key="${UNNORM_KEY:-franka}"
your_ckpt="${YOUR_CKPT:-$REPO_ROOT/results/Checkpoints/libero_qwen25_latest/checkpoints/steps_10000_pytorch_model.pt}"
folder_name="$(basename "$(dirname "$(dirname "$your_ckpt")")")_$(basename "$your_ckpt")"

task_suite_name="${TASK_SUITE_NAME:-libero_goal}"
num_trials_per_task="${NUM_TRIALS_PER_TASK:-5}"
video_out_path="${VIDEO_OUT_PATH:-$REPO_ROOT/results/${task_suite_name}/${folder_name}}"

use_state="${USE_STATE:-true}"
expected_state_dim="${EXPECTED_STATE_DIM:-8}"
auto_pad_state_to_expected_dim="${AUTO_PAD_STATE_TO_EXPECTED_DIM:-false}"
log_payload_every_n_steps="${LOG_PAYLOAD_EVERY_N_STEPS:-1}"
repeat_infer_debug_times="${REPEAT_INFER_DEBUG_TIMES:-3}"

LOG_DIR="$REPO_ROOT/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_libero.log"
exec > >(tee -a "$LOG_FILE") 2>&1

extra_args=()
if [[ "$use_state" == "true" ]]; then
  extra_args+=(--args.use-state)
fi
if [[ "$auto_pad_state_to_expected_dim" == "true" ]]; then
  extra_args+=(--args.auto-pad-state-to-expected-dim)
fi
extra_args+=(--args.expected-state-dim "$expected_state_dim")
extra_args+=(--args.log-payload-every-n-steps "$log_payload_every_n_steps")
extra_args+=(--args.repeat-infer-debug-times "$repeat_infer_debug_times")

echo "Using host=$host"
echo "Using base_port=$base_port"
echo "Using task_suite_name=$task_suite_name"
echo "Using num_trials_per_task=$num_trials_per_task"
echo "Using video_out_path=$video_out_path"
echo "Using your_ckpt=$your_ckpt"
echo "Using unnorm_key=$unnorm_key"
echo "Logs will be saved to $LOG_FILE"

"$LIBERO_PYTHON" "$REPO_ROOT/examples/LIBERO/eval_files/eval_libero.py" \
  --args.pretrained-path "$your_ckpt" \
  --args.host "$host" \
  --args.port "$base_port" \
  --args.task-suite-name "$task_suite_name" \
  --args.num-trials-per-task "$num_trials_per_task" \
  --args.video-out-path "$video_out_path" \
  "${extra_args[@]}"
