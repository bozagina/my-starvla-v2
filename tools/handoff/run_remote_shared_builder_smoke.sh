#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-myserver}"
REMOTE_REPO="${REMOTE_REPO:-/2025233147/zzq_0317/starVLA}"
CONFIG_YAML="${CONFIG_YAML:-starVLA/config/training/starvla_cotrain_oxe.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REMOTE_SETUP_CMD="${REMOTE_SETUP_CMD:-}"
SMOKE_MODE="${SMOKE_MODE:-auto}" # auto|dataloader|builder_unit
SHARED_BUILDER_ENABLED="${SHARED_BUILDER_ENABLED:-true}"
INCLUDE_STATE="${INCLUDE_STATE:-true}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
LOG_DIR="${LOG_DIR:-results/Checkpoints/_launch_logs}"
SMOKE_TAG="${SMOKE_TAG:-shared_builder_smoke_$(date +%Y%m%d_%H%M%S)}"

echo "[smoke] host=$HOST"
echo "[smoke] remote_repo=$REMOTE_REPO"
echo "[smoke] config_yaml=$CONFIG_YAML"
echo "[smoke] smoke_mode=$SMOKE_MODE python_bin=$PYTHON_BIN"
echo "[smoke] shared_builder_enabled=$SHARED_BUILDER_ENABLED include_state=$INCLUDE_STATE sample_index=$SAMPLE_INDEX"

ssh "$HOST" \
  "REMOTE_REPO='$REMOTE_REPO' CONFIG_YAML='$CONFIG_YAML' PYTHON_BIN='$PYTHON_BIN' REMOTE_SETUP_CMD='$REMOTE_SETUP_CMD' SMOKE_MODE='$SMOKE_MODE' SHARED_BUILDER_ENABLED='$SHARED_BUILDER_ENABLED' INCLUDE_STATE='$INCLUDE_STATE' SAMPLE_INDEX='$SAMPLE_INDEX' LOG_DIR='$LOG_DIR' SMOKE_TAG='$SMOKE_TAG' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

cd "$REMOTE_REPO"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SMOKE_TAG}.log"

{
  echo "=== Shared Builder Smoke ==="
  echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "repo=$REMOTE_REPO"
  echo "config=$CONFIG_YAML"
  echo "python=$PYTHON_BIN"
  echo "smoke_mode=$SMOKE_MODE"
  echo "shared_builder_enabled=$SHARED_BUILDER_ENABLED include_state=$INCLUDE_STATE sample_index=$SAMPLE_INDEX"

  if [ -n "$REMOTE_SETUP_CMD" ]; then
    echo "[smoke] running REMOTE_SETUP_CMD"
    eval "$REMOTE_SETUP_CMD"
  fi

  "$PYTHON_BIN" - <<'PY'
import ast
import json
import os
import traceback
from pathlib import Path


def str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def shape_of(x):
    if x is None:
        return None
    shape = getattr(x, "shape", None)
    if shape is not None:
        try:
            return list(shape)
        except Exception:
            return str(shape)
    if isinstance(x, list):
        return [len(x)]
    return None


config_yaml = os.environ["CONFIG_YAML"]
smoke_mode = os.environ.get("SMOKE_MODE", "auto").strip().lower()
shared_builder_enabled = os.environ["SHARED_BUILDER_ENABLED"]
include_state = os.environ["INCLUDE_STATE"]
sample_index = int(os.environ["SAMPLE_INDEX"])

def _extract_builder_fn(repo_root: Path):
    import numpy as np

    src = repo_root / "starVLA/dataloader/gr00t_lerobot/datasets.py"
    tree = ast.parse(src.read_text(encoding="utf-8"), filename=src.as_posix())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"_cfg_enabled", "build_shared_vla_sample"}:
            selected.append(node)
    if not selected:
        raise RuntimeError(f"build_shared_vla_sample not found in {src}")
    module = ast.Module(body=selected, type_ignores=[])
    ns = {"np": np}
    exec(compile(module, src.as_posix(), "exec"), ns, ns)
    return ns["build_shared_vla_sample"]


def run_builder_unit_smoke():
    import numpy as np
    from PIL import Image

    builder = _extract_builder_fn(Path(".").resolve())
    include_state_flag = str2bool(include_state)
    enabled_flag = str2bool(shared_builder_enabled)

    images = [
        Image.new("RGB", (224, 224), color=(0, 0, 0)),
        Image.new("RGB", (224, 224), color=(16, 16, 16)),
    ]
    action = np.zeros((16, 7), dtype=np.float32)
    state = np.zeros((14,), dtype=np.float32) if include_state_flag else None

    args = dict(
        images=images,
        action_chunk=action,
        language="remote_builder_unit_smoke",
        state=state,
        dataset_name="unit_smoke_dataset",
        trajectory_id=0,
        step_index=0,
        action_keys=["eef_xyz", "eef_quat", "gripper"],
        state_keys=["eef_xyz", "eef_quat", "gripper"] if include_state_flag else [],
        video_keys=["video.image_primary", "video.image_wrist"],
    )
    sample_enabled = builder(**args, builder_enabled=True)
    sample_disabled = builder(**args, builder_enabled=False)
    sample = sample_enabled if enabled_flag else sample_disabled

    return {
        "mode": "builder_unit",
        "shared_builder_enabled": enabled_flag,
        "sample_keys": sorted(sample.keys()),
        "enabled_sample_keys": sorted(sample_enabled.keys()),
        "legacy_sample_keys": sorted(sample_disabled.keys()),
        "lang_type": type(sample.get("lang")).__name__,
        "image_views": len(sample.get("image", [])) if isinstance(sample.get("image"), list) else None,
        "obs_views": len(sample.get("obs", [])) if isinstance(sample.get("obs"), list) else None,
        "action_shape": shape_of(sample.get("action")),
        "action_chunk_shape": shape_of(sample.get("action_chunk")),
        "state_shape": shape_of(sample.get("state")),
        "meta": sample.get("meta", None),
    }


def run_dataloader_smoke():
    from omegaconf import OmegaConf
    import torch.distributed as dist
    from starVLA.dataloader import build_dataloader

    if not dist.is_initialized():
        dist.get_rank = lambda: 0  # noqa: E731

    cfg_path = Path(config_yaml)
    cfg = OmegaConf.load(cfg_path.as_posix())
    cfg.output_dir = "/tmp/starvla_shared_builder_smoke"
    cfg.datasets.vla_data.shared_builder_enabled = str2bool(shared_builder_enabled)
    cfg.datasets.vla_data.include_state = str2bool(include_state)

    dataloader = build_dataloader(cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    batch = next(iter(dataloader))
    sample = batch[sample_index]

    return {
        "mode": "dataloader",
        "sample_keys": sorted(sample.keys()),
        "lang_type": type(sample.get("lang")).__name__,
        "image_views": len(sample.get("image", [])) if isinstance(sample.get("image"), list) else None,
        "obs_views": len(sample.get("obs", [])) if isinstance(sample.get("obs"), list) else None,
        "action_shape": shape_of(sample.get("action")),
        "action_chunk_shape": shape_of(sample.get("action_chunk")),
        "state_shape": shape_of(sample.get("state")),
        "meta": sample.get("meta", None),
    }


def main():
    if smoke_mode == "dataloader":
        payload = run_dataloader_smoke()
    elif smoke_mode == "builder_unit":
        payload = run_builder_unit_smoke()
    elif smoke_mode == "auto":
        try:
            payload = run_dataloader_smoke()
        except Exception as exc:
            print("[smoke] dataloader smoke failed, fallback to builder_unit")
            traceback.print_exc()
            payload = run_builder_unit_smoke()
            payload["fallback_from"] = "dataloader"
            payload["fallback_reason"] = f"{type(exc).__name__}: {exc}"
    else:
        raise ValueError(f"Unsupported SMOKE_MODE: {smoke_mode}")

    print("SMOKE_RESULT_JSON_START")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("SMOKE_RESULT_JSON_END")


if __name__ == "__main__":
    main()
PY
} 2>&1 | tee "$LOG_FILE"

echo "[smoke] log_file=$REMOTE_REPO/$LOG_FILE"
REMOTE_SCRIPT
