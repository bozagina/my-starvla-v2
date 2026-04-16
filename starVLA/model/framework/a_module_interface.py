from __future__ import annotations

import atexit
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import torch

from starVLA.model.framework.a_fusion_heads import AModuleRuntimeBatch


def _cfg_get(cfg_obj, key: str, default=None):
    if cfg_obj is None:
        return default
    if hasattr(cfg_obj, "get"):
        try:
            return cfg_obj.get(key, default)
        except Exception:
            pass
    return getattr(cfg_obj, key, default)


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


_VERSIONED_DATASET_RE = re.compile(r"^(?P<prefix>.+?)_\d+\.\d+\.\d+_lerobot$")
_REQUIRED_REGION_LEN = 15
_REQUIRED_EMBED_LEN = 16


def _dataset_name_aliases(dataset_name: str) -> list[str]:
    raw = str(dataset_name)
    aliases = {raw}
    basename = Path(raw).name
    aliases.add(basename)

    match = _VERSIONED_DATASET_RE.match(basename)
    if match is not None:
        aliases.add(match.group("prefix"))
    if basename.endswith("_lerobot"):
        aliases.add(basename[: -len("_lerobot")])
    return [alias for alias in aliases if alias]


def _coerce_vector(value: Any, target_len: int) -> torch.Tensor | None:
    if target_len <= 0:
        return None
    try:
        arr = torch.as_tensor(value, dtype=torch.float32)
    except Exception:
        return None
    if arr.numel() == 0:
        return None
    arr = arr.reshape(-1)
    arr = torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.numel() >= target_len:
        return arr[:target_len]
    padded = torch.zeros((target_len,), dtype=torch.float32)
    padded[: arr.numel()] = arr
    return padded


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clip01(value: float) -> float:
    return _clip(value, 0.0, 1.0)


def _logit(prob: float) -> float:
    p = _clip(prob, 1e-4, 1.0 - 1e-4)
    return float(math.log(p / (1.0 - p)))


def _all_finite_list(value: Any, expected_len: int) -> bool:
    if not isinstance(value, list):
        return False
    if len(value) != expected_len:
        return False
    return all(_safe_float(v) is not None for v in value)


def _normalize_image_views(image_payload: Any) -> list[np.ndarray]:
    views: list[np.ndarray] = []
    if image_payload is None:
        return views

    def _append_array(item: Any) -> None:
        arr = item if isinstance(item, np.ndarray) else np.asarray(item)
        if arr.ndim == 3:
            views.append(arr.astype(np.uint8, copy=False))
        elif arr.ndim == 4:
            for i in range(arr.shape[0]):
                sub = np.asarray(arr[i])
                if sub.ndim == 3:
                    views.append(sub.astype(np.uint8, copy=False))

    if isinstance(image_payload, np.ndarray):
        _append_array(image_payload)
        return views
    if isinstance(image_payload, (list, tuple)):
        for item in image_payload:
            _append_array(item)
        return views

    _append_array(image_payload)
    return views


def _payload_contract_ok(payload: dict[str, Any], expected_region_len: int, expected_embedding_dim: int) -> bool:
    required = (
        "risk_pred",
        "trigger_logit",
        "delta_pred",
        "region_logits",
        "dynamic_embedding",
        "version",
        "source",
    )
    if not all(k in payload for k in required):
        return False
    if _safe_float(payload.get("risk_pred")) is None:
        return False
    if _safe_float(payload.get("trigger_logit")) is None:
        return False
    delta = _safe_float(payload.get("delta_pred"))
    if delta is None or delta < 0.0:
        return False
    if not _all_finite_list(payload.get("region_logits"), expected_region_len):
        return False
    if not _all_finite_list(payload.get("dynamic_embedding"), expected_embedding_dim):
        return False
    if not isinstance(payload.get("version"), str) or not payload.get("version"):
        return False
    if not isinstance(payload.get("source"), str) or not payload.get("source"):
        return False
    return True


def _extract_runtime_payload(record: dict, input_key: str) -> dict | None:
    payload = record.get(input_key)
    if isinstance(payload, dict):
        return payload
    pseudo = record.get("pseudo_labels")
    if isinstance(pseudo, dict):
        pseudo_payload = pseudo.get(input_key)
        if isinstance(pseudo_payload, dict):
            return pseudo_payload
    required_keys = {"risk_pred", "trigger_logit", "delta_pred", "region_logits", "dynamic_embedding"}
    if required_keys.issubset(record.keys()):
        return {key: record.get(key) for key in required_keys.union({"version", "source"})}
    return None


def _load_runtime_a_outputs_index(jsonl_path: str, input_key: str) -> tuple[dict[tuple[str, int, int], dict], dict[str, int]]:
    path = Path(jsonl_path).expanduser()
    stats = {
        "rows": 0,
        "indexed": 0,
        "duplicates": 0,
        "invalid": 0,
        "invalid_json": 0,
        "invalid_meta": 0,
        "invalid_payload": 0,
    }
    if not path.exists():
        raise FileNotFoundError(f"in-loop runtime jsonl not found: {path}")

    index: dict[tuple[str, int, int], dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid"] += 1
                stats["invalid_json"] += 1
                continue
            if not isinstance(record, dict):
                stats["invalid"] += 1
                stats["invalid_json"] += 1
                continue

            meta = record.get("meta")
            if not isinstance(meta, dict):
                stats["invalid"] += 1
                stats["invalid_meta"] += 1
                continue
            dataset_name = str(meta.get("dataset_name", "")).strip()
            trajectory_id = _safe_int(meta.get("trajectory_id"))
            sample_step = _safe_int(meta.get("sample_step"))
            if not dataset_name or trajectory_id is None or sample_step is None:
                stats["invalid"] += 1
                stats["invalid_meta"] += 1
                continue

            payload = _extract_runtime_payload(record, input_key=input_key)
            if not isinstance(payload, dict):
                stats["invalid"] += 1
                stats["invalid_payload"] += 1
                continue

            for alias in _dataset_name_aliases(dataset_name):
                key = (alias, trajectory_id, sample_step)
                if key in index:
                    stats["duplicates"] += 1
                index[key] = payload
                stats["indexed"] += 1

    return index, stats


_VDPM_WORKER_SCRIPT = r"""#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clip01(value: float) -> float:
    return _clip(value, 0.0, 1.0)


def _logit(prob: float) -> float:
    p = _clip(prob, 1e-4, 1.0 - 1e-4)
    return float(math.log(p / (1.0 - p)))


def _partition_mean_3x5(matrix: torch.Tensor) -> list[float]:
    h, w = matrix.shape
    h_edges = torch.linspace(0, h, steps=4).round().to(torch.int64)
    w_edges = torch.linspace(0, w, steps=6).round().to(torch.int64)
    out = []
    for r in range(3):
        hs = int(h_edges[r].item())
        he = int(h_edges[r + 1].item())
        for c in range(5):
            ws = int(w_edges[c].item())
            we = int(w_edges[c + 1].item())
            patch = matrix[hs:he, ws:we]
            out.append(float(patch.mean().item()) if patch.numel() else 0.0)
    return out


def _build_contract_from_pred(pred: dict, num_frames: int, version: str, source: str) -> dict:
    pointmaps = pred["pointmaps"]
    pm = pointmaps[-1]
    pts = pm["pts3d"]
    conf = pm["conf"]

    s = int(pts.shape[1])
    cur_idx = s - 1
    prev_idx = max(0, s - 2)
    pts_cur = pts[0, cur_idx].float().detach().cpu()
    pts_prev = pts[0, prev_idx].float().detach().cpu()
    conf_cur = conf[0, cur_idx].float().detach().cpu()

    delta_map = torch.linalg.norm(pts_cur - pts_prev, dim=-1)
    delta_mean = float(delta_map.mean().item())
    delta_std = float(delta_map.std(unbiased=False).item())
    delta_max = float(delta_map.max().item())
    delta_q90 = float(torch.quantile(delta_map, 0.9).item())

    conf_sig = torch.sigmoid(conf_cur)
    conf_mean = float(conf_sig.mean().item())
    conf_std = float(conf_sig.std(unbiased=False).item())
    conf_min = float(conf_sig.min().item())
    conf_max = float(conf_sig.max().item())
    uncertainty = float((1.0 - conf_sig).mean().item())

    risk_pred = _clip01(0.6 * (delta_q90 / (delta_q90 + 1.0)) + 0.4 * uncertainty)
    trigger_prob = _clip01(delta_q90 / (delta_q90 + 0.25))
    trigger_logit = _logit(trigger_prob)
    delta_pred = max(0.0, delta_mean)

    region_raw = _partition_mean_3x5(delta_map)
    region_max = max(max(region_raw), 1e-8)
    region_probs = [_clip01(v / region_max) for v in region_raw]
    region_logits = [_logit(p) for p in region_probs]

    pose = pred.get("pose_enc")
    if isinstance(pose, torch.Tensor) and pose.ndim >= 3:
        pose_vec = pose[0, -1].float().detach().cpu()
        pose_trans_norm = float(torch.linalg.norm(pose_vec[:3]).item())
        pose_rot_norm = float(torch.linalg.norm(pose_vec[3:]).item())
    else:
        pose_trans_norm = 0.0
        pose_rot_norm = 0.0

    pts_norm = torch.linalg.norm(pts_cur, dim=-1)
    pts_norm_mean = float(pts_norm.mean().item())
    pts_norm_std = float(pts_norm.std(unbiased=False).item())
    pts_norm_max = float(pts_norm.max().item())

    dynamic_embedding = [
        float(risk_pred),
        float(trigger_logit),
        float(delta_pred),
        float(delta_mean),
        float(delta_std),
        float(delta_max),
        float(conf_mean),
        float(conf_std),
        float(conf_min),
        float(conf_max),
        float(pts_norm_mean),
        float(pts_norm_std),
        float(pts_norm_max),
        float(pose_trans_norm),
        float(pose_rot_norm),
        float(num_frames),
    ]

    return {
        "risk_pred": float(risk_pred),
        "trigger_logit": float(trigger_logit),
        "delta_pred": float(delta_pred),
        "region_logits": [float(x) for x in region_logits],
        "dynamic_embedding": [float(x) for x in dynamic_embedding],
        "version": version,
        "source": source,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vdpm-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    vdpm_root = Path(args.vdpm_root).expanduser().resolve()
    if not vdpm_root.exists():
        raise FileNotFoundError(f"vdpm_root not found: {vdpm_root}")
    sys.path.insert(0, str(vdpm_root))

    from hydra import compose, initialize_config_dir
    from visualise import load_model, preprocess_images

    cfg_dir = str(vdpm_root / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="visualise")
    model = load_model(cfg, args.device)
    model.eval()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        req_id = str(req.get("req_id", ""))
        cmd = str(req.get("cmd", "infer"))
        if cmd == "close":
            print(json.dumps({"ok": True, "closed": True, "req_id": req_id}), flush=True)
            return
        if cmd != "infer":
            print(json.dumps({"ok": False, "error": f"unknown_cmd:{cmd}", "req_id": req_id}), flush=True)
            continue

        try:
            frames_npz = Path(req["frames_npz"]).expanduser()
            arr = np.load(frames_npz)["frames"]
            if arr.ndim != 4:
                raise ValueError(f"frames shape invalid: {arr.shape}")
            frames = [arr[i] for i in range(arr.shape[0])]
            images = preprocess_images(frames).to(args.device)
            with torch.no_grad():
                pred = model.inference(None, images=images.unsqueeze(0))
            payload = _build_contract_from_pred(
                pred=pred,
                num_frames=int(arr.shape[0]),
                version=str(req.get("version", "vdpm_runtime_model_v1")),
                source=str(req.get("source", "vdpm/runtime_model")),
            )
            print(json.dumps({"ok": True, "payload": payload, "req_id": req_id}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc), "req_id": req_id}), flush=True)


if __name__ == "__main__":
    main()
"""


class _VDPMRuntimeModelWorker:
    def __init__(
        self,
        *,
        python_bin: str,
        vdpm_root: str,
        device: str = "cuda:0",
        log_path: str | None = None,
    ) -> None:
        self.python_bin = str(python_bin)
        self.vdpm_root = str(vdpm_root)
        self.device = str(device)
        self.log_path = str(log_path) if log_path else ""
        self.proc: subprocess.Popen | None = None
        self._script_path: str | None = None
        self._request_seq = 0

    def _ensure_script(self) -> str:
        if self._script_path and Path(self._script_path).exists():
            return self._script_path
        script_path = Path(tempfile.gettempdir()) / "starvla_vdpm_runtime_worker.py"
        script_path.write_text(_VDPM_WORKER_SCRIPT, encoding="utf-8")
        script_path.chmod(0o755)
        self._script_path = str(script_path)
        return self._script_path

    def _ensure_proc(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        script_path = self._ensure_script()
        stderr_stream = open(self.log_path, "a", encoding="utf-8") if self.log_path else subprocess.PIPE
        self.proc = subprocess.Popen(
            [self.python_bin, script_path, "--vdpm-root", self.vdpm_root, "--device", self.device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_stream,
            text=True,
            bufsize=1,
        )
        atexit.register(self.close)

    def infer(
        self,
        *,
        frames: list[np.ndarray],
        version: str,
        source: str,
    ) -> dict[str, Any]:
        self._ensure_proc()
        assert self.proc is not None
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

        stacked = np.stack(frames, axis=0).astype(np.uint8, copy=False)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="starvla_vdpm_frames_", suffix=".npz")
        os.close(tmp_fd)
        np.savez_compressed(tmp_path, frames=stacked)
        req = {
            "cmd": "infer",
            "frames_npz": tmp_path,
            "version": version,
            "source": source,
            "req_id": str(self._request_seq + 1),
        }
        self._request_seq += 1
        expected_req_id = str(self._request_seq)
        try:
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            resp: dict[str, Any] | None = None
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("vdpm worker returned empty response")
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    candidate = json.loads(stripped)
                except json.JSONDecodeError:
                    # Some VDPM dependencies print startup/status lines to stdout.
                    # Keep reading until we see a JSON response line.
                    continue
                if not isinstance(candidate, dict):
                    continue
                if str(candidate.get("req_id", "")) != expected_req_id:
                    continue
                resp = candidate
                break
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if not isinstance(resp, dict):
            raise RuntimeError("vdpm worker returned invalid response")
        if not resp.get("ok", False):
            raise RuntimeError(str(resp.get("error", "unknown_vdpm_worker_error")))
        payload = resp.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeError("vdpm worker payload missing")
        return payload

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None and self.proc.stdin is not None and self.proc.stdout is not None:
                self.proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                self.proc.stdin.flush()
                self.proc.stdout.readline()
        except Exception:
            pass
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


@dataclass
class AModulePredictions:
    risk_pred: torch.Tensor
    trigger_logit: torch.Tensor
    delta_pred: torch.Tensor
    region_logits: torch.Tensor
    debug_metrics: dict[str, float]
    dynamic_embedding_pred: torch.Tensor | None = None
    region_logits_contract15: torch.Tensor | None = None


class LiteAModuleInterface:
    """Use in-graph lightweight heads for A/corrective predictions."""

    mode = "lite"

    def __init__(
        self,
        *,
        a_risk_head,
        a_trigger_head,
        corrective_delta_head,
        corrective_region_head,
    ) -> None:
        self.a_risk_head = a_risk_head
        self.a_trigger_head = a_trigger_head
        self.corrective_delta_head = corrective_delta_head
        self.corrective_region_head = corrective_region_head

    def predict(
        self,
        *,
        pooled_hidden: torch.Tensor,
        examples: list[dict],
        action_loss: torch.Tensor,
        chunk_len: int,
    ) -> AModulePredictions:
        _ = examples, chunk_len
        pooled_hidden = pooled_hidden.to(dtype=action_loss.dtype)
        return AModulePredictions(
            risk_pred=self.a_risk_head(pooled_hidden).squeeze(-1),
            trigger_logit=self.a_trigger_head(pooled_hidden).squeeze(-1),
            delta_pred=self.corrective_delta_head(pooled_hidden).squeeze(-1),
            region_logits=self.corrective_region_head(pooled_hidden),
            debug_metrics={"debug/a_module_mode_lite": 1.0},
        )


class StandaloneAModuleInterface:
    """Read predictions emitted by an external A module from per-sample payload."""

    mode = "standalone"

    def __init__(
        self,
        *,
        chunk_len: int,
        input_key: str = "a_outputs",
        allow_pseudo_labels_fallback: bool = False,
        strict_missing: bool = False,
        fallback_to_lite=None,
        strict_output_contract: bool = False,
        expected_region_len: int | None = None,
        expected_embedding_dim: int | None = None,
        vdpm_mode: str = "offline",
        vdpm_source: str = "precomputed",
        vdpm_runtime_jsonl: str | None = None,
        vdpm_runtime_infer_enabled: bool = False,
        vdpm_fallback_policy: str = "runtime_then_precomputed",
        vdpm_runtime_python: str | None = None,
        vdpm_repo_root: str | None = None,
        vdpm_device: str = "cuda:0",
        vdpm_worker_log_path: str | None = None,
    ) -> None:
        self.chunk_len = int(chunk_len)
        self.input_key = str(input_key)
        self.allow_pseudo_labels_fallback = bool(allow_pseudo_labels_fallback)
        self.strict_missing = bool(strict_missing)
        self.fallback_to_lite = fallback_to_lite
        self.strict_output_contract = bool(strict_output_contract)
        self.expected_region_len = int(expected_region_len) if expected_region_len else None
        self.expected_embedding_dim = int(expected_embedding_dim) if expected_embedding_dim else None
        self.vdpm_mode = str(vdpm_mode or "offline").strip().lower()
        self.vdpm_source = str(vdpm_source or "precomputed").strip().lower()
        self.vdpm_runtime_jsonl = str(vdpm_runtime_jsonl or "").strip()
        self.vdpm_runtime_infer_enabled = bool(vdpm_runtime_infer_enabled)
        self.vdpm_fallback_policy = str(vdpm_fallback_policy or "runtime_then_precomputed").strip().lower()
        if self.vdpm_fallback_policy not in {"strict_no_precomputed", "runtime_then_precomputed"}:
            self.vdpm_fallback_policy = "runtime_then_precomputed"
        self.vdpm_runtime_python = str(
            vdpm_runtime_python
            or "/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/vdpm/.venv_vdpm_infer/bin/python"
        ).strip()
        self.vdpm_repo_root = str(
            vdpm_repo_root or "/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/vdpm"
        ).strip()
        self.vdpm_device = str(vdpm_device or "cuda:0").strip()
        self.vdpm_worker_log_path = str(
            vdpm_worker_log_path or "/tmp/starvla_vdpm_runtime_worker.log"
        ).strip()
        self.inloop_enabled = self.vdpm_mode == "inloop"
        self.inloop_runtime_lookup_enabled = self.inloop_enabled and self.vdpm_source == "runtime_infer"
        self.inloop_runtime_model_enabled = (
            self.inloop_enabled
            and self.vdpm_source == "runtime_model"
            and self.vdpm_runtime_infer_enabled
        )

        self._inloop_calls = 0
        self._inloop_success = 0
        self._inloop_fail = 0
        self._inloop_fallback_count = 0
        self._inloop_latency_ms: list[float] = []

        self._inloop_runtime_index: dict[tuple[str, int, int], dict] = {}
        self._inloop_runtime_index_loaded = False
        self._inloop_runtime_index_stats = {
            "rows": 0,
            "indexed": 0,
            "duplicates": 0,
            "invalid": 0,
            "invalid_json": 0,
            "invalid_meta": 0,
            "invalid_payload": 0,
        }
        self._inloop_runtime_load_error = ""

        self._inloop_runtime_model_cache: dict[tuple[str, int, int], dict] = {}
        self._vdpm_worker: _VDPMRuntimeModelWorker | None = None
        self._vdpm_model_calls_total = 0
        self._vdpm_cache_hit = 0
        self._vdpm_cache_miss = 0
        self._vdpm_infer_latency_ms: list[float] = []
        self._fallback_to_precomputed_count = 0
        self._rows_seen_for_source = 0
        self._rows_with_precomputed_a_outputs = 0
        self._vdpm_last_error = ""

        if self.inloop_runtime_lookup_enabled and self.vdpm_runtime_jsonl:
            try:
                index, stats = _load_runtime_a_outputs_index(
                    self.vdpm_runtime_jsonl,
                    input_key=self.input_key,
                )
                self._inloop_runtime_index = index
                self._inloop_runtime_index_stats = stats
                self._inloop_runtime_index_loaded = True
            except Exception as exc:
                self._inloop_runtime_load_error = str(exc)

    def _extract_precomputed_source(self, example: dict) -> dict:
        source = example.get(self.input_key)
        if isinstance(source, dict):
            return source
        # Optional fallback for migration. Keep disabled by default to avoid label leakage.
        pseudo = example.get("pseudo_labels")
        if self.allow_pseudo_labels_fallback and isinstance(pseudo, dict):
            return pseudo
        return {}

    def _meta_cache_key(self, example: dict) -> tuple[str, int, int] | None:
        meta = example.get("meta")
        if not isinstance(meta, dict):
            return None
        dataset_name = str(meta.get("dataset_name", "")).strip()
        trajectory_id = _safe_int(meta.get("trajectory_id"))
        sample_step = _safe_int(meta.get("sample_step"))
        if not dataset_name or trajectory_id is None or sample_step is None:
            return None
        return (dataset_name, trajectory_id, sample_step)

    def _lookup_runtime_lookup_source(self, example: dict) -> dict | None:
        if not self.inloop_runtime_lookup_enabled:
            return None
        if not self._inloop_runtime_index_loaded:
            return None
        cache_key = self._meta_cache_key(example)
        if cache_key is None:
            return None
        dataset_name, trajectory_id, sample_step = cache_key
        for alias in _dataset_name_aliases(dataset_name):
            key = (alias, trajectory_id, sample_step)
            payload = self._inloop_runtime_index.get(key)
            if isinstance(payload, dict):
                return payload
        return None

    def _select_runtime_model_frames(self, example: dict) -> list[np.ndarray]:
        views = _normalize_image_views(example.get("image"))
        if not views:
            raise ValueError("example.image missing for runtime_model infer")
        if len(views) == 1:
            return [views[0], views[0]]
        return [views[0], views[-1]]

    def _ensure_vdpm_worker(self) -> _VDPMRuntimeModelWorker:
        if self._vdpm_worker is None:
            self._vdpm_worker = _VDPMRuntimeModelWorker(
                python_bin=self.vdpm_runtime_python,
                vdpm_root=self.vdpm_repo_root,
                device=self.vdpm_device,
                log_path=self.vdpm_worker_log_path,
            )
        return self._vdpm_worker

    def _vdpm_infer_latency_stats(self) -> tuple[float, float]:
        if not self._vdpm_infer_latency_ms:
            return 0.0, 0.0
        values = sorted(self._vdpm_infer_latency_ms)
        mean = sum(values) / len(values)
        p95_idx = max(0, min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1))
        return float(mean), float(values[p95_idx])

    def _infer_runtime_model_payload(self, example: dict) -> dict[str, Any]:
        frames = self._select_runtime_model_frames(example)
        worker = self._ensure_vdpm_worker()
        started = time.perf_counter()
        self._vdpm_model_calls_total += 1
        payload = worker.infer(
            frames=frames,
            version="vdpm_runtime_model_v1",
            source="vdpm/runtime_model",
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if math.isfinite(elapsed_ms) and elapsed_ms >= 0.0:
            self._vdpm_infer_latency_ms.append(float(elapsed_ms))
            if len(self._vdpm_infer_latency_ms) > 4096:
                self._vdpm_infer_latency_ms = self._vdpm_infer_latency_ms[-4096:]
        expected_region_len = self.expected_region_len if self.expected_region_len is not None else _REQUIRED_REGION_LEN
        expected_embed_dim = (
            self.expected_embedding_dim if self.expected_embedding_dim is not None else _REQUIRED_EMBED_LEN
        )
        if not _payload_contract_ok(payload, expected_region_len=expected_region_len, expected_embedding_dim=expected_embed_dim):
            raise ValueError("runtime_model payload contract violation")
        return payload

    def _record_inloop_call(self, elapsed_ms: float, success: bool, fallback: bool) -> None:
        self._inloop_calls += 1
        if success:
            self._inloop_success += 1
        else:
            self._inloop_fail += 1
        if fallback:
            self._inloop_fallback_count += 1
        if math.isfinite(float(elapsed_ms)) and elapsed_ms >= 0.0:
            self._inloop_latency_ms.append(float(elapsed_ms))
            if len(self._inloop_latency_ms) > 4096:
                self._inloop_latency_ms = self._inloop_latency_ms[-4096:]

    def _inloop_latency_stats(self) -> tuple[float, float]:
        if not self._inloop_latency_ms:
            return 0.0, 0.0
        values = sorted(self._inloop_latency_ms)
        mean = sum(values) / len(values)
        p95_idx = max(0, min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1))
        return float(mean), float(values[p95_idx])

    def _extract_source(self, example: dict) -> dict:
        if not self.inloop_enabled:
            return self._extract_precomputed_source(example)

        precomputed_source = self._extract_precomputed_source(example)
        has_precomputed = bool(isinstance(precomputed_source, dict) and precomputed_source)
        self._rows_seen_for_source += 1
        if has_precomputed:
            self._rows_with_precomputed_a_outputs += 1

        started = time.perf_counter()
        success = False
        fallback = False

        source = {}
        if self.inloop_runtime_lookup_enabled:
            runtime_source = self._lookup_runtime_lookup_source(example)
            if isinstance(runtime_source, dict) and runtime_source:
                source = runtime_source
                example[self.input_key] = runtime_source
                success = True
            else:
                source = precomputed_source
                fallback = bool(source)
                if fallback:
                    self._fallback_to_precomputed_count += 1
        elif self.inloop_runtime_model_enabled:
            cache_key = self._meta_cache_key(example)
            if cache_key is not None and cache_key in self._inloop_runtime_model_cache:
                self._vdpm_cache_hit += 1
                source = self._inloop_runtime_model_cache[cache_key]
                example[self.input_key] = source
                success = True
            else:
                self._vdpm_cache_miss += 1
                try:
                    runtime_payload = self._infer_runtime_model_payload(example)
                    source = runtime_payload
                    example[self.input_key] = runtime_payload
                    if cache_key is not None:
                        self._inloop_runtime_model_cache[cache_key] = runtime_payload
                    success = True
                except Exception as exc:
                    self._vdpm_last_error = str(exc)
                    if self.vdpm_fallback_policy == "runtime_then_precomputed" and has_precomputed:
                        source = precomputed_source
                        fallback = True
                        self._fallback_to_precomputed_count += 1
                    else:
                        source = {}
        else:
            source = precomputed_source
            success = bool(source)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not success and not fallback and source:
            success = True
        self._record_inloop_call(elapsed_ms=elapsed_ms, success=success, fallback=fallback)
        return source

    def _inloop_debug_metrics(self) -> dict[str, float]:
        latency_mean, latency_p95 = self._inloop_latency_stats()
        calls_denom = max(1, self._inloop_calls)
        vdpm_latency_mean, vdpm_latency_p95 = self._vdpm_infer_latency_stats()
        cache_denom = max(1, self._vdpm_cache_hit + self._vdpm_cache_miss)
        precomputed_ratio = float(self._rows_with_precomputed_a_outputs / max(1, self._rows_seen_for_source))
        return {
            "inloop_calls": float(self._inloop_calls),
            "inloop_success": float(self._inloop_success),
            "inloop_fail": float(self._inloop_fail),
            "inloop_fallback_count": float(self._inloop_fallback_count),
            "inloop_infer_success_ratio": float(self._inloop_success / calls_denom),
            "inloop_latency_ms_mean": float(latency_mean),
            "inloop_latency_ms_p95": float(latency_p95),
            "vdpm_model_calls_total": float(self._vdpm_model_calls_total),
            "vdpm_cache_hit": float(self._vdpm_cache_hit),
            "vdpm_cache_miss": float(self._vdpm_cache_miss),
            "cache_hit_ratio": float(self._vdpm_cache_hit / cache_denom),
            "vdpm_infer_latency_ms_mean": float(vdpm_latency_mean),
            "vdpm_infer_latency_ms_p95": float(vdpm_latency_p95),
            "fallback_to_precomputed_count": float(self._fallback_to_precomputed_count),
            "rows_with_precomputed_a_outputs_ratio": precomputed_ratio,
            "debug/inloop_mode_enabled": 1.0 if self.inloop_enabled else 0.0,
            "debug/inloop_runtime_source_enabled": 1.0 if self.inloop_runtime_lookup_enabled else 0.0,
            "debug/inloop_runtime_model_enabled": 1.0 if self.inloop_runtime_model_enabled else 0.0,
            "debug/inloop_runtime_index_loaded": 1.0 if self._inloop_runtime_index_loaded else 0.0,
            "debug/inloop_runtime_index_rows": float(self._inloop_runtime_index_stats.get("indexed", 0)),
            "debug/inloop_runtime_index_invalid_rows": float(self._inloop_runtime_index_stats.get("invalid", 0)),
            "debug/inloop_runtime_load_error": 1.0 if self._inloop_runtime_load_error else 0.0,
            "debug/vdpm_runtime_last_error": 1.0 if self._vdpm_last_error else 0.0,
        }

    def collect_runtime_batch(
        self,
        *,
        examples: list[dict],
        device: torch.device,
        dtype: torch.dtype,
        require_embedding: bool = False,
    ) -> tuple[AModuleRuntimeBatch, dict[str, float]]:
        embedding_dim = (
            int(self.expected_embedding_dim)
            if self.expected_embedding_dim is not None and int(self.expected_embedding_dim) > 0
            else _REQUIRED_EMBED_LEN
        )
        batch_size = len(examples)
        dynamic_embedding = torch.zeros((batch_size, embedding_dim), device=device, dtype=dtype)
        embedding_mask = torch.zeros((batch_size,), device=device, dtype=torch.bool)

        rows_with_payload = 0
        rows_with_embedding = 0
        rows_missing_embedding = 0

        for i, example in enumerate(examples):
            source = self._extract_source(example)
            has_payload = bool(isinstance(source, dict) and len(source) > 0)
            if has_payload:
                rows_with_payload += 1
            embedding_raw = source.get("dynamic_embedding") if isinstance(source, dict) else None
            embedding_vec = _coerce_vector(embedding_raw, target_len=embedding_dim)
            if embedding_vec is None:
                rows_missing_embedding += 1
                continue
            dynamic_embedding[i] = embedding_vec.to(device=device, dtype=dtype)
            embedding_mask[i] = True
            rows_with_embedding += 1

        if require_embedding and rows_missing_embedding > 0:
            raise KeyError(
                "Fusion A-module mode requires runtime dynamic_embedding for every row, "
                f"missing={rows_missing_embedding}/{batch_size}."
            )

        runtime_batch = AModuleRuntimeBatch(
            dynamic_embedding=dynamic_embedding,
            embedding_mask=embedding_mask,
            rows_with_payload=rows_with_payload,
            rows_with_embedding=rows_with_embedding,
            rows_missing_embedding=rows_missing_embedding,
            fallback_to_precomputed_count=int(self._fallback_to_precomputed_count),
            inloop_runtime_model_enabled=bool(self.inloop_runtime_model_enabled),
        )
        debug_metrics = self._inloop_debug_metrics()
        debug_metrics["debug/vdpm_embedding_coverage"] = float(rows_with_embedding / max(1, batch_size))
        debug_metrics["debug/inloop_runtime_model_enabled"] = (
            1.0 if runtime_batch.inloop_runtime_model_enabled else 0.0
        )
        debug_metrics["debug/vdpm_fallback_to_lite"] = 0.0
        return runtime_batch, debug_metrics

    def _prediction_from_examples(
        self,
        *,
        examples: list[dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[AModulePredictions, bool]:
        batch_size = len(examples)
        risk_pred = torch.zeros((batch_size,), device=device, dtype=dtype)
        trigger_logit = torch.zeros((batch_size,), device=device, dtype=dtype)
        delta_pred = torch.zeros((batch_size,), device=device, dtype=dtype)
        region_logits = torch.zeros((batch_size, self.chunk_len), device=device, dtype=dtype)

        any_signal = False
        fallback_used = 0
        rows_with_payload = 0
        rows_with_all_fields = 0
        risk_found = 0
        trigger_found = 0
        delta_found = 0
        region_found = 0
        region_input_present = 0
        region_len_mismatch = 0
        embedding_present = 0
        embedding_len_mismatch = 0
        risk_sum = 0.0
        delta_sum = 0.0
        trigger_pos = 0

        for i, example in enumerate(examples):
            source = self._extract_source(example)
            has_payload = bool(isinstance(source, dict) and len(source) > 0)
            if has_payload:
                rows_with_payload += 1
            row_has_all_fields = True

            if has_payload:
                embedding_raw = source.get("dynamic_embedding")
                if isinstance(embedding_raw, (list, tuple)):
                    embedding_present += 1
                    if self.expected_embedding_dim is not None and len(embedding_raw) != self.expected_embedding_dim:
                        embedding_len_mismatch += 1
                        if self.strict_output_contract:
                            raise ValueError(
                                "Standalone A output contract violation: "
                                f"dynamic_embedding len={len(embedding_raw)} "
                                f"!= expected_embedding_dim={self.expected_embedding_dim}"
                            )

            risk_value = _safe_float(source.get("risk_pred", source.get("risk_score")))
            if risk_value is not None:
                risk_pred[i] = risk_value
                any_signal = True
                risk_found += 1
                risk_sum += float(risk_value)
            else:
                fallback_used += 1
                row_has_all_fields = False

            trigger_logit_value = _safe_float(source.get("trigger_logit"))
            if trigger_logit_value is None:
                trigger_prob = _safe_float(source.get("trigger_prob", source.get("trigger_label")))
                if trigger_prob is not None:
                    prob = max(1e-6, min(1.0 - 1e-6, trigger_prob))
                    trigger_logit_value = float(torch.logit(torch.tensor(prob)).item())
            if trigger_logit_value is not None:
                trigger_logit[i] = trigger_logit_value
                any_signal = True
                trigger_found += 1
                if float(trigger_logit_value) > 0.0:
                    trigger_pos += 1
            else:
                fallback_used += 1
                row_has_all_fields = False

            delta_value = _safe_float(
                source.get("delta_pred", source.get("delta_action_norm", source.get("delta_norm")))
            )
            if delta_value is not None:
                delta_pred[i] = delta_value
                any_signal = True
                delta_found += 1
                delta_sum += float(delta_value)
            else:
                fallback_used += 1
                row_has_all_fields = False

            region_raw = source.get(
                "region_logits",
                source.get("region_prior_pred", source.get("affected_region_prior")),
            )
            if isinstance(region_raw, (list, tuple)):
                region_input_present += 1
                expected_region_len = self.expected_region_len if self.expected_region_len is not None else self.chunk_len
                if len(region_raw) != expected_region_len:
                    region_len_mismatch += 1
                    if self.strict_output_contract:
                        raise ValueError(
                            "Standalone A output contract violation: "
                            f"region len={len(region_raw)} != expected_region_len={expected_region_len}"
                        )
            region_vec = _coerce_vector(region_raw, target_len=self.chunk_len)
            if region_vec is not None:
                region_logits[i] = region_vec.to(device=device, dtype=dtype)
                any_signal = True
                region_found += 1
            else:
                fallback_used += 1
                row_has_all_fields = False

            if row_has_all_fields:
                rows_with_all_fields += 1

        denom = max(1, batch_size)
        trigger_rate_denom = max(1, trigger_found)
        region_present_denom = max(1, region_input_present)
        embedding_present_denom = max(1, embedding_present)
        debug_metrics = {
            "debug/a_module_mode_standalone": 1.0,
            "debug/a_module_external_signal_found": 1.0 if any_signal else 0.0,
            "debug/a_module_external_missing_fields": float(fallback_used),
            "debug/a_module_external_rows_with_payload_ratio": float(rows_with_payload / denom),
            "debug/a_module_external_rows_with_all_fields_ratio": float(rows_with_all_fields / denom),
            "debug/a_module_external_risk_coverage": float(risk_found / denom),
            "debug/a_module_external_trigger_coverage": float(trigger_found / denom),
            "debug/a_module_external_delta_coverage": float(delta_found / denom),
            "debug/a_module_external_region_coverage": float(region_found / denom),
            "debug/a_module_external_risk_mean": float(risk_sum / max(1, risk_found)),
            "debug/a_module_external_delta_mean": float(delta_sum / max(1, delta_found)),
            "debug/a_module_external_trigger_pos_rate": float(trigger_pos / trigger_rate_denom),
            "debug/a_module_external_region_input_present_ratio": float(region_input_present / denom),
            "debug/a_module_external_region_len_mismatch_ratio": float(region_len_mismatch / region_present_denom),
            "debug/a_module_external_embedding_present_ratio": float(embedding_present / denom),
            "debug/a_module_external_embedding_len_mismatch_ratio": float(
                embedding_len_mismatch / embedding_present_denom
            ),
        }
        debug_metrics.update(self._inloop_debug_metrics())
        return (
            AModulePredictions(
                risk_pred=risk_pred,
                trigger_logit=trigger_logit,
                delta_pred=delta_pred,
                region_logits=region_logits,
                debug_metrics=debug_metrics,
            ),
            any_signal,
        )

    def predict(
        self,
        *,
        pooled_hidden: torch.Tensor,
        examples: list[dict],
        action_loss: torch.Tensor,
        chunk_len: int,
    ) -> AModulePredictions:
        _ = pooled_hidden, chunk_len
        predictions, any_signal = self._prediction_from_examples(
            examples=examples,
            device=action_loss.device,
            dtype=action_loss.dtype,
        )
        if any_signal:
            return predictions

        if self.fallback_to_lite is not None:
            lite_predictions = self.fallback_to_lite.predict(
                pooled_hidden=pooled_hidden,
                examples=examples,
                action_loss=action_loss,
                chunk_len=chunk_len,
            )
            lite_predictions.debug_metrics.update(predictions.debug_metrics)
            lite_predictions.debug_metrics["debug/a_module_standalone_fallback_to_lite"] = 1.0
            return lite_predictions

        if self.strict_missing:
            inloop_hint = ""
            if self.inloop_enabled:
                last_error = f"; last_runtime_error={self._vdpm_last_error}" if self._vdpm_last_error else ""
                inloop_hint = (
                    " "
                    f"inloop(mode={self.vdpm_mode}, source={self.vdpm_source}, "
                    f"fallback_policy={self.vdpm_fallback_policy}){last_error}"
                )
            raise KeyError(
                f"Standalone A module expects key={self.input_key!r} in examples, "
                f"but no usable prediction fields were found.{inloop_hint}"
            )
        return predictions


class FusionAModuleInterface(StandaloneAModuleInterface):
    """Runtime feature provider for graph-internal trainable fusion A-head."""

    mode = "fusion"


def build_a_module_interface(
    *,
    config,
    chunk_len: int,
    a_risk_head,
    a_trigger_head,
    corrective_delta_head,
    corrective_region_head,
):
    """Factory for A-module prediction interface.

    Modes:
      - lite: use in-model lightweight heads (current default behavior)
      - standalone: read external A predictions from sample payload
      - fusion: collect in-loop runtime features + keep external contract path available
    """
    framework_cfg = getattr(config, "framework", None)
    a_cfg = _cfg_get(framework_cfg, "a_module", None)
    mode = str(_cfg_get(a_cfg, "mode", "lite")).strip().lower()

    lite_interface = LiteAModuleInterface(
        a_risk_head=a_risk_head,
        a_trigger_head=a_trigger_head,
        corrective_delta_head=corrective_delta_head,
        corrective_region_head=corrective_region_head,
    )
    if mode == "lite":
        return lite_interface

    if mode in {"standalone", "fusion"}:
        standalone_cfg = _cfg_get(a_cfg, "standalone", None)
        input_key = str(_cfg_get(standalone_cfg, "input_key", "a_outputs"))
        allow_pseudo_labels_fallback = bool(
            _cfg_get(standalone_cfg, "allow_pseudo_labels_fallback", False)
        )
        strict_missing = bool(_cfg_get(standalone_cfg, "strict_missing", False))
        fallback_to_lite = bool(_cfg_get(standalone_cfg, "fallback_to_lite", True))
        contract_cfg = _cfg_get(standalone_cfg, "output_contract", None)
        strict_output_contract = bool(_cfg_get(contract_cfg, "strict_shape", False))
        expected_region_len = _to_int_or_none(_cfg_get(contract_cfg, "expected_region_len", None))
        expected_embedding_dim = _to_int_or_none(_cfg_get(contract_cfg, "expected_embedding_dim", None))
        vdpm_mode = str(_cfg_get(a_cfg, "vdpm_mode", "offline")).strip().lower()
        vdpm_source = str(_cfg_get(a_cfg, "vdpm_source", "precomputed")).strip().lower()
        vdpm_runtime_jsonl = _cfg_get(
            a_cfg,
            "vdpm_runtime_jsonl",
            _cfg_get(standalone_cfg, "vdpm_runtime_jsonl", None),
        )
        vdpm_runtime_infer_enabled = bool(_cfg_get(a_cfg, "vdpm_runtime_infer_enabled", False))
        vdpm_fallback_policy = str(_cfg_get(a_cfg, "vdpm_fallback_policy", "runtime_then_precomputed"))
        vdpm_runtime_python = _cfg_get(a_cfg, "vdpm_runtime_python", None)
        vdpm_repo_root = _cfg_get(a_cfg, "vdpm_repo_root", None)
        vdpm_device = str(_cfg_get(a_cfg, "vdpm_device", "cuda:0"))
        vdpm_worker_log_path = _cfg_get(a_cfg, "vdpm_worker_log_path", None)

        interface_cls = StandaloneAModuleInterface
        if mode == "fusion":
            # Fusion mode is explicitly tied to frozen VDPM runtime-model features.
            interface_cls = FusionAModuleInterface
            vdpm_mode = "inloop"
            vdpm_source = "runtime_model"
            vdpm_runtime_infer_enabled = True
            vdpm_fallback_policy = "strict_no_precomputed"
            strict_missing = True
            fallback_to_lite = False

        return interface_cls(
            chunk_len=chunk_len,
            input_key=input_key,
            allow_pseudo_labels_fallback=allow_pseudo_labels_fallback,
            strict_missing=strict_missing,
            fallback_to_lite=lite_interface if fallback_to_lite else None,
            strict_output_contract=strict_output_contract,
            expected_region_len=expected_region_len,
            expected_embedding_dim=expected_embedding_dim,
            vdpm_mode=vdpm_mode,
            vdpm_source=vdpm_source,
            vdpm_runtime_jsonl=vdpm_runtime_jsonl,
            vdpm_runtime_infer_enabled=vdpm_runtime_infer_enabled,
            vdpm_fallback_policy=vdpm_fallback_policy,
            vdpm_runtime_python=vdpm_runtime_python,
            vdpm_repo_root=vdpm_repo_root,
            vdpm_device=vdpm_device,
            vdpm_worker_log_path=vdpm_worker_log_path,
        )

    raise ValueError(
        f"Unsupported framework.a_module.mode={mode!r}, expected one of ['lite', 'standalone', 'fusion']"
    )
