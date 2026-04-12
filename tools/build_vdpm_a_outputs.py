#!/usr/bin/env python3
from __future__ import annotations

"""Build contract-preserving A-module outputs from real VDPM inference.

Naming note:
- tools/build_fasa_a_outputs.py: heuristic/offline FASA builder.
- tools/build_vdpm_a_outputs.py: real VDPM-backed builder.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import imageio_ffmpeg as ffm
import numpy as np
import torch


REQUIRED_REGION_LEN = 15
REQUIRED_EMBED_LEN = 16
CONTRACT_FIELDS = (
    "risk_pred",
    "trigger_logit",
    "delta_pred",
    "region_logits",
    "dynamic_embedding",
    "version",
    "source",
)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clip01(value: float) -> float:
    return _clip(value, 0.0, 1.0)


def _logit(prob: float) -> float:
    p = _clip(prob, 1e-4, 1.0 - 1e-4)
    return float(math.log(p / (1.0 - p)))


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _all_finite(xs: list[float]) -> bool:
    return all(math.isfinite(float(x)) for x in xs)


def _partition_mean_3x5(matrix: torch.Tensor) -> list[float]:
    h, w = matrix.shape
    h_edges = torch.linspace(0, h, steps=4).round().to(torch.int64)
    w_edges = torch.linspace(0, w, steps=6).round().to(torch.int64)
    out: list[float] = []
    for r in range(3):
        hs = int(h_edges[r].item())
        he = int(h_edges[r + 1].item())
        for c in range(5):
            ws = int(w_edges[c].item())
            we = int(w_edges[c + 1].item())
            patch = matrix[hs:he, ws:we]
            if patch.numel() == 0:
                out.append(0.0)
            else:
                out.append(float(patch.mean().item()))
    if len(out) != REQUIRED_REGION_LEN:
        raise RuntimeError(f"region partition len mismatch: got {len(out)}")
    return out


def _decode_frames_rgb(video_path: str, frame_indices: list[int]) -> tuple[list[np.ndarray], dict[str, Any]]:
    if not frame_indices:
        raise ValueError("frame_indices is empty")
    frame_indices = sorted(set(max(0, int(i)) for i in frame_indices))

    reader = ffm.read_frames(video_path, pix_fmt="rgb24")
    meta = next(reader)
    w, h = int(meta["size"][0]), int(meta["size"][1])

    wanted = set(frame_indices)
    got: dict[int, np.ndarray] = {}
    for idx, buf in enumerate(reader):
        if idx in wanted:
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            got[idx] = arr
            if len(got) == len(wanted):
                break

    missing = [idx for idx in frame_indices if idx not in got]
    if missing:
        raise RuntimeError(f"missing frames {missing} for video={video_path}")

    return [got[idx] for idx in frame_indices], meta


def _load_vdpm(vdpm_root: str, device: str):
    vdpm_root_path = Path(vdpm_root).expanduser().resolve()
    if not vdpm_root_path.exists():
        raise FileNotFoundError(f"vdpm_root not found: {vdpm_root_path}")

    sys.path.insert(0, str(vdpm_root_path))
    from hydra import compose, initialize_config_dir  # type: ignore
    from visualise import load_model, preprocess_images  # type: ignore

    cfg_dir = str(vdpm_root_path / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="visualise")
    model = load_model(cfg, device)
    return model, preprocess_images


def _build_contract_from_pred(pred: dict[str, Any], num_frames: int, version: str, source: str) -> dict[str, Any]:
    if "pointmaps" not in pred or not pred["pointmaps"]:
        raise RuntimeError("pred.pointmaps missing")
    pointmaps = pred["pointmaps"]

    pm = pointmaps[-1]
    pts = pm["pts3d"]  # [1, S, H, W, 3]
    conf = pm["conf"]  # [1, S, H, W]
    if pts.ndim != 5 or conf.ndim != 4:
        raise RuntimeError(f"unexpected shapes pts={tuple(pts.shape)} conf={tuple(conf.shape)}")

    s = int(pts.shape[1])
    cur_idx = s - 1
    prev_idx = max(0, s - 2)

    pts_cur = pts[0, cur_idx].float().detach().cpu()
    pts_prev = pts[0, prev_idx].float().detach().cpu()
    conf_cur = conf[0, cur_idx].float().detach().cpu()

    delta_map = torch.linalg.norm(pts_cur - pts_prev, dim=-1)  # [H, W]
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

    # Contract scalar heads.
    risk_pred = _clip01(0.6 * (delta_q90 / (delta_q90 + 1.0)) + 0.4 * uncertainty)
    trigger_prob = _clip01(delta_q90 / (delta_q90 + 0.25))
    trigger_logit = _logit(trigger_prob)
    delta_pred = max(0.0, delta_mean)

    # Region logits (len=15): 3x5 partitions over displacement map.
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

    payload = {
        "risk_pred": float(risk_pred),
        "trigger_logit": float(trigger_logit),
        "delta_pred": float(delta_pred),
        "region_logits": [float(x) for x in region_logits],
        "dynamic_embedding": [float(x) for x in dynamic_embedding],
        "version": version,
        "source": source,
    }
    return payload


def _payload_contract_ok(payload: dict[str, Any]) -> tuple[bool, str]:
    for k in CONTRACT_FIELDS:
        if k not in payload:
            return False, f"missing:{k}"
    if not _is_finite(payload["risk_pred"]):
        return False, "risk_pred_nonfinite"
    if not _is_finite(payload["trigger_logit"]):
        return False, "trigger_logit_nonfinite"
    if not _is_finite(payload["delta_pred"]) or float(payload["delta_pred"]) < 0.0:
        return False, "delta_pred_nonfinite_or_negative"

    region = payload["region_logits"]
    if not isinstance(region, list) or len(region) != REQUIRED_REGION_LEN or not _all_finite(region):
        return False, "region_logits_invalid"
    embed = payload["dynamic_embedding"]
    if not isinstance(embed, list) or len(embed) != REQUIRED_EMBED_LEN or not _all_finite(embed):
        return False, "dynamic_embedding_invalid"
    if not isinstance(payload["version"], str) or not payload["version"]:
        return False, "version_invalid"
    if not isinstance(payload["source"], str) or not payload["source"]:
        return False, "source_invalid"
    return True, ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build real VDPM-backed a_outputs JSONL (contract-preserving).")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--sample-check-json", default=None)
    p.add_argument("--vdpm-root", required=True, help="Path to vdpm repository root.")
    p.add_argument("--libero-root", required=True, help="Path to LIBERO root with dataset_name subdirs.")
    p.add_argument("--camera-view", default="observation.images.image", help="Video subdir view key.")
    p.add_argument("--frame-offsets", default="-1,0", help="Comma offsets around sample_step, e.g. -1,0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--version", default="vdpm_v1")
    p.add_argument("--source", default="vdpm/real")
    p.add_argument("--strict-missing", action="store_true")
    p.add_argument("--max-print-errors", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()
    sample_path = Path(args.sample_check_json).expanduser().resolve() if args.sample_check_json else None
    libero_root = Path(args.libero_root).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_path}")
    if not libero_root.exists():
        raise FileNotFoundError(f"libero_root not found: {libero_root}")

    frame_offsets = []
    for token in args.frame_offsets.split(","):
        token = token.strip()
        if token:
            frame_offsets.append(int(token))
    if not frame_offsets:
        frame_offsets = [-1, 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_path:
        sample_path.parent.mkdir(parents=True, exist_ok=True)

    model, preprocess_images = _load_vdpm(args.vdpm_root, args.device)
    model.eval()

    rows_total = 0
    rows_ok = 0
    rows_failed = 0
    rows_contract_ok = 0
    rows_contract_bad = 0
    failure_counts: Counter[str] = Counter()
    error_examples: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    infer_seconds_total = 0.0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            text = line.strip()
            if not text:
                continue
            rows_total += 1
            row = json.loads(text)
            meta = row.get("meta", {}) if isinstance(row, dict) else {}

            try:
                dataset_name = str(meta["dataset_name"])
                trajectory_id = int(meta["trajectory_id"])
                sample_step = int(meta["sample_step"])
                video_path = (
                    libero_root
                    / dataset_name
                    / "videos"
                    / "chunk-000"
                    / args.camera_view
                    / f"episode_{trajectory_id:06d}.mp4"
                )
                if not video_path.exists():
                    raise FileNotFoundError(f"video missing: {video_path}")

                target_frames = [max(0, sample_step + off) for off in frame_offsets]
                frames, decode_meta = _decode_frames_rgb(str(video_path), target_frames)
                if len(frames) < 1:
                    raise RuntimeError("decoded frame list empty")

                start_t = time.perf_counter()
                images = preprocess_images(frames).to(args.device)
                with torch.no_grad():
                    pred = model.inference(None, images=images.unsqueeze(0))
                infer_sec = time.perf_counter() - start_t
                infer_seconds_total += infer_sec

                payload = _build_contract_from_pred(
                    pred=pred,
                    num_frames=len(frames),
                    version=args.version,
                    source=args.source,
                )
                ok, reason = _payload_contract_ok(payload)
                if ok:
                    rows_contract_ok += 1
                else:
                    rows_contract_bad += 1
                    raise RuntimeError(f"contract_check_failed:{reason}")

                out_row = dict(row)
                out_row["a_outputs"] = payload
                out_row["a_outputs_meta"] = {
                    "generator": "vdpm_real",
                    "vdpm_root": str(Path(args.vdpm_root).expanduser().resolve()),
                    "camera_view": args.camera_view,
                    "video_path": str(video_path),
                    "frame_indices": [int(x) for x in target_frames],
                    "decode_codec": decode_meta.get("codec"),
                    "decode_fps": decode_meta.get("fps"),
                    "infer_seconds": infer_sec,
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                rows_ok += 1

                if len(sample_rows) < max(1, args.max_samples):
                    sample_rows.append(
                        {
                            "line_no": line_no,
                            "dataset_name": dataset_name,
                            "trajectory_id": trajectory_id,
                            "sample_step": sample_step,
                            "video_path": str(video_path),
                            "frame_indices": [int(x) for x in target_frames],
                            "risk_pred": payload["risk_pred"],
                            "trigger_logit": payload["trigger_logit"],
                            "delta_pred": payload["delta_pred"],
                            "region_logits_len": len(payload["region_logits"]),
                            "dynamic_embedding_len": len(payload["dynamic_embedding"]),
                            "version": payload["version"],
                            "source": payload["source"],
                        }
                    )

            except Exception as exc:  # noqa: BLE001
                rows_failed += 1
                reason = str(exc)
                failure_key = reason.split(":", 1)[0]
                failure_counts[failure_key] += 1
                if len(error_examples) < max(1, args.max_print_errors):
                    error_examples.append(
                        {
                            "line_no": line_no,
                            "error": reason,
                            "meta": meta,
                        }
                    )

    rows_ok_ratio = float(rows_ok / max(1, rows_total))
    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "vdpm_root": str(Path(args.vdpm_root).expanduser().resolve()),
        "libero_root": str(libero_root),
        "camera_view": args.camera_view,
        "frame_offsets": [int(x) for x in frame_offsets],
        "rows_total": rows_total,
        "rows_ok": rows_ok,
        "rows_ok_ratio": rows_ok_ratio,
        "rows_failed": rows_failed,
        "rows_contract_ok": rows_contract_ok,
        "rows_contract_bad": rows_contract_bad,
        "failure_counts": dict(failure_counts),
        "error_examples": error_examples,
        "infer_seconds_total": infer_seconds_total,
        "infer_seconds_avg": (infer_seconds_total / max(1, rows_ok)),
        "strict_missing_enabled": bool(args.strict_missing),
        "strict_missing_pass": rows_failed == 0,
        "contract": {
            "required_fields": list(CONTRACT_FIELDS),
            "region_logits_len": REQUIRED_REGION_LEN,
            "dynamic_embedding_len": REQUIRED_EMBED_LEN,
            "version": args.version,
            "source": args.source,
        },
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if sample_path:
        sample_path.write_text(json.dumps({"sample_rows": sample_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.strict_missing and rows_failed > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
